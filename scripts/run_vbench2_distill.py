#!/usr/bin/env python3
"""
使用 LTX-2.3 DistilledPipeline,按 VBench-2.0 提示词分维度批量生成评测所需视频。

本脚本由原 2-Stage 评测脚本(run_vbench2_ltx2_zwq.py)改写而来,完整保留其经过验证的:
  - 全局提示词跨维度去重:重叠 prompt 在 GPU 上只渲染一次,再瞬间复制分发
  - 断点续传/自愈:已存在则跳过,缺失的复用副本自动补全
  - seed 公式:seed = (base_seed + prompt_idx * 100 + index) % 2**32
  - 采样数:Diversity 维度 20 个样本(index 0~19),其余维度 3 个样本(index 0~2)
  - 子维度存储:{output_dir}/{dimension}/{prompt[:180]}-{index}.mp4
  - 多卡 / 每卡多进程并发(spawn)

与 2-Stage 的关键差异(因为换成了 DistilledPipeline):
  - 流水线类:DistilledPipeline 取代 TI2VidTwoStagesPipeline
  - DistilledPipeline 用固定蒸馏 sigmas(8+4 步),因此不接受
    negative_prompt / num_inference_steps / cfg_scale / stg_scale / guider 参数
  - 加载我们自训的 LoRA(config.model.loras),sd_ops 用 LTXV_LORA_COMFY_RENAMING_MAP
  - 不再依赖私有 ltx2_inference_config_zwq 模块;ltx_pipelines 直接从已安装的 venv 导入

用法:
  python run_vbench2_distill.py --config config_distill_vbench2.yaml            # 单进程
  python run_vbench2_distill.py --config config_distill_vbench2.yaml --gpus 0,1,2,3 --per-gpu-workers 1
  # 不传 --gpus 时,读取 config.vbench2.parallel.gpus("auto" / "0,1" / 留空)
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
from pathlib import Path

import yaml

# VBench-2.0 全部 18 个维度
ALL_DIMS = [
    "Human_Anatomy", "Human_Identity", "Human_Clothes", "Diversity", "Composition",
    "Dynamic_Spatial_Relationship", "Dynamic_Attribute", "Motion_Order_Understanding",
    "Human_Interaction", "Complex_Landscape", "Complex_Plot", "Camera_Motion",
    "Motion_Rationality", "Instance_Preservation", "Mechanics", "Thermotics",
    "Material", "Multi-View_Consistency",
]


# --------------------------------------------------------------------------- #
# 配置加载
# --------------------------------------------------------------------------- #
def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_run_context(config_path: Path):
    raw = load_config(config_path)
    vb = raw.get("vbench2") or {}

    prompts_dir = Path(vb["prompts_dir"]).expanduser().resolve()
    if not prompts_dir.is_dir():
        raise FileNotFoundError(f"未找到 VBench-2.0 prompts 目录: {prompts_dir}")

    aug_raw = vb.get("prompts_aug_dir")
    prompts_aug_dir = Path(aug_raw).expanduser().resolve() if aug_raw else None

    out_dir = Path(vb["output_dir"]).expanduser().resolve()
    base_seed = int(vb.get("base_seed", 42))
    limit_prompts = int(vb.get("limit_prompts", 0))

    dims = vb.get("dimensions") or []
    if not dims:
        dims = ALL_DIMS
    else:
        for d in dims:
            if d not in ALL_DIMS:
                raise ValueError(f"无效的 VBench-2.0 维度: {d}. 可选: {ALL_DIMS}")

    return raw, dims, prompts_dir, prompts_aug_dir, out_dir, base_seed, limit_prompts


def _parse_gpus(gpus_arg: str | None, parallel_cfg: dict | None) -> list[int] | None:
    val = None
    if gpus_arg is not None and str(gpus_arg).strip():
        val = str(gpus_arg).strip()
    elif parallel_cfg and parallel_cfg.get("gpus") is not None:
        g = parallel_cfg["gpus"]
        if isinstance(g, list):
            return [int(x) for x in g]
        val = str(g).strip()

    if not val:
        return None
    if val.lower() == "auto":
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 1
        return list(range(n))
    try:
        return [int(x.strip()) for x in val.split(",") if x.strip()]
    except ValueError:
        return None


def _parse_per_gpu_workers(cli_val: int | None, parallel_cfg: dict | None) -> int:
    if cli_val is not None and cli_val >= 1:
        return int(cli_val)
    if parallel_cfg and parallel_cfg.get("per_gpu_workers") is not None:
        return max(1, int(parallel_cfg["per_gpu_workers"]))
    return 1


# --------------------------------------------------------------------------- #
# 任务清单构建(扫描 prompt + 跨维度去重) —— 与原 2-Stage 脚本逻辑一致
# --------------------------------------------------------------------------- #
def _build_task_list(dims, prompts_dir, prompts_aug_dir, limit_prompts):
    prompt_to_first = {}  # (prompt, index) -> (dimension, gen_prompt, prompt_idx)
    task_map = {}         # (first_dim, prompt, index) -> [其它重合维度]

    for dimension in dims:
        txt_file = prompts_dir / f"{dimension}.txt"
        if not txt_file.is_file():
            logging.warning("维度提示词文件未找到,跳过: %s", txt_file)
            continue
        with open(txt_file, encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]

        prompts_aug = []
        if prompts_aug_dir:
            aug_file = prompts_aug_dir / f"{dimension}.txt"
            if aug_file.is_file():
                with open(aug_file, encoding="utf-8") as f:
                    prompts_aug = [ln.strip() for ln in f if ln.strip()]

        if limit_prompts > 0:
            prompts = prompts[:limit_prompts]
            if prompts_aug:
                prompts_aug = prompts_aug[:limit_prompts]

        for prompt_idx, prompt in enumerate(prompts):
            iter_count = 20 if dimension == "Diversity" else 3
            gen_prompt = prompt
            if prompts_aug and prompt_idx < len(prompts_aug):
                gen_prompt = prompts_aug[prompt_idx]

            for index in range(iter_count):
                key = (prompt, index)
                if key not in prompt_to_first:
                    prompt_to_first[key] = (dimension, gen_prompt, prompt_idx)
                    task_map[(dimension, prompt, index)] = []
                else:
                    first_dim, _, _ = prompt_to_first[key]
                    task_map[(first_dim, prompt, index)].append(dimension)

    all_tasks = []
    for (dimension, prompt, index), other_dims in task_map.items():
        _, gen_prompt, prompt_idx = prompt_to_first[(prompt, index)]
        all_tasks.append((dimension, prompt, gen_prompt, prompt_idx, index, other_dims))
    return all_tasks


def _filter_done(all_tasks, out_dir):
    active = []
    for task in all_tasks:
        dimension, prompt, _, _, index, other_dims = task
        filename = f"{prompt[:180]}-{index}.mp4"
        main_exist = (out_dir / dimension / filename).is_file()
        all_exist = main_exist and all((out_dir / od / filename).is_file() for od in other_dims)
        if not all_exist:
            active.append(task)
    return active


# --------------------------------------------------------------------------- #
# worker:在设置 CUDA_VISIBLE_DEVICES 之后再 import torch / 构建流水线
# --------------------------------------------------------------------------- #
def _worker_entry(payload: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(payload["cuda_device"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    worker_id = payload["worker_id"]
    log = logging.getLogger(f"distill.vbench2.w{worker_id}")
    cfg_path = Path(payload["config_path"])
    tasks = payload["tasks"]

    import torch
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.helpers import get_device
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.quantization_factory import QuantizationKind

    raw, _, _, _, out_dir, base_seed, _ = _load_run_context(cfg_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = raw["params"]
    m = raw["model"]

    # --- 组装 LoRA 列表(顺序与强度严格按 config) ---
    loras = []
    for lora_cfg in (m.get("loras") or []):
        loras.append(
            LoraPathStrengthAndSDOps(
                path=lora_cfg["path"],
                strength=float(lora_cfg.get("strength", 1.0)),
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        )

    # --- 量化策略(可选);fp8-scaled-mm 需要 checkpoint 路径以读取 scale ---
    quant_str = m.get("quantization")
    quantization = None
    if quant_str:
        quantization = QuantizationKind(quant_str).to_policy(
            checkpoint_path=m["distilled_checkpoint_path"]
        )

    device = get_device()
    log.info(
        "[worker %s cuda=%s] 加载 DistilledPipeline (LoRA=%d, Quant=%s),待生成 %d 个样本",
        worker_id, payload["cuda_device"], len(loras), quant_str or "None", len(tasks),
    )

    pipeline = DistilledPipeline(
        distilled_checkpoint_path=m["distilled_checkpoint_path"],
        spatial_upsampler_path=m["spatial_upsampler_path"],
        gemma_root=m["gemma_root"],
        loras=loras,
        device=device,
        quantization=quantization,
    )

    tiling_config = TilingConfig.default()
    video_chunks = get_video_chunks_number(p["num_frames"], tiling_config)

    for task_idx, (dimension, prompt, gen_prompt, prompt_idx, index, other_dims) in enumerate(tasks):
        dim_dir = out_dir / dimension
        dim_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{prompt[:180]}-{index}.mp4"
        mp4_path = dim_dir / filename

        # 断点自愈:已存在则只补全缺失的复用副本
        if mp4_path.is_file():
            for od in other_dims:
                op = out_dir / od / filename
                if not op.is_file():
                    op.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(mp4_path, op)
                    log.info("[worker %s] 补全复用副本 -> %s/%s", worker_id, od, filename)
            log.info("[worker %s] 已存在,跳过 %s/%s", worker_id, dimension, filename)
            continue

        seed = (base_seed + prompt_idx * 100 + index) % (2**32)
        log.info(
            "[worker %s] [%d/%d] 生成 %s/%s | seed=%s",
            worker_id, task_idx + 1, len(tasks), dimension, filename, seed,
        )

        with tempfile.TemporaryDirectory(prefix="distill_vbench2_") as tmp:
            tmp_mp4 = Path(tmp) / "clip.mp4"
            with torch.inference_mode():
                video, audio = pipeline(
                    prompt=gen_prompt,
                    seed=seed,
                    height=p["height"],
                    width=p["width"],
                    num_frames=p["num_frames"],
                    frame_rate=p["frame_rate"],
                    images=[],
                    tiling_config=tiling_config,
                    enhance_prompt=p.get("enhance_prompt", False),
                )
                encode_video(
                    video=video,
                    fps=p["frame_rate"],
                    audio=audio,
                    output_path=str(tmp_mp4),
                    video_chunks_number=video_chunks,
                )
            shutil.move(str(tmp_mp4), str(mp4_path))

            # 瞬间复制分发到重合维度
            for od in other_dims:
                op = out_dir / od / filename
                op.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mp4_path, op)
                log.info("[worker %s] 复用分发 -> %s/%s", worker_id, od, filename)

        del video, audio
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    log.info("[worker %s] 本进程任务结束", worker_id)


# --------------------------------------------------------------------------- #
# 调度
# --------------------------------------------------------------------------- #
def run_single_process(config_path: Path) -> None:
    raw, dims, prompts_dir, prompts_aug_dir, out_dir, base_seed, limit_prompts = _load_run_context(config_path)
    all_tasks = _build_task_list(dims, prompts_dir, prompts_aug_dir, limit_prompts)
    active = _filter_done(all_tasks, out_dir)
    logging.info("排重后共 %d 组,仍需生成 %d 个唯一样本。", len(all_tasks), len(active))
    if not active:
        logging.info("所有视频已生成完毕,无需继续。")
        return
    _worker_entry({
        "cuda_device": "0",
        "worker_id": 0,
        "config_path": str(config_path.resolve()),
        "tasks": active,
    })


def run_parallel(config_path: Path, gpu_ids: list[int], per_gpu_workers: int) -> None:
    raw, dims, prompts_dir, prompts_aug_dir, out_dir, base_seed, limit_prompts = _load_run_context(config_path)
    all_tasks = _build_task_list(dims, prompts_dir, prompts_aug_dir, limit_prompts)
    active = _filter_done(all_tasks, out_dir)
    logging.info("排重后共 %d 组,仍需生成 %d 个唯一样本。", len(all_tasks), len(active))
    if not active:
        logging.info("所有视频已生成完毕,无需继续。")
        return

    num_workers = len(gpu_ids) * per_gpu_workers
    buckets = [[] for _ in range(num_workers)]
    for i, task in enumerate(active):
        buckets[i % num_workers].append(task)

    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    wid = 0
    for gpu in gpu_ids:
        for _slot in range(per_gpu_workers):
            tasks = buckets[wid]
            if tasks:
                payload = {
                    "cuda_device": str(gpu),
                    "worker_id": wid,
                    "config_path": str(config_path.resolve()),
                    "tasks": tasks,
                }
                pr = ctx.Process(target=_worker_entry, args=(payload,), name=f"distill-vbench2-{wid}")
                pr.start()
                procs.append(pr)
            wid += 1

    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"子进程 {pr.name} 异常退出,exitcode={pr.exitcode}")

    logging.info("全部 worker 完成,视频生成完毕!输出根目录: %s", out_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="DistilledPipeline 批量生成 VBench-2.0 评测视频")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置,例如 config_distill_vbench2.yaml")
    parser.add_argument("--gpus", type=str, default=None, help="物理 GPU 编号,逗号分隔(如 0,1,2,3);不设则读 config 或单进程")
    parser.add_argument("--per-gpu-workers", type=int, default=None, help="每张卡的进程数,默认 1")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    raw = load_config(cfg_path)
    parallel_cfg = (raw.get("vbench2") or {}).get("parallel") or {}

    gpu_list = _parse_gpus(args.gpus, parallel_cfg)
    per_gpu = _parse_per_gpu_workers(args.per_gpu_workers, parallel_cfg)

    if not gpu_list:
        run_single_process(cfg_path)
        return

    logging.info("并行模式: GPUs=%s,每卡进程=%s,总并发=%s", gpu_list, per_gpu, len(gpu_list) * per_gpu)
    run_parallel(cfg_path, gpu_list, per_gpu)


if __name__ == "__main__":
    main()
