import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video


def load_prompts(path: Path, limit: int) -> list[dict]:
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt") or item.get("text")
            if not prompt:
                raise ValueError(f"Missing prompt/text in {item}")
            prompts.append(
                {
                    "prompt_index": int(item.get("prompt_index", len(prompts) + 1)),
                    "id": str(item.get("id", f"prompt_{len(prompts) + 1:03d}")),
                    "prompt": str(prompt),
                }
            )
            if limit > 0 and len(prompts) >= limit:
                break
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--prompt-count", type=int, default=50)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--distilled-checkpoint", default="/keyan/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors")
    parser.add_argument("--gemma-root", default="/keyan/LTX-2.3/gemma")
    parser.add_argument("--spatial-upsampler", default="/keyan/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompt_file), args.prompt_count)
    outdir = Path(args.outdir) / args.label
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / f"{args.label}_hot_batch_summary.json"
    csv_path = outdir / f"{args.label}_hot_batch_timing.csv"

    loras = []
    lora_path = Path(args.lora_path) if args.lora_path else None
    if lora_path is not None:
        if not lora_path.exists():
            raise FileNotFoundError(lora_path)
        loras.append(LoraPathStrengthAndSDOps(str(lora_path), args.lora_scale, LTXV_LORA_COMFY_RENAMING_MAP))

    print("HOT_BATCH_START", args.label, "prompt_count", len(prompts), flush=True)
    construct_start = time.perf_counter()
    pipe = DistilledPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint,
        spatial_upsampler_path=args.spatial_upsampler,
        gemma_root=args.gemma_root,
        loras=loras,
    )
    construct_sec = time.perf_counter() - construct_start
    tiling = TilingConfig.default()
    chunks = get_video_chunks_number(args.frames, tiling)
    rows = []

    with torch.inference_mode():
        for item in prompts:
            i = item["prompt_index"]
            seed = 1000 + i - 1
            outfile = outdir / f"prompt{i:03d}_{args.label}_{args.width}x{args.height}x{args.frames}.mp4"
            if outfile.exists() and outfile.stat().st_size > 0:
                print("GENERATE_SKIP", args.label, i, outfile, flush=True)
                rows.append(
                    {
                        "label": args.label,
                        "prompt_index": i,
                        "id": item["id"],
                        "seed": seed,
                        "path": str(outfile),
                        "infer_sec": None,
                        "encode_sec": None,
                        "file_size": outfile.stat().st_size,
                    }
                )
                continue
            print("GENERATE_START", args.label, i, item["id"], outfile, flush=True)
            start = time.perf_counter()
            video, audio = pipe(
                prompt=item["prompt"],
                seed=seed,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                frame_rate=args.fps,
                images=[],
                tiling_config=tiling,
                enhance_prompt=False,
            )
            infer_sec = time.perf_counter() - start
            encode_start = time.perf_counter()
            encode_video(video=video, fps=args.fps, audio=audio, output_path=str(outfile), video_chunks_number=chunks)
            encode_sec = time.perf_counter() - encode_start
            row = {
                "label": args.label,
                "prompt_index": i,
                "id": item["id"],
                "seed": seed,
                "path": str(outfile),
                "infer_sec": infer_sec,
                "encode_sec": encode_sec,
                "file_size": outfile.stat().st_size,
            }
            rows.append(row)
            print("GENERATE_DONE", json.dumps(row, ensure_ascii=False), flush=True)
            del video, audio
            gc.collect()
            torch.cuda.empty_cache()

    summary = {
        "label": args.label,
        "construct_sec": construct_sec,
        "prompt_file": args.prompt_file,
        "lora_path": str(lora_path) if lora_path else None,
        "lora_scale": args.lora_scale,
        "rows": rows,
    }
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("HOT_BATCH_DONE", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
