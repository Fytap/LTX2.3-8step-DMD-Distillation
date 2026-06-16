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


PROMPTS = [
    "A man in a high-visibility vest is operating a machine in a workshop. He stands in front of a blue metal frame with a purple and white machine on it, surrounded by various tools and equipment such as a red container and some metal pipes. The workshop has a concrete floor with some stains and a door in the background. Music plays in the background throughout the scene.",
    "A character in an anime style is shown, bald and wearing a red and yellow checkered shirt with a black design and black pants. They are holding a sword in a fighting stance against a mountainous landscape with trees and a cloudy sky. Clouds of dust or smoke surround the character, and there are sound effects of objects being overturned, a whooshing noise, and bursts of energy.",
    "A bustling market scene is shown with several people, some wearing masks, gathered around a table filled with various fruits and drinks. The table has a green surface and is covered with plastic cups containing different colored liquids, likely fruit juices or smoothies. Containers of fruits like watermelon and mangoes are also present. People are talking, and there is a busy atmosphere with more stalls and individuals walking around in the background.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--distilled-checkpoint", default="/keyan/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors")
    parser.add_argument("--gemma-root", default="/keyan/LTX-2.3/gemma")
    parser.add_argument("--spatial-upsampler", default="/keyan/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--prompt-count", type=int, default=3)
    args = parser.parse_args()

    outdir = Path(args.outdir) / args.label
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / f"{args.label}_hot_batch_summary.json"
    csv_path = outdir / f"{args.label}_hot_batch_timing.csv"

    loras = []
    lora_path = Path(args.lora_path) if args.lora_path else None
    if lora_path is not None:
        if not lora_path.exists():
            raise FileNotFoundError(lora_path)
        loras.append(
            LoraPathStrengthAndSDOps(
                str(lora_path),
                args.lora_scale,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        )

    print("HOT_BATCH_START", args.label, "lora_path", str(lora_path) if lora_path else "none", flush=True)
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
    prompt_count = min(args.prompt_count, len(PROMPTS))

    with torch.inference_mode():
        for i, prompt in enumerate(PROMPTS[:prompt_count], start=1):
            seed = 1000 + i - 1
            outfile = outdir / f"prompt{i}_{args.label}_{args.width}x{args.height}x{args.frames}.mp4"
            print("GENERATE_START", args.label, i, outfile, flush=True)
            gen_start = time.perf_counter()
            video, audio = pipe(
                prompt=prompt,
                seed=seed,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                frame_rate=args.fps,
                images=[],
                tiling_config=tiling,
                enhance_prompt=False,
            )
            infer_sec = time.perf_counter() - gen_start
            enc_start = time.perf_counter()
            encode_video(
                video=video,
                fps=args.fps,
                audio=audio,
                output_path=str(outfile),
                video_chunks_number=chunks,
            )
            encode_sec = time.perf_counter() - enc_start
            row = {
                "label": args.label,
                "prompt_index": i,
                "seed": seed,
                "width": args.width,
                "height": args.height,
                "frames": args.frames,
                "fps": args.fps,
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
        "lora_path": str(lora_path) if lora_path else None,
        "lora_scale": args.lora_scale,
        "rows": rows,
    }
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("HOT_BATCH_DONE", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
