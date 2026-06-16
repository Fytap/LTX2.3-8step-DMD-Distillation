# Troubleshooting Notes

## Why not commit weights/videos?

GitHub is not suitable for large `.safetensors`, generated `.mp4`, datasets, or
VBench model caches. Use `.gitignore` and upload those artifacts to external
storage.

## Path portability

Most scripts were captured from the 98-server experiment and contain absolute
paths such as `/keyan/lsh/...` and `/keyan/LTX-2.3/...`. Update these before
running on another machine.

## GPU policy

Training experiments usually used GPUs 4-7. The one full DMD2000 VBench run used
all 8 H200 GPUs by user request, then future work should return to 4-GPU usage
unless explicitly changed.

## DMD instability

Earlier DMD runs showed saturation/quality degradation when the LoRA update was
too strong. The DMD2000 phase used conservative LR, rank 16, low DMD weights, and
gradient clipping to reduce damage to the official distilled prior.
