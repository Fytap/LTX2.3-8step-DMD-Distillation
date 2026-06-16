# Experiment Timeline

1. Started from Self-Forcing-Plus style 4-step DMD adaptation for LTX2.3.
2. Found early 4-step LoRA outputs were unstable/noisy under corrected inference.
3. Shifted to the official LTX DistilledPipeline so training and inference use the
   same 2-stage distilled logic.
4. Explored AV LoRA, AV-DMD, teacher8/student4, teacher8/student8, and
   teacher-dev40 variants.
5. Settled on an 8-step official-distill-compatible path:
   phase1 quality LoRA to 1500 steps, then phase2 DMD from that checkpoint.
6. Evaluated DMD2000 with full VBench-2.0 under `external/VBench-2.0`.

Key conclusion: DMD2000 improved the VBench leaderboard score versus the copied
baseline reference in this workspace, but action ordering and dynamic spatial
relationship remain weak points.
