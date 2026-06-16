# LTX2.3 8-step DMD Distillation Experiments

This repository summarizes a set of LTX2.3 distillation experiments focused on
an 8-step audio-video student, official LTX DistilledPipeline inference, and a
Self-Forcing-style DMD phase with critic / fake-real distribution matching.

## What This Is

The main artifact documented here is **DMD2000**:

```text
official LTX2.3 8-step distilled base
+ phase1 official-distill quality LoRA to step 1500
+ phase2 DMD LoRA training to global step 2000
```

The tested LoRA checkpoint in the original workspace was:

```text
/keyan/lsh/lsh_teacher_cache_8step_distill/runs/official8_student8_av_dmd_rank16_phase2_from1500_10000/checkpoints/lora_weights_step_02000.safetensors
```

Large artifacts are intentionally not tracked in Git. Put model weights,
datasets, generated videos, VBench caches, and checkpoints in external storage.

## Framework Choice

The DMD2000 experiment was **not trained with OmniForcing**. It uses LTX2.3's
official DistilledPipeline ecosystem with a DMD/critic/fake-real matching
extension inspired by Self-Forcing-style DMD.

In short:

```text
LTX official DistilledPipeline + LoRA + DMD critic/fake-real matching
```

## Main Results

Full VBench-2.0 evaluation of DMD2000 at 832x512, 121 frames:

| Metric | Score |
|---|---:|
| Leaderboard Score | 0.572016 |
| Original VBench Overall | 0.629842 |
| Action Consistency | 0.415284 |
| Physical Laws | 0.724794 |
| Human Consistency | 0.798936 |
| Camera Consistency | 0.517902 |
| Contextual Consistency | 0.415221 |
| Diversity & Composition | 0.559085 |

See `results/dmd2000_vbench2_480p_cal_local_scores.log` and
`docs/vbench_results.md` for the full table.

## Repository Layout

```text
configs/   Training and VBench configs
scripts/   Training, inference, and VBench runner scripts
src/       Snapshot of the modified ltx-trainer files
docs/      Experiment timeline and method notes
results/   Loss CSVs and VBench result summaries
```

## Typical Workflow

1. Prepare LTX2.3 official weights and Gemma text encoder.
2. Prepare preprocessed AV latent dataset.
3. Run phase1 official 8-step quality LoRA training.
4. Resume phase2 DMD training from phase1 checkpoint.
5. Inject LoRA into LTX DistilledPipeline for inference.
6. Run VBench-2.0 with generated videos.

Example entry scripts:

```bash
bash scripts/train_phase1_then_dmd_phase2.sh
python scripts/run_vbench2_distill.py --config configs/vbench_dmd2000_480p.yaml
```

The scripts contain absolute paths from the original 98-server workspace. Before
running elsewhere, update `/keyan/...` paths to your machine.
