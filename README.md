# LTX2.3 8-step DMD Distillation

This repository documents an experimental 8-step DMD distillation workflow for
LTX2.3 audio-video generation. The project keeps the official LTX2.3
DistilledPipeline as the inference target, adds LoRA-based post-training, and
then introduces a Self-Forcing-style DMD phase with critic / fake-real
distribution matching.

The goal is not to replace the official LTX2.3 pipeline. The goal is to study
whether a lightweight LoRA trained under the same distilled inference regime can
improve selected video-generation metrics while remaining compatible with the
official DistilledPipeline.

## Highlights

- Official LTX2.3 DistilledPipeline is used for inference and VBench generation.
- Student model starts from `ltx-2.3-22b-distilled-1.1`.
- Training is split into a quality-preserving phase and a DMD phase.
- DMD2000 is a LoRA checkpoint, not a full model checkpoint.
- DMD logic is closer to Self-Forcing-style DMD / distribution matching, not
  OmniForcing.
- Full VBench-2.0 evaluation is included for official 8-step baseline vs
  DMD2000.
- Large artifacts are excluded from Git: checkpoints, datasets, generated
  videos, and VBench model caches.

## Method Summary

The final tested artifact is called **DMD2000**.

```text
official LTX2.3 8-step distilled base
+ phase1 official-distill quality LoRA to step 1500
+ phase2 DMD LoRA training to global step 2000
```

Expected local checkpoint layout:

```text
checkpoints/
├── ltx-2.3-22b-distilled-1.1.safetensors
├── ltx-2.3-spatial-upscaler-x2-1.1.safetensors
├── gemma/
└── lora_weights_step_02000.safetensors   # DMD2000 LoRA, not committed
```

During inference and VBench generation, this LoRA is loaded on top of the
official distilled checkpoint:

```text
ltx-2.3-22b-distilled-1.1.safetensors + DMD2000 LoRA
```

## Framework Choice

This experiment is **not OmniForcing training**.

The project started from earlier Self-Forcing-Plus / DMD adaptation attempts,
but DMD2000 was trained in a modified LTX trainer path and evaluated through the
official LTX DistilledPipeline. In practical terms:

```text
LTX official DistilledPipeline
+ LoRA training
+ trajectory distillation
+ quality protection losses
+ DMD critic / fake-real distribution matching
```

OmniForcing and JoyAI-Echo were useful references for how open-source
audio-video distillation projects organize pipeline, evaluation, and result
reporting. However, this repository's DMD2000 checkpoint was not produced by
OmniForcing.

## Training Pipeline

### Phase 1: official 8-step quality LoRA

Config:

```text
configs/official8_phase1_1500.yaml
```

Purpose:

The first phase warms up a LoRA under the official 8-step distilled model before
introducing DMD. This reduces the risk that the DMD objective immediately
damages the already useful official distilled prior.

Main settings:

| Setting | Value |
|---|---|
| Base student | `ltx-2.3-22b-distilled-1.1` |
| Teacher | `ltx-2.3-22b-distilled-1.1` |
| Training mode | LoRA |
| LoRA rank / alpha / dropout | `16 / 16 / 0.05` |
| LoRA target modules | `to_q`, `to_k`, `to_v`, `to_out.0` |
| DMD | disabled |
| Steps | `1500` |
| Learning rate | `5.0e-7` |
| Video loss weight | `1.0` |
| Audio loss weight | `0.2` |
| Clean x0 loss weight | `0.01` |
| Quality moment / delta loss | `0.01 / 0.01` |
| Gradient checkpointing | enabled |
| Precision | bf16 |
| Dataset format | preprocessed AV latents |
| Resolution in config | `1280 x 704 x 121 frames` |

The phase1 checkpoint used by DMD phase2 was:

```text
checkpoints/phase1/lora_weights_step_01500.safetensors
```

### Phase 2: DMD from phase1

Config:

```text
configs/dmd_phase2_from1500_to10000.yaml
```

Purpose:

The second phase resumes from the phase1 LoRA and adds DMD. The DMD component is
kept conservative: low learning rate, low DMD loss weight, critic stabilization,
gradient normalization, and clipping.

Main settings:

| Setting | Value |
|---|---|
| Base student | `ltx-2.3-22b-distilled-1.1` |
| Load checkpoint | phase1 LoRA step 1500 |
| Teacher model | `ltx-2.3-22b-distilled-1.1` |
| Critic model | `ltx-2.3-22b-distilled-1.1` |
| DMD | enabled |
| Max train steps in config | `10000` |
| Tested checkpoint | global step `2000` |
| Generator LR | `2.0e-7` |
| Critic LR | `1.0e-7` |
| `dfake_gen_update_ratio` | `3` |
| Distill loss weight | `0.8` |
| DMD loss weight | `0.06` |
| Audio DMD loss weight | `0.01` |
| Critic loss weight | `1.0` |
| Audio critic loss weight | `0.05` |
| Critic loss type | Huber |
| Critic Huber delta | `0.1` |
| Normalize DMD gradient | enabled |
| DMD gradient clip | `3.0` |
| Generator grad norm | `0.2` |
| Critic grad norm | `0.2` |
| Checkpoint interval | `500` |
| Precision | bf16 |

The tested DMD2000 checkpoint is a LoRA file:

```text
checkpoints/dmd_phase2/lora_weights_step_02000.safetensors
```

## Sigma Schedule

Both phase1 and phase2 use the official 8-step stage1 sigma list:

```yaml
stage1_sigmas:
  - 1.0
  - 0.99375
  - 0.9875
  - 0.98125
  - 0.975
  - 0.909375
  - 0.725
  - 0.421875
  - 0.0
sigma_sampling_strategy: interval_weighted
trajectory_distillation: true
trajectory_step_sampling: cycle
```

The intent is to stay aligned with the official distilled schedule rather than
training a separate sampler that would be hard to compare fairly.

## Dataset

The training configs expect a preprocessed audio-video latent dataset. A portable
layout would look like:

```text
data/
└── full-modality-video-caption/
    └── preprocessed_1280x704x121_av/
```

The original raw dataset source used in the experiment was:

```text
ngqtrung/full-modality-video-caption
```

The repository does not include this dataset. Only the training configs and
loss histories are included. If you reproduce the experiment, update the
absolute paths in the YAML files to your local dataset location.

## Inference

DMD2000 is evaluated by injecting the LoRA into the official LTX DistilledPipeline:

```text
official distilled checkpoint + DMD2000 LoRA
```

The VBench generation config is:

```text
configs/vbench_dmd2000_480p.yaml
```

The original DMD2000 VBench generation used:

| Setting | Value |
|---|---|
| Pipeline | LTX DistilledPipeline |
| Base checkpoint | `ltx-2.3-22b-distilled-1.1` |
| LoRA scale | `1.0` |
| Output resolution | `832 x 512` |
| Frames | `121` |
| FPS | `24` |
| Prompt set | VBench-2.0 prompts |
| GPU mode for full VBench run | 8 x H200 |

Example:

```bash
python scripts/run_vbench2_distill.py \
  --config configs/vbench_dmd2000_480p.yaml \
  --gpus 0,1,2,3,4,5,6,7 \
  --per-gpu-workers 3
```

## VBench-2.0 Evaluation

The full VBench-2.0 comparison below uses:

- official distilled 8-step baseline: `results/official8_vbench2_baseline/`
- DMD2000: `results/dmd2000_vbench2_480p_cal_local_scores.log`

The baseline aggregate and sub-dimension scores are copied from the original
official 8-step VBench run. The DMD2000 full score log is included in this
repository under `results/`.

### Aggregate Metrics

| Metric | Official 8-step | DMD2000 | Delta |
|---|---:|---:|---:|
| Leaderboard Score | 0.555218 | 0.572016 | +0.016798 |
| Original VBench Overall | 0.615066 | 0.629842 | +0.014776 |
| Action Consistency | 0.430087 | 0.415284 | -0.014803 |
| Physical Laws | 0.735490 | 0.724794 | -0.010696 |
| Human Consistency | 0.793271 | 0.798936 | +0.005665 |
| Camera Consistency | 0.429382 | 0.517902 | +0.088520 |
| Contextual Consistency | 0.409747 | 0.415221 | +0.005474 |
| Diversity & Composition | 0.523568 | 0.559085 | +0.035517 |

### Full Sub-dimension Comparison

| VBench dimension | Official 8-step | DMD2000 | Delta |
|---|---:|---:|---:|
| Dynamic Attribute | 0.454212 | 0.443223 | -0.010989 |
| Motion Rationality | 0.488506 | 0.545977 | +0.057471 |
| Dynamic Spatial Relationship | 0.410628 | 0.328502 | -0.082126 |
| Motion Order Understanding | 0.367003 | 0.343434 | -0.023569 |
| Mechanics | 0.733813 | 0.708955 | -0.024858 |
| Thermotics | 0.653333 | 0.675676 | +0.022343 |
| Human Interaction | 0.740000 | 0.760000 | +0.020000 |
| Material | 0.814815 | 0.754545 | -0.060270 |
| Human Identity | 0.622477 | 0.656827 | +0.034350 |
| Human Anatomy | 0.899147 | 0.878915 | -0.020232 |
| Human Clothes | 0.911458 | 0.900000 | -0.011458 |
| Camera Motion | 0.503086 | 0.503086 | +0.000000 |
| Multi-View Consistency | 0.355678 | 0.532717 | +0.177039 |
| Complex Plot | 0.133333 | 0.139815 | +0.006482 |
| Complex Landscape | 0.177778 | 0.211111 | +0.033333 |
| Instance Preservation | 0.918129 | 0.894737 | -0.023392 |
| Diversity | 0.513804 | 0.558211 | +0.044407 |
| Composition | 0.533333 | 0.559958 | +0.026625 |

### Result Interpretation

DMD2000 improves the two headline scores in this VBench run:

- Leaderboard Score: `+0.016798`
- Original VBench Overall: `+0.014776`

The strongest gains are:

- Multi-View Consistency: `+0.177039`
- Motion Rationality: `+0.057471`
- Diversity: `+0.044407`
- Human Identity: `+0.034350`
- Complex Landscape: `+0.033333`
- Composition: `+0.026625`

The main regressions are:

- Dynamic Spatial Relationship: `-0.082126`
- Material: `-0.060270`
- Mechanics: `-0.024858`
- Motion Order Understanding: `-0.023569`
- Instance Preservation: `-0.023392`

This suggests that the conservative DMD phase helped some global consistency and
camera/diversity metrics, but it did not solve fine-grained action ordering or
dynamic spatial reasoning. Those remain the next targets for training changes.

## Repository Layout

```text
.
├── configs/
│   ├── official8_phase1_1500.yaml
│   ├── dmd_phase2_from1500_to10000.yaml
│   └── vbench_dmd2000_480p.yaml
├── scripts/
│   ├── train_phase1_then_dmd_phase2.sh
│   ├── infer_distilled_pipeline_lora.py
│   ├── infer_distilled_pipeline_prompt_file.py
│   ├── run_vbench2_distill.py
│   ├── evaluate_vbench2.sh
│   └── eval_dmd2000_javisbench50.sh
├── src/
│   └── ltx_trainer_snapshot/
│       └── ltx_trainer/
│           ├── trainer.py
│           └── training_strategies/
├── docs/
│   ├── experiment_timeline.md
│   ├── training_strategy.md
│   ├── troubleshooting.md
│   └── vbench_results.md
└── results/
    ├── dmd2000_summary.md
    ├── dmd2000_vbench2_480p_cal_local_scores.log
    ├── phase1_1500_loss_history.csv
    └── dmd_phase2_loss_history.csv
```

## Reproduction Notes

This repository is a clean summary of the experiment, not a fully self-contained
model release. To reproduce the run, prepare the following local paths first:

```text
checkpoints/
├── ltx-2.3-22b-distilled-1.1.safetensors
├── ltx-2.3-spatial-upscaler-x2-1.1.safetensors
├── gemma/
├── phase1/lora_weights_step_01500.safetensors
└── dmd_phase2/lora_weights_step_02000.safetensors

data/
└── full-modality-video-caption/preprocessed_1280x704x121_av/

external/
└── VBench-2.0/
```

Then update these fields in the YAML files before running:

```text
model.model_path
model.text_encoder_path
model.load_checkpoint
training_strategy.teacher_model_path
training_strategy.critic_model_path
data.preprocessed_data_root
output_dir
vbench2.prompts_dir
vbench2.prompts_aug_dir
vbench2.output_dir
```

The committed configs preserve the original experiment values for auditability,
so they are not plug-and-play until those paths are changed.

## Commands

### Train phase1 and phase2

```bash
bash scripts/train_phase1_then_dmd_phase2.sh
```

### Generate with official DistilledPipeline + LoRA

```bash
python scripts/infer_distilled_pipeline_prompt_file.py \
  --label dmd2000 \
  --outdir outputs/example \
  --prompt-file prompts/example.jsonl \
  --lora-path checkpoints/lora_weights_step_02000.safetensors \
  --lora-scale 1.0 \
  --width 1280 \
  --height 704 \
  --frames 121 \
  --fps 24
```

### Run VBench generation

```bash
python scripts/run_vbench2_distill.py \
  --config configs/vbench_dmd2000_480p.yaml \
  --gpus 0,1,2,3,4,5,6,7 \
  --per-gpu-workers 3
```

### Run VBench scoring

```bash
VIDEOS_ROOT=/path/to/generated/videos \
OUTPUT_ROOT=/path/to/evaluation/output \
bash scripts/evaluate_vbench2.sh
```

## What Is Not Included

The following are intentionally excluded:

- `.safetensors` model weights and LoRA checkpoints
- generated `.mp4` files
- raw datasets
- preprocessed latent datasets
- VBench model caches
- full Python/conda environments
- large logs and temporary outputs

## Limitations

- DMD2000 is a research checkpoint, not a production release.
- Improvements are not uniform across all VBench dimensions.
- Dynamic spatial relationship and motion order remain weak.
- The training configs are path-specific and require cleanup for public
  reproduction.
- VBench scores are useful for regression tracking, but they do not replace
  human inspection of generated videos.

## References

- [LTX-Video official repository](https://github.com/Lightricks/LTX-Video)
- [JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo)
- [OmniForcing](https://github.com/OmniForcing/OmniForcing)
- [Distribution Matching Distillation](https://arxiv.org/abs/2311.18828)

## License and Usage

This repository contains experiment notes, configuration files, scripts, and
modified-code snapshots. The underlying LTX2.3 model, weights, and upstream code
remain governed by their original licenses. Check the LTX2.3 / LTX-Video license
before using any derivative model commercially.
