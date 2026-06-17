# Training Strategy

## Objective

Train an 8-step LTX2.3 LoRA that remains compatible with the official LTX
DistilledPipeline while improving/maintaining quality under fast distilled
inference.

## Phase 1: Official 8-step Quality LoRA

Config: `configs/official8_phase1_1500.yaml`

- Base student: `ltx-2.3-22b-distilled-1.1.safetensors`
- Teacher: official 8-step distilled model
- DMD: disabled
- LoRA rank: 16
- Target modules: `to_q`, `to_k`, `to_v`, `to_out.0`
- Steps: 1500
- Learning rate: `5e-7`
- AV training: enabled
- Dataset: preprocessed full-modality video caption AV latents
- Resolution used by config: 1280x704, 121 frames

This phase was intended as a quality-preserving warmup before adversarial/DMD
updates.

## Phase 2: DMD From Phase1

Config: `configs/dmd_phase2_from1500_to10000.yaml`

- Load checkpoint: phase1 LoRA step 1500
- DMD: enabled
- Critic: enabled through the same official distilled checkpoint
- Max train steps in config: 10000
- DMD2000 tested checkpoint: global step 2000
- Learning rate: `2e-7`
- Critic learning rate: `1e-7`
- `dfake_gen_update_ratio`: 3
- Distill loss weight: 0.8
- DMD loss weight: 0.06
- Audio DMD loss weight: 0.01
- Critic loss weight: 1.0
- Audio critic loss weight: 0.05
- DMD gradient normalization: enabled
- Gradient clipping: generator 0.2, critic 0.2


## Why Use an 8-step Teacher?

The final DMD2000 setup uses the official 8-step distilled checkpoint as both
the teacher/reference model and the student initialization. This means the goal
is not direct 40-step-to-8-step compression. The goal is to improve or maintain
the official distilled model under the same inference pipeline through LoRA-based
domain adaptation and conservative DMD distribution matching.

This choice has several practical benefits:

- it avoids training/inference mismatch with the official DistilledPipeline;
- it preserves the official 8-step sigma schedule and two-stage inference path;
- it keeps DMD updates small enough to reduce saturation and structural damage;
- it allows the final artifact to remain a lightweight LoRA rather than a full
  checkpoint;
- it makes baseline comparison cleaner because teacher/reference and inference
  pipeline are fixed.

The limitation is also clear: because the teacher is not a stronger 40-step dev
model, this setup should be described as 8-step distilled-model adaptation with
DMD, not as a full stronger-teacher distillation experiment. A 40-step dev
teacher can be introduced in future work for higher quality ceiling, for example
through final-latent distillation, trajectory matching, and perceptual/temporal
losses while still preserving the 8-step inference target.

## DMD Lineage

This work is closer to Self-Forcing-style DMD than OmniForcing:

- fake-real distribution matching is added to the LTX trainer path
- a critic is used for DMD stabilization
- the official LTX DistilledPipeline remains the inference target
- OmniForcing was referenced conceptually, but not used as the training
  framework for DMD2000
