# How We Arrived at DMD2000

This document records the practical path from the early LTX2.3 DMD attempts to
the final DMD2000 setup. The important point is that DMD2000 was not chosen as a
single-shot design. It is the result of repeatedly finding training/inference
mismatches, overly aggressive LoRA updates, unstable DMD objectives, and weak
quality preservation.

## 1. Starting Point: Direct 4-step DMD Adaptation

The first direction was to adapt Self-Forcing-Plus style DMD to LTX2.3 and train
a 4-step student. The goal was straightforward: make LTX2.3 much faster by using
DMD to compress the sampling process.

What went wrong:

- Early 4-step LoRA checkpoints often generated noisy or structureless videos.
- Some outputs looked like latent noise rather than decoded semantic video.
- Quality degraded quickly when DMD was applied too aggressively.
- The training code path and the inference code path were not fully aligned with
  the official LTX2.3 distilled inference behavior.

What we learned:

- For LTX2.3, simply making the step count smaller is not enough. The sigma
  schedule, latent scaling, LoRA injection position, VAE decode path, and
  distilled inference logic all need to match.
- A 4-step target is very sensitive. Small mismatches can completely destroy
  image structure.

## 2. Correcting Inference: Move Toward Official LTX DistilledPipeline

After the noisy 4-step results, we focused on whether inference itself was fair.
We compared ordinary LTX2.3 generation, official distilled generation, and LoRA
injection through the official LTX pipeline.

What went wrong before this correction:

- Some comparisons used a custom or partially adapted inference path.
- The DMD LoRA could be loaded, but the sampler behavior was not always the same
  as official LTX2.3 distilled inference.
- This made it hard to know whether bad results came from training failure or
  inference mismatch.

Correction:

- We moved evaluation to the official LTX DistilledPipeline.
- The LoRA is injected into the same official distilled checkpoint used by the
  baseline.
- VBench generation also follows this official-distilled inference target.

What we learned:

- Training and inference must share the same pipeline assumptions. Otherwise the
  checkpoint may look worse simply because it is being sampled incorrectly.

## 3. Trying AV LoRA and AV-DMD

Because LTX2.3 is an audio-video model, we then explored AV training: video and
audio latents together, not just video-only latent matching.

What went wrong:

- AV-DMD was much heavier and harder to stabilize.
- Keeping long text context, audio latent flow, video latent flow, teacher,
  student, and critic all active created high memory pressure.
- Even after memory optimizations, DMD could still damage visual quality if the
  objective was too strong.
- In several runs, saturation, blur, and shape deformation appeared after more
  training steps.

Corrections attempted:

- Put as much as possible on GPU to reduce CPU bottlenecks.
- Enable gradient checkpointing and staged release of intermediate activations.
- Reduce DMD/audio-DMD weights.
- Lower learning rates.
- Use smaller LoRA rank in later runs.
- Add loss logging and checkpoint-based comparisons.

What we learned:

- AV training is possible, but AV-DMD increases the number of failure modes.
- Audio-video synchronization is not guaranteed just because the model supports
  audio. The DMD objective must explicitly respect the AV latent path.
- For a public baseline experiment, video quality and pipeline alignment had to
  be stabilized first.

## 4. Teacher8/Student4 and 4-step Distillation Attempts

We repeatedly returned to the 4-step goal because it was the original speed
objective. We tried teacher-8/student-4 style settings and official-scheduler
DMD variants.

What went wrong:

- 4-step outputs were often blurrier than the official baseline.
- Structure and motion could become unstable even when loss values looked
  reasonable.
- Longer training did not consistently improve quality; sometimes it made
  saturation and deformation worse.
- Lowering or increasing the LoRA scale at inference helped slightly, but did
  not fully solve the quality gap.

What we learned:

- The official 8-step distilled model is already a strong prior.
- Forcing another aggressive reduction to 4 steps with limited data and LoRA-only
  capacity can easily damage the prior.
- Better losses and better teacher trajectories are needed before claiming a
  reliable 4-step model.

## 5. Testing 8-step Distillation Instead of 4-step

Because 4-step quality was unstable, we changed the question. Instead of asking
`can we immediately produce a good 4-step model?`, we asked:

> Can we safely adapt the official 8-step distilled model with LoRA and DMD while
> keeping the official DistilledPipeline unchanged?

This led to the 8-step-to-8-step setup.

Why this is still meaningful:

- The goal becomes controlled adaptation, not classic strong-teacher compression.
- The official 8-step model is used as a stable reference prior.
- DMD becomes a distribution correction method rather than a claim that the
  teacher is stronger.
- Any improvement or degradation can be measured against the official 8-step
  baseline under the same pipeline.

## 6. Phase1: Quality-Preserving LoRA Warmup

Before enabling DMD, we added a phase1 LoRA warmup.

Purpose:

- Keep the student close to the official 8-step distilled prior.
- Adapt to the target AV latent dataset without immediately applying adversarial
  or distribution-matching pressure.
- Produce a stable checkpoint that phase2 can resume from.

Why this was needed:

- Earlier DMD-only or high-DMD-weight runs damaged quality quickly.
- A warmup stage helps the LoRA learn a conservative update first.

Resulting phase1 choice:

- Base student: official `ltx-2.3-22b-distilled-1.1`.
- Teacher/reference: official `ltx-2.3-22b-distilled-1.1`.
- LoRA rank: 16.
- Low learning rate.
- Audio/video latent dataset.
- DMD disabled.
- Phase1 checkpoint: step 1500.

## 7. Phase2: Conservative DMD From Phase1

After phase1, DMD was reintroduced conservatively.

Key choices:

- Resume from phase1 step 1500.
- Keep the official 8-step distilled model as teacher/reference.
- Use critic/fake-real distribution matching.
- Use low DMD loss weight.
- Use low critic/audio critic weights.
- Normalize and clip DMD gradients.
- Test early at global step 2000 instead of blindly training to 10000.

Why step 2000:

- Previous runs showed that more steps can reduce saturation or improve loss, but
  can also over-adapt and damage visual fidelity.
- DMD2000 is an early, conservative checkpoint chosen to avoid the over-training
  behavior seen in longer runs.


## 8. Why the 40-step Teacher to 8-step Student Route Was Not Used as the Current Main Result

From a pure distillation perspective, using the official 40-step dev model as a
teacher and training an 8-step student is the more theoretically attractive
setup. The 40-step teacher has a higher quality ceiling, richer denoising
trajectory, and should in principle provide stronger supervision than the
official 8-step distilled checkpoint. We did explore this direction conceptually
and in trial runs, but it did not become the current public DMD2000 result.

The main reason is that the early 40-step-teacher experiments did not produce a
clear, stable improvement under our target inference path. Several problems
appeared repeatedly.

First, the teacher-student gap was larger than our LoRA-only student could
comfortably absorb. The 40-step dev model follows a much denser denoising
trajectory, while the student was expected to land on an 8-step distilled
trajectory. Matching only final latents or a small number of sampled trajectory
points was not enough to transfer the teacher's quality reliably. When the loss
was made stronger, the LoRA update became too aggressive and visual quality
degraded. When the loss was made weaker, the student stayed close to the
official 8-step baseline and the benefit of the 40-step teacher became unclear.

Second, the 40-step teacher introduced a stronger training/inference mismatch.
Our deployment target was still the official DistilledPipeline, whose behavior is
built around the official distilled schedule and two-stage inference logic. A
40-step dev teacher can supervise high-quality final samples, but the student is
not actually sampled with the 40-step dev pipeline at inference time. Without a
carefully designed bridge between the 40-step teacher trajectory and the
8-step distilled schedule, the student can learn targets that are difficult to
realize during official distilled inference. In practice this showed up as blur,
washed-out color, unstable structure, or weaker motion consistency.

Third, the cost and instability were much higher. Running the 40-step teacher
during training or caching enough teacher trajectories is expensive in both time
and storage, especially at higher resolution and with audio-video latents. The
extra supervision only makes sense if it consistently improves the student, but
our early results did not justify making that more complex route the main public
checkpoint.

Fourth, DMD made the mismatch even more sensitive. DMD is not just a simple
regression loss; it relies on fake-real distribution matching and critic
feedback. If teacher latents, student latents, critic inputs, sigma sampling, and
inference schedule are not strictly aligned, DMD can amplify the mismatch rather
than fix it. Some of the earlier degraded outputs were consistent with this:
loss values could move in the expected direction, while perceptual quality became
worse.

Because of these issues, we chose not to present the 40-step-teacher route as the
current main result. Instead, DMD2000 uses the official 8-step distilled model as
both student initialization and teacher/reference. This is a more conservative
choice, but it gives a cleaner experiment: the teacher/reference, sigma schedule,
LoRA injection, and final inference pipeline are all aligned. The result is not
claimed as a full 40-step-to-8-step distillation success; it is presented as a
pipeline-compatible 8-step distilled-model adaptation with conservative DMD.

The 40-step teacher route is still the most important future direction, but it
needs a better design than the early trials. A stronger future version should
probably combine:

- 40-step dev teacher final-latent supervision for quality ceiling;
- selected intermediate trajectory matching from the 40-step teacher;
- official 8-step distilled teacher regularization for path stability;
- strict sigma mapping between 40-step teacher states and 8-step student states;
- conservative LoRA rank and learning rate;
- delayed or staged DMD, enabled only after the student has learned the teacher
  trajectory reasonably well;
- VBench and visual inspection at early checkpoints to avoid over-training.

In short, the 40-step teacher is theoretically better, but our previous attempts
showed that simply adding a stronger teacher is not enough. Without careful
trajectory alignment and stability regularization, it can make the 8-step student
worse rather than better.

## 9. Why We Did Not Present This as 40-step Teacher Distillation

We discussed and tested directions involving the 40-step dev model as a stronger
teacher. That direction is theoretically more suitable if the goal is to exceed
an 8-step teacher's quality ceiling.

However, the public DMD2000 result is not that experiment.

DMD2000 uses the official 8-step distilled checkpoint as both:

- the base student initialization;
- the teacher/reference model for phase1 and phase2.

This was chosen because the immediate priority became reliable pipeline-aligned
adaptation. The limitation is clear: it should not be described as direct
40-step-to-8-step knowledge distillation. It is better described as:

> Official 8-step LTX2.3 distilled-model adaptation with LoRA + conservative DMD
> distribution matching.

## 10. Final Positioning of DMD2000

DMD2000 is best understood as a controlled research checkpoint:

- It keeps the official LTX DistilledPipeline as the inference target.
- It uses LoRA, not full-model fine-tuning.
- It uses phase1 quality warmup before phase2 DMD.
- It keeps DMD conservative to avoid the failures seen in earlier runs.
- It reports full VBench-2.0 scores instead of only selected visual examples.

The main advantage is not that 8-step student must be better than 8-step teacher.
The advantage is that the experiment is fair, reproducible, lightweight, and
pipeline-compatible. It gives a safer base for future work, especially a future
variant that introduces the 40-step dev model as an additional stronger teacher.

## 11. Future Direction After These Lessons

The next stronger experiment should keep the pipeline alignment lessons but use a
stronger teacher signal:

- student initialized from official 8-step distilled checkpoint;
- teacher A: official 40-step dev model for final quality and trajectory targets;
- teacher B: official 8-step distilled model for path stability;
- phase1: quality/domain adaptation;
- phase2: DMD with conservative critic and fake-real matching;
- strict evaluation with official DistilledPipeline, LoRA scales, VBench, and
  human visual inspection.

This would preserve the lessons from DMD2000 while giving the student a real
chance to learn from a stronger teacher.
