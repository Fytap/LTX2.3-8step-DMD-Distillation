"""Official-schedule text-to-video distillation strategy.

This strategy keeps the LTX trainer data path intact, but replaces the
flow-matching target with teacher/student x0 matching on the same noisy
audio-video latents. The sampled sigmas are the public distilled pipeline
sigmas, so validation with ltx_pipelines.distilled uses the same denoising
parameterization as training.

When enable_dmd is true, the same official audio/video latent path also trains
an explicit fake-score critic. The generator loss then uses the DMD fake-real
score difference (critic x0 - teacher x0) as the distribution-matching
gradient, while a small teacher/student x0 anchor remains available for
stability.
"""

from dataclasses import replace
from typing import Any, Literal

import torch
from pydantic import Field, field_validator
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_core.utils import to_denoised
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)

DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
STAGE_2_DISTILLED_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)


class DistillTextToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for teacher/student AV distillation on official distilled sigmas."""

    name: Literal["distill_text_to_video"] = "distill_text_to_video"

    teacher_model_path: str = Field(
        description="Frozen teacher checkpoint path, usually the official 8-step distilled checkpoint."
    )

    with_audio: bool = Field(
        default=True,
        description="Whether to distill the audio branch together with video.",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents when with_audio is True.",
    )

    distill_stage: Literal["stage1", "stage2", "both"] = Field(
        default="stage2",
        description="Official distilled sigma subset to sample. stage2 matches the 4-step refinement schedule.",
    )
    stage1_sigmas: tuple[float, ...] | None = Field(
        default=None,
        description="Optional custom Stage-1 sigma points, including terminal 0.0.",
    )
    stage2_sigmas: tuple[float, ...] | None = Field(
        default=None,
        description="Optional custom Stage-2 sigma points, including terminal 0.0.",
    )

    video_loss_weight: float = Field(default=1.0, ge=0.0)
    audio_loss_weight: float = Field(default=0.2, ge=0.0)
    clean_x0_loss_weight: float = Field(
        default=0.05,
        ge=0.0,
        description="Small clean latent anchor that discourages teacher/student collapse into texture noise.",
    )
    enable_dmd: bool = Field(
        default=False,
        description="Train a fake-score critic and add DMD fake-real distribution matching to the generator loss.",
    )
    critic_model_path: str | None = Field(
        default=None,
        description="Checkpoint used to initialize the fake-score critic. Defaults to teacher_model_path.",
    )
    critic_learning_rate: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional critic LR. Defaults to optimization.learning_rate.",
    )
    dfake_gen_update_ratio: int = Field(
        default=3,
        ge=1,
        description="Train generator every N global steps; train critic every step.",
    )
    distill_loss_weight: float = Field(
        default=0.1,
        ge=0.0,
        description="Teacher/student x0 anchor weight when DMD is enabled.",
    )
    dmd_loss_weight: float = Field(default=1.0, ge=0.0)
    audio_dmd_loss_weight: float = Field(default=0.2, ge=0.0)
    critic_loss_weight: float = Field(default=1.0, ge=0.0)
    audio_critic_loss_weight: float = Field(default=0.2, ge=0.0)
    normalize_dmd_gradient: bool = Field(default=True)
    dmd_gradient_clip: float | None = Field(
        default=10.0,
        ge=0.0,
        description="Clamp normalized DMD gradient for stability. Set null to disable.",
    )
    max_grad_norm_generator: float = Field(default=0.3, ge=0.0)
    max_grad_norm_critic: float = Field(default=0.5, ge=0.0)
    sigma_sampling_strategy: Literal["uniform", "interval_weighted", "late_biased"] = Field(
        default="interval_weighted",
        description="How to sample official sigma intervals for single-step distillation.",
    )
    trajectory_distillation: bool = Field(
        default=False,
        description=(
            "Roll the student along the official distilled sigma path and train one selected "
            "transition per batch. This covers the full 8-step trajectory over time without "
            "retaining the whole rollout graph."
        ),
    )
    trajectory_step_sampling: Literal["cycle", "uniform", "interval_weighted", "late_biased"] = Field(
        default="cycle",
        description="How to choose the trainable transition when trajectory_distillation is enabled.",
    )
    trajectory_transition_loss_weight: float = Field(
        default=0.5,
        ge=0.0,
        description="Extra teacher/student next-latent loss for the selected rollout transition.",
    )
    trajectory_endpoint_loss_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Teacher/student x0 loss weight inside the trajectory objective.",
    )
    quality_moment_loss_weight: float = Field(
        default=0.02,
        ge=0.0,
        description="Latent mean/std preservation against data latents to reduce washed-out outputs.",
    )
    quality_delta_loss_weight: float = Field(
        default=0.02,
        ge=0.0,
        description="Neighbor-token latent delta preservation against data latents to protect local detail.",
    )
    dmd_start_step: int = Field(
        default=0,
        ge=0,
        description="Global step at which generator DMD gradients start. Before this, use distillation anchors only.",
    )
    dmd_ramp_steps: int = Field(
        default=0,
        ge=0,
        description="Linearly ramp DMD weight over this many steps after dmd_start_step.",
    )
    critic_warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Start training the critic this many steps before DMD generator updates begin.",
    )
    critic_loss_type: Literal["mse", "huber"] = Field(
        default="huber",
        description="Loss used for fake-score critic regression.",
    )
    critic_huber_delta: float = Field(default=0.1, gt=0.0)

    @field_validator("teacher_model_path")
    @classmethod
    def validate_teacher_model_path(cls, v: str) -> str:
        if not v:
            raise ValueError("teacher_model_path must be provided")
        return v


class DistillTextToVideoStrategy(TrainingStrategy):
    """Distill a student LoRA against a frozen teacher on official distilled sigmas."""

    config: DistillTextToVideoConfig

    def __init__(self, config: DistillTextToVideoConfig):
        super().__init__(config)
        self._current_step = 0
        logger.debug(
            "Using official distilled sigma schedule %s for teacher/student x0 matching",
            config.distill_stage,
        )

    @property
    def requires_audio(self) -> bool:
        return self.config.with_audio

    @property
    def requires_teacher(self) -> bool:
        return True

    @property
    def teacher_model_path(self) -> str:
        return self.config.teacher_model_path

    @property
    def requires_critic(self) -> bool:
        return self.config.enable_dmd

    @property
    def critic_model_path(self) -> str:
        return self.config.critic_model_path or self.config.teacher_model_path

    @property
    def uses_trajectory_distillation(self) -> bool:
        return self.config.trajectory_distillation

    def set_current_step(self, step: int) -> None:
        self._current_step = max(0, int(step))

    def should_apply_dmd(self, step: int | None = None) -> bool:
        step = self._current_step if step is None else step
        return self.config.enable_dmd and step >= self.config.dmd_start_step

    def should_train_critic(self, step: int) -> bool:
        if not self.config.enable_dmd:
            return False
        critic_start = max(0, self.config.dmd_start_step - self.config.critic_warmup_steps)
        return step >= critic_start

    def should_train_generator(self, step: int) -> bool:
        if not self.should_apply_dmd(step):
            return True
        return (step - 1) % self.config.dfake_gen_update_ratio == 0

    def dmd_weight_scale(self, step: int | None = None) -> float:
        if not self.should_apply_dmd(step):
            return 0.0
        step = self._current_step if step is None else step
        if self.config.dmd_ramp_steps <= 0:
            return 1.0
        progress = (step - self.config.dmd_start_step + 1) / self.config.dmd_ramp_steps
        return max(0.0, min(1.0, progress))

    def get_data_sources(self) -> list[str] | dict[str, str]:
        sources = {
            "latents": "latents",
            "conditions": "conditions",
        }
        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"
        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,  # noqa: ARG002
    ) -> ModelInputs:
        latents = batch["latents"]
        video_latents = self._video_patchifier.patchify(latents["latents"])

        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                "Different FPS values found in the batch. Found: %s, using the first one: %s",
                fps.tolist(),
                fps[0].item(),
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size, video_seq_len, _channels = video_latents.shape
        device = video_latents.device
        dtype = video_latents.dtype

        if self.config.trajectory_distillation:
            sigmas = self._trajectory_start_sigmas(batch_size=batch_size, device=device)
            trajectory_step_index = self._sample_trajectory_step_index(device=device)
        else:
            sigmas = self._sample_official_sigmas(batch_size=batch_size, device=device)
            trajectory_step_index = None
        video_noise = torch.randn_like(video_latents)
        noisy_video = (1 - sigmas.view(-1, 1, 1)) * video_latents + sigmas.view(-1, 1, 1) * video_noise
        video_timesteps = sigmas.view(-1, 1).expand(-1, video_seq_len)
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )
        video_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )
        video_loss_mask = torch.ones(batch_size, video_seq_len, dtype=torch.bool, device=device)

        audio_modality = None
        audio_targets = None
        audio_loss_mask = None
        if self.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        extra: dict[str, Any] = {}
        if trajectory_step_index is not None:
            schedule = self._selected_sigmas()
            extra.update(
                {
                    "trajectory_enabled": True,
                    "trajectory_step_index": trajectory_step_index,
                    "trajectory_sigmas": schedule,
                    "trajectory_next_sigma": float(schedule[trajectory_step_index + 1]),
                }
            )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_latents,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            teacher_video=video_modality,
            teacher_audio=audio_modality,
            extra=extra,
        )

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor]:
        audio_latents = self._audio_patchifier.patchify(batch["audio_latents"]["latents"])
        audio_seq_len = audio_latents.shape[1]
        audio_noise = torch.randn_like(audio_latents)
        noisy_audio = (1 - sigmas.view(-1, 1, 1)) * audio_latents + sigmas.view(-1, 1, 1) * audio_noise
        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len)
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        audio_modality = Modality(
            enabled=True,
            latent=noisy_audio,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )
        audio_loss_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.bool, device=device)
        return audio_modality, audio_latents, audio_loss_mask

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        loss, _metrics = self.compute_generator_loss(video_pred, audio_pred, inputs)
        return loss

    def compute_generator_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if inputs.teacher_video_pred is None or inputs.teacher_video is None:
            raise ValueError("DistillTextToVideoStrategy requires teacher video predictions")

        student_video_x0 = self._to_x0(inputs.video, video_pred)
        teacher_video_x0 = self._to_x0(inputs.teacher_video, inputs.teacher_video_pred).detach()
        video_distill_loss = self._masked_mse(student_video_x0, teacher_video_x0, inputs.video_loss_mask)
        video_loss = self.config.trajectory_endpoint_loss_weight * video_distill_loss

        if inputs.extra.get("trajectory_enabled", False):
            next_sigma = float(inputs.extra["trajectory_next_sigma"])
            student_video_next = self.euler_step(inputs.video, video_pred, next_sigma)
            teacher_video_next = self.euler_step(inputs.teacher_video, inputs.teacher_video_pred, next_sigma).detach()
            video_transition_loss = self._masked_mse(student_video_next, teacher_video_next, inputs.video_loss_mask)
            video_loss = video_loss + self.config.trajectory_transition_loss_weight * video_transition_loss
        else:
            video_transition_loss = None

        if self.config.clean_x0_loss_weight > 0:
            clean_video_loss = self._masked_mse(student_video_x0, inputs.video_targets, inputs.video_loss_mask)
            video_loss = video_loss + self.config.clean_x0_loss_weight * clean_video_loss
        else:
            clean_video_loss = None

        video_quality_loss = self._quality_loss(
            pred=student_video_x0,
            target=inputs.video_targets,
            mask=inputs.video_loss_mask,
        )
        video_loss = video_loss + video_quality_loss

        audio_distill_loss = None
        audio_transition_loss = None
        clean_audio_loss = None
        audio_quality_loss = None
        audio_loss = None
        student_audio_x0 = None
        teacher_audio_x0 = None

        if (
            self.config.with_audio
            and audio_pred is not None
            and inputs.audio is not None
            and inputs.audio_targets is not None
            and inputs.audio_loss_mask is not None
            and inputs.teacher_audio is not None
            and inputs.teacher_audio_pred is not None
        ):
            student_audio_x0 = self._to_x0(inputs.audio, audio_pred)
            teacher_audio_x0 = self._to_x0(inputs.teacher_audio, inputs.teacher_audio_pred).detach()
            audio_distill_loss = self._masked_mse(student_audio_x0, teacher_audio_x0, inputs.audio_loss_mask)
            audio_loss = self.config.trajectory_endpoint_loss_weight * audio_distill_loss
            if inputs.extra.get("trajectory_enabled", False):
                next_sigma = float(inputs.extra["trajectory_next_sigma"])
                student_audio_next = self.euler_step(inputs.audio, audio_pred, next_sigma)
                teacher_audio_next = self.euler_step(inputs.teacher_audio, inputs.teacher_audio_pred, next_sigma).detach()
                audio_transition_loss = self._masked_mse(student_audio_next, teacher_audio_next, inputs.audio_loss_mask)
                audio_loss = audio_loss + self.config.trajectory_transition_loss_weight * audio_transition_loss
            if self.config.clean_x0_loss_weight > 0:
                clean_audio_loss = self._masked_mse(student_audio_x0, inputs.audio_targets, inputs.audio_loss_mask)
                audio_loss = audio_loss + self.config.clean_x0_loss_weight * clean_audio_loss
            audio_quality_loss = self._quality_loss(
                pred=student_audio_x0,
                target=inputs.audio_targets,
                mask=inputs.audio_loss_mask,
            )
            audio_loss = audio_loss + audio_quality_loss

        distill_total = self.config.video_loss_weight * video_loss
        if audio_loss is not None:
            distill_total = distill_total + self.config.audio_loss_weight * audio_loss

        metrics: dict[str, Tensor] = {
            "video_distill_loss": video_distill_loss.detach(),
            "distill_loss": distill_total.detach(),
        }
        if video_transition_loss is not None:
            metrics["video_transition_loss"] = video_transition_loss.detach()
        if clean_video_loss is not None:
            metrics["clean_video_loss"] = clean_video_loss.detach()
        if self.config.quality_moment_loss_weight > 0 or self.config.quality_delta_loss_weight > 0:
            metrics["video_quality_loss"] = video_quality_loss.detach()
        if audio_distill_loss is not None:
            metrics["audio_distill_loss"] = audio_distill_loss.detach()
        if audio_transition_loss is not None:
            metrics["audio_transition_loss"] = audio_transition_loss.detach()
        if clean_audio_loss is not None:
            metrics["clean_audio_loss"] = clean_audio_loss.detach()
        if (
            audio_quality_loss is not None
            and (self.config.quality_moment_loss_weight > 0 or self.config.quality_delta_loss_weight > 0)
        ):
            metrics["audio_quality_loss"] = audio_quality_loss.detach()

        dmd_scale = self.dmd_weight_scale()
        metrics["dmd_weight_scale"] = dmd_scale
        if not self.config.enable_dmd or dmd_scale <= 0:
            metrics["generator_loss"] = distill_total.detach()
            return distill_total, metrics

        if inputs.critic_video_pred is None:
            raise ValueError("DMD generator loss requires critic video predictions")

        critic_video_x0 = self._to_x0(inputs.video, inputs.critic_video_pred).detach()
        video_grad = critic_video_x0 - teacher_video_x0
        if self.config.normalize_dmd_gradient:
            video_grad = video_grad / self._masked_mean_abs(
                student_video_x0.detach() - teacher_video_x0,
                inputs.video_loss_mask,
            )
        if self.config.dmd_gradient_clip is not None:
            video_grad = video_grad.clamp(-self.config.dmd_gradient_clip, self.config.dmd_gradient_clip)
        video_dmd_target = (student_video_x0.detach() - video_grad).detach()
        video_dmd_loss = 0.5 * self._masked_mse(student_video_x0, video_dmd_target, inputs.video_loss_mask)

        dmd_total = video_dmd_loss
        metrics["video_dmd_loss"] = video_dmd_loss.detach()
        metrics["dmd_gradient_norm"] = self._masked_mean_abs(video_grad, inputs.video_loss_mask).flatten().detach()

        if (
            self.config.with_audio
            and student_audio_x0 is not None
            and teacher_audio_x0 is not None
            and inputs.audio is not None
            and inputs.audio_loss_mask is not None
            and inputs.critic_audio_pred is not None
        ):
            critic_audio_x0 = self._to_x0(inputs.audio, inputs.critic_audio_pred).detach()
            audio_grad = critic_audio_x0 - teacher_audio_x0
            if self.config.normalize_dmd_gradient:
                audio_grad = audio_grad / self._masked_mean_abs(
                    student_audio_x0.detach() - teacher_audio_x0,
                    inputs.audio_loss_mask,
                )
            if self.config.dmd_gradient_clip is not None:
                audio_grad = audio_grad.clamp(-self.config.dmd_gradient_clip, self.config.dmd_gradient_clip)
            audio_dmd_target = (student_audio_x0.detach() - audio_grad).detach()
            audio_dmd_loss = 0.5 * self._masked_mse(student_audio_x0, audio_dmd_target, inputs.audio_loss_mask)
            dmd_total = dmd_total + self.config.audio_dmd_loss_weight * audio_dmd_loss
            metrics["audio_dmd_loss"] = audio_dmd_loss.detach()
            metrics["audio_dmd_gradient_norm"] = self._masked_mean_abs(audio_grad, inputs.audio_loss_mask).flatten().detach()

        total = dmd_scale * self.config.dmd_loss_weight * dmd_total + self.config.distill_loss_weight * distill_total
        metrics["dmd_loss"] = dmd_total.detach()
        metrics["generator_loss"] = total.detach()
        return total, metrics

    def compute_critic_loss(
        self,
        critic_video_pred: Tensor,
        critic_audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if inputs.fake_video_targets is None:
            raise ValueError("DMD critic loss requires fake_video_targets from the frozen generator")

        critic_video_x0 = self._to_x0(inputs.video, critic_video_pred)
        video_critic_loss = self._masked_regression_loss(
            critic_video_x0,
            inputs.fake_video_targets.detach(),
            inputs.video_loss_mask,
        )
        total = self.config.critic_loss_weight * video_critic_loss

        metrics: dict[str, Tensor] = {
            "video_critic_loss": video_critic_loss.detach(),
            "critic_loss": total.detach(),
        }

        if (
            self.config.with_audio
            and critic_audio_pred is not None
            and inputs.audio is not None
            and inputs.audio_loss_mask is not None
            and inputs.fake_audio_targets is not None
        ):
            critic_audio_x0 = self._to_x0(inputs.audio, critic_audio_pred)
            audio_critic_loss = self._masked_regression_loss(
                critic_audio_x0,
                inputs.fake_audio_targets.detach(),
                inputs.audio_loss_mask,
            )
            total = total + self.config.audio_critic_loss_weight * audio_critic_loss
            metrics["audio_critic_loss"] = audio_critic_loss.detach()
            metrics["critic_loss"] = total.detach()

        return total, metrics

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "training_strategy": self.config.name,
            "teacher_model_path": self.config.teacher_model_path,
            "distill_stage": self.config.distill_stage,
            "official_distilled_sigmas": ",".join(str(v) for v in self._selected_sigmas()),
            "enable_dmd": self.config.enable_dmd,
            "critic_model_path": self.critic_model_path if self.config.enable_dmd else "",
            "dmd_loss_weight": self.config.dmd_loss_weight,
            "audio_dmd_loss_weight": self.config.audio_dmd_loss_weight,
            "trajectory_distillation": self.config.trajectory_distillation,
            "trajectory_transition_loss_weight": self.config.trajectory_transition_loss_weight,
            "quality_moment_loss_weight": self.config.quality_moment_loss_weight,
            "quality_delta_loss_weight": self.config.quality_delta_loss_weight,
            "dmd_start_step": self.config.dmd_start_step,
            "dmd_ramp_steps": self.config.dmd_ramp_steps,
            "critic_warmup_steps": self.config.critic_warmup_steps,
        }

    def _selected_sigmas(self) -> tuple[float, ...]:
        stage1 = tuple(self.config.stage1_sigmas or DISTILLED_SIGMAS)
        stage2 = tuple(self.config.stage2_sigmas or STAGE_2_DISTILLED_SIGMAS)
        if self.config.distill_stage == "stage1":
            return stage1
        if self.config.distill_stage == "stage2":
            return stage2
        selected: list[float] = []
        for sigma in (*stage1, *stage2):
            if sigma not in selected:
                selected.append(float(sigma))
        if selected[-1] != 0.0:
            selected.append(0.0)
        return tuple(selected)

    def _sample_official_sigmas(self, batch_size: int, device: torch.device) -> Tensor:
        schedule = self._selected_sigmas()
        sigmas = torch.tensor(schedule[:-1], dtype=torch.float32, device=device)
        indices = self._sample_interval_indices(
            num_intervals=sigmas.numel(),
            strategy=self.config.sigma_sampling_strategy,
            shape=(batch_size,),
            device=device,
            schedule=schedule,
        )
        return sigmas[indices]

    def _trajectory_start_sigmas(self, batch_size: int, device: torch.device) -> Tensor:
        start_sigma = float(self._selected_sigmas()[0])
        return torch.full((batch_size,), start_sigma, dtype=torch.float32, device=device)

    def _sample_trajectory_step_index(self, device: torch.device) -> int:
        num_intervals = len(self._selected_sigmas()) - 1
        if self.config.trajectory_step_sampling == "cycle":
            return self._current_step % num_intervals
        index = self._sample_interval_indices(
            num_intervals=num_intervals,
            strategy=self.config.trajectory_step_sampling,
            shape=(),
            device=device,
            schedule=self._selected_sigmas(),
        )
        return int(index.item())

    @staticmethod
    def _sample_interval_indices(
        num_intervals: int,
        strategy: Literal["uniform", "interval_weighted", "late_biased"],
        shape: tuple[int, ...],
        device: torch.device,
        schedule: tuple[float, ...] | None = None,
    ) -> Tensor:
        if strategy == "uniform":
            return torch.randint(0, num_intervals, shape, device=device)
        if strategy == "late_biased":
            weights = torch.linspace(1.0, 2.0, num_intervals, device=device)
        else:
            sigma_schedule = torch.tensor(
                tuple(schedule or DISTILLED_SIGMAS)[: num_intervals + 1],
                dtype=torch.float32,
                device=device,
            )
            weights = (sigma_schedule[:-1] - sigma_schedule[1:]).abs().clamp(min=1e-6)
        num_samples = 1
        for dim in shape:
            num_samples *= dim
        flat = torch.multinomial(weights / weights.sum(), num_samples=max(1, num_samples), replacement=True)
        return flat.view(shape) if shape else flat[0]

    def trajectory_prefix_sigmas(self, inputs: ModelInputs) -> list[float]:
        if not inputs.extra.get("trajectory_enabled", False):
            return []
        schedule = inputs.extra["trajectory_sigmas"]
        step_index = int(inputs.extra["trajectory_step_index"])
        return [float(v) for v in schedule[: step_index + 1]]

    def rollout_next_sigma(self, inputs: ModelInputs) -> float:
        return float(inputs.extra["trajectory_next_sigma"])

    @staticmethod
    def with_sigma(modality: Modality | None, sigma: float, latent: Tensor | None = None) -> Modality | None:
        if modality is None:
            return None
        sigma_tensor = torch.full((modality.latent.shape[0],), sigma, dtype=torch.float32, device=modality.latent.device)
        timesteps = sigma_tensor.view(-1, 1).expand(-1, modality.latent.shape[1])
        return replace(
            modality,
            latent=modality.latent if latent is None else latent,
            sigma=sigma_tensor,
            timesteps=timesteps,
        )

    @staticmethod
    def euler_step(modality: Modality, velocity: Tensor, next_sigma: float) -> Tensor:
        sigma = modality.timesteps
        if sigma.dim() == velocity.dim() - 1:
            sigma = sigma.unsqueeze(-1)
        next_sigma_tensor = torch.full_like(sigma, float(next_sigma))
        return modality.latent + (next_sigma_tensor - sigma) * velocity

    @staticmethod
    def _to_x0(modality: Modality, velocity: Tensor) -> Tensor:
        sigmas = modality.timesteps
        if sigmas.dim() == velocity.dim() - 1:
            sigmas = sigmas.unsqueeze(-1)
        return to_denoised(modality.latent, velocity, sigmas)

    @staticmethod
    def _masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss = (pred - target).pow(2)
        mask_f = mask.unsqueeze(-1).to(dtype=loss.dtype)
        denom = (mask_f.sum(dim=(1, 2)) * loss.shape[-1]).clamp(min=1.0)
        return (loss * mask_f).sum(dim=(1, 2)) / denom

    def _masked_regression_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if self.config.critic_loss_type == "mse":
            return self._masked_mse(pred, target, mask)
        diff = pred - target
        abs_diff = diff.abs()
        delta = self.config.critic_huber_delta
        loss = torch.where(abs_diff < delta, 0.5 * diff.pow(2) / delta, abs_diff - 0.5 * delta)
        mask_f = mask.unsqueeze(-1).to(dtype=loss.dtype)
        denom = (mask_f.sum(dim=(1, 2)) * loss.shape[-1]).clamp(min=1.0)
        return (loss * mask_f).sum(dim=(1, 2)) / denom

    def _quality_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        quality = pred.new_zeros(pred.shape[0])
        if self.config.quality_moment_loss_weight > 0:
            quality = quality + self.config.quality_moment_loss_weight * self._masked_moment_loss(pred, target, mask)
        if self.config.quality_delta_loss_weight > 0:
            quality = quality + self.config.quality_delta_loss_weight * self._masked_delta_loss(pred, target, mask)
        return quality

    @staticmethod
    def _masked_moment_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=pred.dtype)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pred_mean = (pred * mask_f).sum(dim=1) / denom
        target_mean = (target * mask_f).sum(dim=1) / denom
        pred_var = ((pred - pred_mean.unsqueeze(1)).pow(2) * mask_f).sum(dim=1) / denom
        target_var = ((target - target_mean.unsqueeze(1)).pow(2) * mask_f).sum(dim=1) / denom
        mean_loss = (pred_mean - target_mean).pow(2).mean(dim=1)
        std_loss = (pred_var.clamp(min=1e-6).sqrt() - target_var.clamp(min=1e-6).sqrt()).pow(2).mean(dim=1)
        return mean_loss + std_loss

    @staticmethod
    def _masked_delta_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if pred.shape[1] < 2:
            return pred.new_zeros(pred.shape[0])
        pred_delta = pred[:, 1:] - pred[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        delta_mask = mask[:, 1:] & mask[:, :-1]
        return DistillTextToVideoStrategy._masked_mse(pred_delta, target_delta, delta_mask)


    @staticmethod
    def _masked_mean_abs(value: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=value.dtype)
        denom = (mask_f.sum(dim=(1, 2)) * value.shape[-1]).clamp(min=1.0)
        mean_abs = (value.abs() * mask_f).sum(dim=(1, 2), keepdim=True) / denom.view(-1, 1, 1)
        return mean_abs.clamp(min=1e-6)
