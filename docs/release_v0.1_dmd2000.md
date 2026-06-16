# Release v0.1-dmd2000

This release contains the DMD2000 LoRA checkpoint and the preprocessed
audio-video latent dataset used by the LTX2.3 8-step DMD distillation
experiment.

## Assets

Upload all files from:

```text
release_assets/v0.1-dmd2000/
```

Expected assets:

```text
dmd2000_lora_weights_step_02000.safetensors
full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-000
full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-001
full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-002
full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-003
full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-004
full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-005
full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-006
full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-007
full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-008
full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-009
full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-010
full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-011
README_RELEASE_ASSETS.md
SHA256SUMS.txt
```

Each dataset part is below the GitHub Release asset limit of 2 GiB.

## Reconstruct Dataset

```bash
cat full-modality-video-caption_preprocessed_1280x704x121_av.tar.part-* > full-modality-video-caption_preprocessed_1280x704x121_av.tar
tar -xf full-modality-video-caption_preprocessed_1280x704x121_av.tar
sha256sum -c SHA256SUMS.txt
```

The extracted dataset directory is:

```text
preprocessed_1280x704x121_av/
```

## Notes

- The LoRA is a lightweight adapter for the official LTX2.3 distilled model.
- The original LTX2.3 model weights are not included in this release.
- The dataset is a preprocessed latent dataset derived from
  `ngqtrung/full-modality-video-caption`.
