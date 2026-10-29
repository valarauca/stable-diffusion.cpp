# SeedVR2

SeedVR2 restores a video in a single diffusion step. Each window of frames is
encoded by its causal video VAE, denoised once by the 7B NaDiT conditioned on
the low quality latent, decoded, color matched to the input, and stitched to
the previous window.

## Model files

- `--diffusion-model`: SeedVR2 7B DiT (`seedvr2_ema_7b_fp16.safetensors` or
  `seedvr2_ema_7b_sharp_fp16.safetensors`)
- `--vae`: SeedVR2 video VAE (`ema_vae_fp16.safetensors`)
- `--text-embedding`: the fixed positive text embedding `pos_emb.pt`

The 3B DiT is not supported yet.

### Download weights

- DiT and VAE: https://huggingface.co/numz/SeedVR2_comfyUI/tree/main
- `pos_emb.pt`: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler (repository root)

## Restore a video

Extract the frames to a directory, restore them, and write a video:

```sh
ffmpeg -i input.mp4 frames/%05d.png

./bin/sd-cli -M vid_gen \
  --diffusion-model seedvr2_ema_7b_sharp_fp16.safetensors \
  --vae ema_vae_fp16.safetensors \
  --text-embedding pos_emb.pt \
  --ref-video frames \
  --fps 24 --seed 42 --fa \
  -o restored.webm
```

Frames are read from `--ref-video` in lexicographical order, and the output
keeps the input size unless `resolution` is set. A `--ref-video-audio` track is
passed through to the output.

## Options

Restore options go in `--extra-sample-args` as comma separated `key=value`
pairs:

| key | default | meaning |
|---|---|---|
| `window` | `21` | frames per window, must be 4n+1 |
| `overlap` | `4` | frames blended between windows, at most half the window |
| `resolution` | `0` | output short side in pixels, `0` keeps the input size |
| `color` | `lab` | color correction against the input: `lab`, `wavelet`, `adain` or `none` |

For example `--extra-sample-args "resolution=1080,color=wavelet"` upscales to a
1080 pixel short side with wavelet color correction.

Large frames need VAE tiling, for example `--vae-tiling --vae-tile-size 512`.
