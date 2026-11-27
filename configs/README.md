# Serving configs

Each YAML file here is one deployment: the model key (`model:`), where each node runs (`node_groups`), and the model's and resources' settings. The [supported models matrix](https://m-star-labs.mintlify.site/models#matrix) lists every model key with its default config and GPU count. Config keys are described in [the config reference](https://m-star-labs.mintlify.site/reference/config).

## How `mstar serve` picks a config

`mstar serve <model>` looks the key up in `DEFAULT_CONFIGS` in `mstar/cli/main.py`, then loads that file from the installed package (`mstar/default_configs/`, which a wheel build copies from `configs/*.yaml`), else from this directory. `--config <path>` loads another file; the model key is still required:

```bash
mstar serve bagel                                        # configs/bagel_single_gpu.yaml
mstar serve bagel --config configs/bagel_cfg_parallel.yaml
```

The file named after a model is not always its default:

| Key | Default config | `<model>.yaml` |
|---|---|---|
| `bagel` | `bagel_single_gpu.yaml` (1 GPU) | `bagel.yaml` splits the model over 2 GPUs |
| `orpheus` | `orpheus_colocated.yaml` (1 GPU) | `orpheus.yaml` uses 2 GPUs |
| `qwen3_omni` | `qwen3omni_2gpu.yaml` (2 GPUs) | `qwen3omni.yaml` uses 3 GPUs |

Some file prefixes are not the model key: `qwen3omni_*` serve `qwen3_omni`, `qwen3tts_*` serve the `qwen3_tts*` keys, and `cosmos3_nano*` serve `cosmos3`.

## Suffixes

| Suffix | Meaning | Example |
|---|---|---|
| `_single_gpu`, `_colocated` | Every node on one GPU. | `bagel_single_gpu.yaml` |
| `_split` | Two or more ranks on one GPU through `rank_devices`, most of them under NVIDIA MPS (`mps: true`). One GPU, not two. | `whisper_large_split.yaml` |
| `_2gpu`, `_4gpu` | GPU count. | `qwen3omni_2gpu.yaml` |
| `_tp2`, `_tp4`, `_tp8` | Tensor parallelism of that degree (`tp_size`). | `ltx2_5_tp2.yaml` |
| `_sp2`, `_sp4`, `_sp8` | Sequence parallelism of that degree (`sp_size`). | `minimax_h3_sp8.yaml` |
| `_dp2` | One replica per GPU; each request runs on one of them. | `wan22_dp2.yaml` |
| `_cfg_parallel`, `_cfg2` | Guidance (CFG) branches on separate ranks: `cfg_size: 2` on Wan2.2, separate branch nodes on BAGEL. | `wan22_cfg_parallel_2gpu.yaml` |
| `_ep2` | Expert parallelism. | `qwen3omni_thinker_ep2.yaml` |
| `_disagg`, `_pd_disaggregated`, `_img_disaggregated` | Stages of the model on separate GPUs: prefill and decode, image generation, or a duplex model's listening and speaking loops. | `orpheus_pd_disaggregated.yaml` |

Other suffixes name a variant rather than a placement: `_h100` (sized for 80 GB GPUs), `_xpu` (Intel XPU), `_mtp` (multi-token prediction), `_longctx`, `_large_kv`, `_bf16`, `_fast`, `_ar` (windowed video), `_360p` and `_droid` (a DROID checkpoint).

`configs/test/` holds test configs. They are not copied into the wheel.
