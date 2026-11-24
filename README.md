# Mizan

**Weigh your fine-tune before you run it.**

[![CI](https://github.com/XenzV/mizan/actions/workflows/ci.yml/badge.svg)](https://github.com/XenzV/mizan/actions/workflows/ci.yml) [![Release](https://img.shields.io/github/v/release/XenzV/mizan)](https://github.com/XenzV/mizan/releases) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Mizan is a small CLI that estimates peak GPU memory for LoRA / QLoRA training from architecture numbers and a training config. It runs in milliseconds, has no runtime dependencies, and never loads model weights or calls a server.

[العربية](README.ar.md) · [Method](docs/METHOD.md) · [Issues](https://github.com/XenzV/mizan/issues)

## The problem

You write a config, start a fine-tune, walk away, and come back 40 minutes later to `CUDA out of memory` at step 3. The config had no syntax error. It just did not fit.

Mizan answers one question before you spend the GPU time: **will this config fit, and if not, what should change?**

## Install

```bash
git clone https://github.com/XenzV/mizan.git
cd mizan
pip install -e .
mizan --help
```

Python 3.10+. No PyTorch, no CUDA toolkit, no model download. YAML configs work with the built-in parser; `pip install -e ".[yaml]"` uses PyYAML when you want full YAML syntax.

## Use

```bash
mizan weigh --model llama-3.1-8b --gpu 8 --seq 512
mizan weigh --config soup.yaml --gpu 8
mizan fit --model qwen2.5-7b --gpu 8 --seq 2048
mizan check --config soup.yaml --strict
mizan doctor
mizan presets
mizan init --model llama-3.1-8b --gpu 8 --out mizan.yaml
mizan report --config soup.yaml --gpu 8 --out preflight.md
```

`--model` accepts a preset, an alias (`llama-8b`, `qwen-7b`), a Hugging Face id like `meta-llama/Llama-3.1-8B-Instruct`, or a local folder containing `config.json`. You can also pass the five architecture flags directly.

### Weigh

```
MIZAN / WEIGH
  Model   llama-3.1-8b (preset, 8.03B params)
  Config  1x512 tok  | 4bit  | LoRA r16  | ckpt on

  base_weights           4.21 GiB  █████████████░░░░░
  lora_weights           0.08 GiB  ░░░░░░░░░░░░░░░░░░
  trainable_states       0.62 GiB  ██░░░░░░░░░░░░░░░░
  activations            0.42 GiB  █░░░░░░░░░░░░░░░░░
  cuda_context           0.68 GiB  ██░░░░░░░░░░░░░░░░
  fragmentation          0.48 GiB
  --------------------------------------------------------------
  ESTIMATED PEAK         6.50 GiB

  VERDICT  OK  (8.0 GiB GPU)
         ~1.10 GiB of headroom under the 5% safety line
```

`--json` gives a machine-readable breakdown for CI.

### Fit

Sweeps quantization, micro-batch size, and gradient checkpointing, then prints the largest config that stays under `--target-util` (default 90%) plus a config snippet you can paste.

```
$ mizan fit -m llama-3.1-8b --gpu 8 --seq 512
  Recommended: 4bit, micro-batch 2, grad-accum 8, ckpt on
  Estimated peak 6.95 GiB
```

### Check

Lints a config for unknown keys, missing data, invalid numbers, bad LoRA targets, and quantization mistakes. Exit code `1` on error, `--strict` promotes warnings. Made for CI:

```yaml
- run: pip install -e .
- run: mizan check --config soup.yaml --strict
- run: mizan weigh --config soup.yaml --gpu 8 --json > preflight.json
```

## How the estimate is built

| Component | What it accounts for |
| --- | --- |
| `base_weights` | total params × bytes/param (NF4 0.5625, int8 1.0625, bf16 2) |
| `lora_weights` | rank × (in + out) for each targeted projection × layers |
| `trainable_states` | weights + gradients + optimizer/master copies (AdamW: 12 B/param) |
| `activations` | saved layer tensors, one recomputed layer when checkpointing, and the logits buffer |
| `cuda_context` | fixed runtime allowance, default 700 MiB (`--cuda-overhead-mb`) |
| `fragmentation` | 8% of the subtotal by default (`--fragmentation`) |

Parameter counts are derived from the architecture, not from a downloaded checkpoint. For Llama-3.1-8B this gives 8.03B; LoRA r16 on all projections gives 41.9M trainable params. See [docs/METHOD.md](docs/METHOD.md) for every formula and knob.

## Limitations, stated plainly

- This is an **analytical estimate**, not a profiler. Expect roughly ±20%. Confirm with a 20-step run before a multi-hour job.
- It models standard QLoRA/LoRA. Layer streaming, CPU/NVMe offload, DeepSpeed ZeRO and tensor parallelism are not modeled.
- MoE experts, speculative decoding, and non-standard attention implementations are outside the model.
- `--headroom` is a policy knob, not a physical constant. Default is 5%.

## Mizan vs. a trainer

Mizan does not train models. A trainer needs Mizan's question answered first. A typical Soup flow:

```bash
mizan check --config soup.yaml --strict
mizan weigh --config soup.yaml --gpu 8
mizan fit   --config soup.yaml --gpu 8 --seq 1024
soup train  --config soup.yaml
```

## Develop

```bash
pip install -e ".[dev]"
pytest -q
python -m unittest discover -s tests -q
```

MIT licensed. Contributions and measured corrections to the formulas are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
