# Estimation method

Mizan takes six architecture integers and turns them into a VRAM estimate. No weights are read.

## Parameter counts

For hidden size `h`, intermediate size `i`, layers `L`, attention heads `H`, KV heads `K`, head dim `d`, vocab `V`:

```
q_dim = H * d
kv_dim = K * d

attention_per_layer = h*q_dim + 2*h*kv_dim + q_dim*h
mlp_per_layer       = 3*h*i
norms_per_layer     = 2*h
transformer         = L * (attention + mlp + norms)
embedding           = V*h
lm_head             = V*h, unless tie_word_embeddings
total               = transformer + embedding + lm_head + h
```

This reproduces 8.03B for Llama-3.1-8B and 7.62B for Qwen2.5-7B.

## LoRA trainable parameters

For rank `r` and targeted projections:

```
q     = r*(h + q_dim)
k, v  = r*(h + kv_dim) each
o     = r*(q_dim + h)
gate, up, down = r*(h + i) / r*(h + i) / r*(i + h)

lora_total = L * sum(selected)
```

Llama-3.1-8B with r=16 on all seven projections gives 41,943,040 trainable parameters.

## Memory components

### Base weights

`total_params * bytes_per_param`, with:

| quant | bytes/param | rationale |
| --- | ---: | --- |
| `4bit` | 0.5625 | 4-bit nibble + fp8 scale per 64 weights |
| `8bit` | 1.0625 | int8 byte + fp8 scale per 64 weights |
| `bf16` / `fp16` / `none` | 2.0 | half precision |
| `fp32` | 4.0 | full precision |

### Trainable states

`trainable_params * (2 + 2 + optimizer_bytes)` — bf16 weights, bf16 gradients, and optimizer state plus master copies:

| optimizer | bytes/param | rationale |
| --- | ---: | --- |
| `adamw` | 12 | fp32 master + m + v |
| `adamw_8bit` / `paged_adamw_8bit` | 6 | fp32 master + two int8 moments |
| `adafactor` | 4 | fp32 master + factored moments |
| `sgd` | 8 | fp32 master + momentum |

Full fine-tuning sets `trainable_params = total_params`.

### Activations

Saved tensors per token per layer:

```
s = 4*h + 2*q_dim + 2*kv_dim + 2*i
s += H * seq_len   when --no-flash-attention (materialised scores)
```

Without checkpointing: `L * b * s_tokens * s * act_bytes`.
With checkpointing: `L * b * s_tokens * h * act_bytes` (one input kept per layer) plus `b * s_tokens * s * act_bytes` (one recomputed layer).

Logits are added as `b * s_tokens * V * 4` when the loss upcasts to fp32, otherwise `* 2`.

### Runtime and fragmentation

A fixed `cuda_overhead_mb` (default 700) covers CUDA context and small workspaces. Then `fragmentation` (default 8%) is added on top of the subtotal.

## Verdict

```
limit = gpu_gib * (1 - headroom)          # headroom default 0.05
OK    when estimate <= limit and headroom above 10% of the card
TIGHT when estimate <= limit but close
OVER  when estimate > limit
```

`fit` repeats the estimate over quantization ∈ {4bit, 8bit, bf16, fp32}, checkpointing ∈ {off, on}, and batch sizes ∈ {16, 8, 4, 2, 1}, keeps the first fitting batch per combination, and ranks survivors by a throughput proxy `batch * quant_speed * (0.78 if checkpointing else 1)`. The proxy is a heuristic, not a benchmark.

## What is deliberately not modeled

- Layer streaming / host-RAM base storage (Soup's beta feature).
- CPU or NVMe offload, ZeRO sharding, tensor/pipeline parallelism.
- MoE expert routing, quantization-aware training, speculative decoding.
- Kernel-specific workspace sizes beyond the fixed allowance.

If you measure a real peak that differs, open an issue with the architecture, config, GPU, and measured number. Corrections to the constants are welcome.
