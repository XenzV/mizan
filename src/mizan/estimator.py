
from __future__ import annotations

from dataclasses import dataclass, field

from .models import ModelSpec

__all__ = [
    "WeighInput", "Result", "BYTES_PER_PARAM", "OPTIMIZER_BYTES",
    "weigh", "param_counts", "lora_param_count", "verdict",
]

MIB = 1024 ** 2
GIB = 1024 ** 3

BYTES_PER_PARAM: dict[str, float] = {
    "4bit": 0.5625,
    "8bit": 1.0625,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp32": 4.0,
    "none": 2.0,
}

OPTIMIZER_BYTES: dict[str, float] = {
    "adamw": 12.0,
    "adamw_torch": 12.0,
    "adamw_8bit": 6.0,
    "paged_adamw_8bit": 6.0,
    "adafactor": 4.0,
    "sgd": 8.0,
}

LORA_TARGETS = ("q", "k", "v", "o", "gate", "up", "down")


@dataclass
class WeighInput:
    spec: ModelSpec
    seq_len: int = 1024
    batch_size: int = 1
    grad_accum: int = 1
    quant: str = "4bit"
    lora_rank: int = 16
    lora_targets: tuple[str, ...] = LORA_TARGETS
    full_finetune: bool = False
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    loss_upcast_fp32: bool = True
    optimizer: str = "adamw"
    cuda_overhead_mb: int = 700
    fragmentation: float = 0.08
    activation_dtype_bytes: int = 2


@dataclass
class Result:
    components: dict[str, float] = field(default_factory=dict)
    total_mib: float = 0.0
    params: dict[str, int] = field(default_factory=dict)
    input: WeighInput | None = None

    def component(self, name: str) -> float:
        return self.components.get(name, 0.0)

    def fits(self, gpu_gib: float, headroom: float = 0.05) -> bool:
        return self.total_mib <= gpu_gib * GIB * (1.0 - headroom) / MIB

    def headroom_gib(self, gpu_gib: float) -> float:
        return gpu_gib - self.total_mib / 1024.0

    def as_dict(self) -> dict:
        assert self.input is not None
        return {
            "model": self.input.spec.name,
            "total_gib": round(self.total_mib / 1024, 3),
            "components_gib": {k: round(v / 1024, 3) for k, v in self.components.items()},
            "parameters": self.params,
            "config": {
                "seq_len": self.input.seq_len,
                "batch_size": self.input.batch_size,
                "grad_accum": self.input.grad_accum,
                "quant": self.input.quant,
                "lora_rank": self.input.lora_rank,
                "gradient_checkpointing": self.input.gradient_checkpointing,
                "full_finetune": self.input.full_finetune,
                "optimizer": self.input.optimizer,
            },
        }


def param_counts(spec: ModelSpec) -> dict[str, int]:
    h, i = spec.hidden_size, spec.intermediate_size
    q, kv = spec.q_dim, spec.kv_dim
    attn = h * q + 2 * h * kv + q * h
    mlp = 3 * h * i
    norms = 2 * h
    per_layer = attn + mlp + norms
    embed = spec.vocab_size * h
    lm_head = 0 if spec.tie_word_embeddings else embed
    transformer = per_layer * spec.num_hidden_layers
    total = transformer + embed + lm_head + h
    return {
        "embedding": embed,
        "lm_head": lm_head,
        "attention_per_layer": attn,
        "mlp_per_layer": mlp,
        "transformer": transformer,
        "total": total,
    }


def lora_param_count(spec: ModelSpec, rank: int,
                     targets: tuple[str, ...] = LORA_TARGETS) -> int:
    h, i = spec.hidden_size, spec.intermediate_size
    q, kv = spec.q_dim, spec.kv_dim
    per_matrix = {
        "q": h + q, "k": h + kv, "v": h + kv, "o": q + h,
        "gate": h + i, "up": h + i, "down": i + h,
    }
    unknown = set(targets) - set(per_matrix)
    if unknown:
        raise ValueError(f"unknown LoRA target(s): {', '.join(sorted(unknown))}")
    return rank * sum(per_matrix[t] for t in targets) * spec.num_hidden_layers


def _activation_mib(cfg: WeighInput) -> float:
    spec = cfg.spec
    h, i = spec.hidden_size, spec.intermediate_size
    q, kv = spec.q_dim, spec.kv_dim
    b, s, layers = cfg.batch_size, cfg.seq_len, spec.num_hidden_layers
    ab = cfg.activation_dtype_bytes

    per_token_layer = 4 * h + 2 * q + 2 * kv + 2 * i
    if not cfg.flash_attention:
        per_token_layer += spec.num_attention_heads * s

    full_run = layers * b * s * per_token_layer * ab
    if cfg.gradient_checkpointing:
        kept = layers * b * s * h * ab
        recompute = b * s * per_token_layer * ab
        bytes_ = kept + recompute
    else:
        bytes_ = full_run

    logits = b * s * spec.vocab_size * (4 if cfg.loss_upcast_fp32 else 2)
    return (bytes_ + logits) / MIB


def weigh(cfg: WeighInput) -> Result:
    if cfg.seq_len <= 0 or cfg.batch_size <= 0 or cfg.grad_accum <= 0:
        raise ValueError("seq_len, batch_size and grad_accum must be positive")
    if cfg.quant not in BYTES_PER_PARAM:
        raise ValueError(f"unknown quant {cfg.quant!r}; choose from {', '.join(BYTES_PER_PARAM)}")
    if cfg.optimizer not in OPTIMIZER_BYTES:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    if cfg.lora_rank <= 0:
        raise ValueError("lora_rank must be positive")

    spec = cfg.spec
    params = param_counts(spec)
    total_params = params["total"]

    if cfg.full_finetune:
        trainable = total_params
        lora = 0
    else:
        trainable = lora_param_count(spec, cfg.lora_rank, cfg.lora_targets)
        lora = trainable

    base_mib = total_params * BYTES_PER_PARAM[cfg.quant] / MIB
    trainable_mib = trainable * (2 + 2 + OPTIMIZER_BYTES[cfg.optimizer]) / MIB
    act_mib = _activation_mib(cfg)

    components = {
        "base_weights": base_mib,
        "lora_weights": lora * 2 / MIB,
        "trainable_states": trainable_mib,
        "activations": act_mib,
        "cuda_context": float(cfg.cuda_overhead_mb),
    }
    subtotal = sum(components.values())
    components["fragmentation"] = subtotal * cfg.fragmentation
    total = subtotal + components["fragmentation"]

    params.update({"lora": lora, "trainable": trainable})
    return Result(components=components, total_mib=total, params=params, input=cfg)


def verdict(result: Result, gpu_gib: float | None, headroom: float = 0.05) -> tuple[str, str]:
    if gpu_gib is None:
        return "UNKNOWN", "no GPU size given; pass --gpu or let `mizan doctor` detect it"
    need = result.total_mib / 1024
    limit = gpu_gib * (1 - headroom)
    if need <= limit:
        margin = limit - need
        if margin < gpu_gib * 0.10:
            return "TIGHT", f"fits, but only ~{margin:.2f} GiB of slack ({margin / gpu_gib:.0%})"
        return "OK", f"~{margin:.2f} GiB of headroom under the {headroom:.0%} safety line"
    return "OVER", f"needs ~{need:.2f} GiB, card has {gpu_gib:.1f} GiB (limit {limit:.2f})"
