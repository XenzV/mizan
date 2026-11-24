
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

__all__ = ["ModelSpec", "PRESETS", "resolve_model", "spec_from_hf_config"]


@dataclass(frozen=True)
class ModelSpec:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    head_dim: int
    num_key_value_heads: int | None = None
    tie_word_embeddings: bool = False
    name: str = "custom"
    source: str = "custom"

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            object.__setattr__(self, "num_key_value_heads", self.num_attention_heads)

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        assert self.num_key_value_heads is not None
        return self.num_key_value_heads * self.head_dim


def _spec(name: str, h: int, i: int, layers: int, heads: int, kv: int,
          vocab: int, hd: int, tie: bool = False) -> ModelSpec:
    return ModelSpec(
        name=name, source="preset", hidden_size=h, intermediate_size=i,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=kv, vocab_size=vocab, head_dim=hd,
        tie_word_embeddings=tie,
    )


PRESETS: dict[str, ModelSpec] = {
    s.name: s for s in [
        _spec("llama-3.1-8b", 4096, 14336, 32, 32, 8, 128256, 128),
        _spec("llama-3.2-1b", 2048, 8192, 16, 32, 8, 128256, 64, True),
        _spec("llama-3.2-3b", 3072, 8192, 28, 24, 8, 128256, 128, True),
        _spec("llama-3.3-70b", 8192, 28672, 80, 64, 8, 128256, 128),
        _spec("mistral-7b-v0.3", 4096, 14336, 32, 32, 8, 32768, 128),
        _spec("qwen2.5-0.5b", 896, 4864, 24, 14, 2, 152064, 64),
        _spec("qwen2.5-1.5b", 1536, 8960, 28, 12, 2, 152064, 128),
        _spec("qwen2.5-7b", 3584, 18944, 28, 28, 4, 152064, 128),
        _spec("qwen2.5-14b", 5120, 13824, 48, 40, 8, 152064, 128),
        _spec("qwen2.5-32b", 5120, 27648, 64, 40, 8, 152064, 128),
        _spec("qwen3-8b", 4096, 12288, 36, 32, 8, 151936, 128),
        _spec("qwen3-14b", 5120, 17408, 48, 40, 8, 151936, 128),
        _spec("gemma-2-9b", 3584, 14336, 42, 16, 8, 262144, 256),
        _spec("gemma-3-4b", 2560, 10240, 34, 8, 4, 262144, 256, True),
        _spec("gemma-3-12b", 3840, 15360, 48, 16, 8, 262144, 256, True),
        _spec("gpt2", 768, 3072, 12, 12, 12, 50257, 64, True),
    ]
}

ALIASES = {
    "llama-8b": "llama-3.1-8b",
    "llama-70b": "llama-3.3-70b",
    "mistral-7b": "mistral-7b-v0.3",
    "qwen-7b": "qwen2.5-7b",
    "qwen-14b": "qwen2.5-14b",
    "qwen-32b": "qwen2.5-32b",
    "gemma-9b": "gemma-2-9b",
    "gemma-3-4b-it": "gemma-3-4b",
    "llama-3.1-70b": "llama-3.3-70b",
    "deepseek-r1-7b": "qwen2.5-7b",
    "deepseek-r1-distill-qwen-7b": "qwen2.5-7b",
}


def spec_from_hf_config(path: str | Path) -> ModelSpec:
    p = Path(path).expanduser()
    cfg_path = p / "config.json" if p.is_dir() else p
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {cfg_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {cfg_path}: {exc}") from exc

    try:
        h = int(raw["hidden_size"])
        layers = int(raw["num_hidden_layers"])
        heads = int(raw["num_attention_heads"])
        vocab = int(raw["vocab_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{cfg_path} is missing a required architecture field: {exc}") from exc

    i = raw.get("intermediate_size") or raw.get("ffn_dim") or raw.get("intermediate_dim")
    if i is None:
        raise ValueError(f"{cfg_path} has no intermediate_size; pass --intermediate-size yourself")
    hd = raw.get("head_dim") or (h // heads)
    kv = raw.get("num_key_value_heads") or raw.get("num_kv_heads") or heads
    tie = bool(raw.get("tie_word_embeddings", False))
    return ModelSpec(
        hidden_size=h, intermediate_size=int(i), num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=int(kv), vocab_size=vocab,
        head_dim=int(hd), tie_word_embeddings=tie,
        name=cfg_path.parent.name or "local", source=f"config.json:{cfg_path}",
    )


def resolve_model(model: str | None, *, hidden_size: int | None = None,
                  intermediate_size: int | None = None, num_hidden_layers: int | None = None,
                  num_attention_heads: int | None = None, vocab_size: int | None = None,
                  head_dim: int | None = None, num_key_value_heads: int | None = None,
                  tie_word_embeddings: bool = False) -> ModelSpec:
    if model:
        key = model.strip().lower()
        if Path(model).exists():
            return spec_from_hf_config(model)
        key = key.rsplit("/", 1)[-1]
        key = re.sub(r"-(instruct|chat|it|base|v\d+(\.\d+)?)$", "", key)
        key = ALIASES.get(key, key)
        if key in PRESETS:
            return PRESETS[key]
        for preset_name, spec in PRESETS.items():
            if preset_name in key or key in preset_name:
                return spec

    explicit = {
        "hidden_size": hidden_size, "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers, "num_attention_heads": num_attention_heads,
        "vocab_size": vocab_size,
    }
    if all(v is not None for v in explicit.values()):
        return ModelSpec(
            hidden_size=int(hidden_size), intermediate_size=int(intermediate_size),
            num_hidden_layers=int(num_hidden_layers),
            num_attention_heads=int(num_attention_heads),
            num_key_value_heads=num_key_value_heads,
            vocab_size=int(vocab_size), head_dim=int(head_dim or hidden_size // num_attention_heads),
            tie_word_embeddings=tie_word_embeddings, name="custom", source="flags",
        )

    if model:
        raise ValueError(
            f"unknown model {model!r}. Use `mizan presets` to list presets, "
            "a path to a folder containing config.json, or pass all five "
            "architecture flags (--hidden-size, --intermediate-size, "
            "--num-hidden-layers, --num-attention-heads, --vocab-size)."
        )
    raise ValueError(
        "no model given. Use --model PRESET|PATH or pass the five architecture flags."
    )


def with_overrides(spec: ModelSpec, **changes: object) -> ModelSpec:
    return replace(spec, **changes)
