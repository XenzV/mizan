
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .estimator import BYTES_PER_PARAM, LORA_TARGETS

try:
    import tomllib as _toml
except ModuleNotFoundError:
    try:
        import tomli as _toml
    except ModuleNotFoundError:
        _toml = None

__all__ = ["load_config", "lint_config", "Issue", "KNOWN_KEYS", "KNOWN_TASKS"]

KNOWN_KEYS = {
    "base", "model", "task", "backend", "data", "training", "output", "mizan",
}
KNOWN_TASKS = {"sft", "dpo", "grpo", "ppo", "kto", "orpo", "simpo", "ipo",
               "bco", "pretrain", "classification", "reward", "distillation"}
KNOWN_TRAINING_KEYS = {
    "epochs", "lr", "learning_rate", "batch_size", "micro_batch_size",
    "gradient_accumulation", "grad_accum", "seq_len", "max_length",
    "lora", "quantization", "gradient_checkpointing", "full_finetune",
    "optimizer", "bf16", "flash_attention",
}
KNOWN_DATA_KEYS = {"train", "eval", "format", "val_split"}
KNOWN_MIZAN_KEYS = {"gpu_gib", "seq_len", "quant", "optimizer", "flash_attention",
                   "loss_upcast_fp32", "cuda_overhead_mb", "fragmentation", "calibration"}


@dataclass
class Issue:
    level: str
    path: str
    message: str
    fix: str = ""

    def line(self) -> str:
        tail = f"  -> {self.fix}" if self.fix else ""
        return f"[{self.level.upper():7}] {self.path}: {self.message}{tail}"


def _strip_comment(line: str) -> str:
    quoted: str | None = None
    for idx, ch in enumerate(line):
        if quoted:
            if ch == quoted:
                quoted = None
        elif ch in "\"'":
            quoted = ch
        elif ch == "#" and (idx == 0 or line[idx - 1].isspace()):
            return line[:idx]
    return line


def _scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    if text[0] in "\"'" and text[-1] == text[0] and len(text) >= 2:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text[1:-1]
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "~", ""):
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_scalar(x) for x in inner.split(",")] if inner else []
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        out: dict[str, Any] = {}
        for part in inner.split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                out[str(_scalar(k))] = _scalar(v)
        return out
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _split_key(text: str) -> tuple[str, str] | None:
    quoted: str | None = None
    for idx, ch in enumerate(text):
        if quoted:
            if ch == quoted:
                quoted = None
        elif ch in "\"'":
            quoted = ch
        elif ch == ":" and (idx + 1 == len(text) or text[idx + 1].isspace()):
            return text[:idx].strip().strip("\"'"), text[idx + 1:].strip()
    return None


def _mini_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip() or line.strip() in ("---", "..."):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            stack = [(-1, root)]
        parent = stack[-1][1]
        if stripped.startswith("- "):
            item = _scalar(stripped[2:])
            if isinstance(parent, list):
                parent.append(item)
            continue
        pair = _split_key(stripped)
        if pair is None:
            raise ValueError(f"YAML line not understood: {raw!r}")
        key, value = pair
        if value:
            parent[key] = _scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _mini_toml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current = root
    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            path = line[1:-1].strip().split(".")
            current = root
            for part in path:
                key = part.strip().strip("\"'")
                child = current.get(key)
                if not isinstance(child, dict):
                    child = {}
                    current[key] = child
                current = child
            continue
        if "=" not in line:
            raise ValueError(f"TOML line not understood: {raw!r}")
        key, value = line.split("=", 1)
        current[key.strip().strip("\"'")] = _scalar(value)
    return root


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix == ".toml":
        if _toml is None:
            data = _mini_toml(text)
        else:
            with p.open("rb") as fh:
                data = _toml.load(fh)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(text)
        except ImportError:
            data = _mini_yaml(text)
    else:
        raise ValueError(f"unsupported config extension {suffix!r}; use .json, .toml or .yaml")
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a mapping at the top level")
    return data


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def lint_config(cfg: dict[str, Any], *, strict: bool = False) -> list[Issue]:
    issues: list[Issue] = []

    for key in cfg:
        if key not in KNOWN_KEYS:
            closest = next((k for k in KNOWN_KEYS if k.startswith(key[:3])), None)
            issues.append(Issue(
                "error", key, "unknown top-level key",
                f"did you mean {closest!r}?" if closest else f"known keys: {', '.join(sorted(KNOWN_KEYS))}",
            ))

    model = cfg.get("base") or cfg.get("model")
    if not model:
        issues.append(Issue("error", "base", "no base model declared", "set base: llama-3.1-8b or a local path"))
    elif not isinstance(model, str):
        issues.append(Issue("error", "base", "must be a string"))

    task = cfg.get("task")
    if task is not None and task not in KNOWN_TASKS:
        issues.append(Issue("error", "task", f"unknown task {task!r}", f"choose from {', '.join(sorted(KNOWN_TASKS))}"))

    data = cfg.get("data", {})
    if not isinstance(data, dict):
        issues.append(Issue("error", "data", "must be a mapping"))
    else:
        for key in data:
            if key not in KNOWN_DATA_KEYS:
                issues.append(Issue("error", f"data.{key}", "unknown key"))
        if not data.get("train"):
            issues.append(Issue("error", "data.train", "no training dataset declared"))

    training = cfg.get("training", {})
    if not isinstance(training, dict):
        issues.append(Issue("error", "training", "must be a mapping"))
    else:
        for key in training:
            if key not in KNOWN_TRAINING_KEYS:
                issues.append(Issue("error", f"training.{key}", "unknown key"))
        for key in ("epochs", "lr", "learning_rate", "batch_size", "gradient_accumulation",
                    "grad_accum", "seq_len", "max_length"):
            if key in training:
                val = _num(training[key])
                if val is None and str(training[key]) != "auto":
                    issues.append(Issue("error", f"training.{key}", f"expected a number, got {training[key]!r}"))
                elif val is not None and val <= 0:
                    issues.append(Issue("error", f"training.{key}", "must be positive"))
        if str(training.get("batch_size", "")).lower() == "auto":
            issues.append(Issue("info", "training.batch_size", "'auto' is not resolved by mizan",
                                "run `mizan fit` and paste the suggested batch_size"))

        quant = training.get("quantization", "4bit")
        if quant not in BYTES_PER_PARAM:
            issues.append(Issue("error", "training.quantization", f"unknown quantization {quant!r}",
                                f"choose from {', '.join(BYTES_PER_PARAM)}"))

        lora = training.get("lora", {})
        if lora is not None:
            if not isinstance(lora, dict):
                issues.append(Issue("error", "training.lora", "must be a mapping"))
            else:
                rank = _num(lora.get("r", 16))
                if rank is not None and (rank <= 0 or rank != int(rank)):
                    issues.append(Issue("error", "training.lora.r", "must be a positive integer"))
                if rank is not None and rank > 256:
                    issues.append(Issue("warning", "training.lora.r", f"rank {rank} is unusually large",
                                        "ranks above 256 rarely help and cost memory"))
                targets = lora.get("target_modules")
                if targets is not None:
                    if isinstance(targets, str):
                        targets = [targets]
                    if not isinstance(targets, list):
                        issues.append(Issue("error", "training.lora.target_modules", "must be a list"))
                    else:
                        unknown = [t for t in targets if t not in LORA_TARGETS]
                        if unknown:
                            issues.append(Issue("warning", "training.lora.target_modules",
                                                f"unknown target(s): {', '.join(map(str, unknown))}",
                                                f"mizan knows: {', '.join(LORA_TARGETS)}"))

        if training.get("full_finetune") and not training.get("quantization"):
            issues.append(Issue("warning", "training", "full fine-tune with no quantization",
                                "expect optimizer states ~12 bytes/param; run `mizan weigh` first"))

    mizan = cfg.get("mizan", {})
    if isinstance(mizan, dict):
        for key in mizan:
            if key not in KNOWN_MIZAN_KEYS:
                issues.append(Issue("error", f"mizan.{key}", "unknown key"))
        gpu = _num(mizan.get("gpu_gib"))
        if gpu is not None and not (1 <= gpu <= 1024):
            issues.append(Issue("error", "mizan.gpu_gib", "looks out of range (1-1024 GiB)"))
        cal = _num(mizan.get("calibration"))
        if cal is not None and not (0.5 <= cal <= 3.0):
            issues.append(Issue("error", "mizan.calibration", "out of range (0.5-3.0)"))
    elif mizan is not None:
        issues.append(Issue("error", "mizan", "must be a mapping"))

    if strict:
        for issue in issues:
            if issue.level == "warning":
                issue.level = "error"
    return issues
