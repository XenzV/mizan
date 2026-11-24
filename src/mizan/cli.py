
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import load_config, lint_config
from .estimator import LORA_TARGETS, WeighInput, param_counts, weigh, verdict
from .gpu import configured_gpu_gib, detect_gpus
from .models import PRESETS, resolve_model
from .render import C, badge, bar, gib, heading, table, tip

__all__ = ["main", "build_parser"]


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", "-m", help="preset name, alias, or local folder with config.json")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--num-hidden-layers", type=int)
    parser.add_argument("--num-attention-heads", type=int)
    parser.add_argument("--num-key-value-heads", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--head-dim", type=int)


def _add_weigh_args(parser: argparse.ArgumentParser) -> None:
    _add_model_args(parser)
    parser.add_argument("--config", "-c", help="JSON / TOML / YAML training config")
    parser.add_argument("--gpu", type=float, help="GPU memory in GiB (overrides config and nvidia-smi)")
    parser.add_argument("--free", action="store_true", help="use free VRAM instead of total when auto-detecting")
    parser.add_argument("--seq", "--seq-len", dest="seq_len", type=int)
    parser.add_argument("--batch", "--batch-size", dest="batch_size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--quant", choices=["4bit", "8bit", "bf16", "fp16", "fp32", "none"])
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--optimizer")
    parser.add_argument("--full-finetune", action="store_true", default=None)
    parser.add_argument("--lora", dest="full_finetune", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--no-flash-attention", dest="flash_attention", action="store_false", default=None)
    parser.add_argument("--no-loss-upcast", dest="loss_upcast_fp32", action="store_false", default=None)
    parser.add_argument("--cuda-overhead-mb", type=int)
    parser.add_argument("--fragmentation", type=float)
    parser.add_argument("--calibration", type=float, help="scale estimate from a measured run (0.5-3.0)")
    parser.add_argument("--gpu-index", type=int, default=0, help="which GPU to use when auto-detecting")
    parser.add_argument("--headroom", type=float, default=0.05, help="safety margin (default: 0.05)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mizan",
        description="Weigh your fine-tune before you run it. No PyTorch, no CUDA, no downloads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  mizan weigh --model llama-3.1-8b --gpu 4 --seq 512\n"
            "  mizan weigh --config soup.yaml --gpu 8\n"
            "  mizan fit --model ./models/Qwen2.5-7B --gpu 8 --seq 2048\n"
            "  mizan check --config soup.yaml --strict\n"
            "  mizan compare a.yaml b.yaml --gpu 8\n"
            "  mizan sweep -m llama-3.1-8b --gpu 8 --seqs 512,1024,2048\n"
            "  mizan calibrate -m llama-3.1-8b --measured 6.1 --seq 512\n"
            "  mizan doctor\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"mizan {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("weigh", help="estimate peak VRAM and print a verdict")
    _add_weigh_args(p)
    p.set_defaults(func=cmd_weigh)

    p = sub.add_parser("fit", help="suggest the largest config that fits your GPU")
    _add_model_args(p)
    p.add_argument("--config", "-c")
    p.add_argument("--gpu", type=float)
    p.add_argument("--free", action="store_true")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--seq", "--seq-len", dest="seq_len", type=int)
    p.add_argument("--calibration", type=float)
    p.add_argument("--target-util", type=float, default=0.90,
                   help="fraction of VRAM to stay under (default: 0.90)")
    p.add_argument("--effective-batch", type=int, default=16, help="target effective batch size")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("check", help="lint a training config (CI-friendly exit codes)")
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--strict", action="store_true", help="treat warnings as errors")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("presets", help="list built-in model architectures")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("doctor", help="show detected GPU and environment facts")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("init", help="print a starter config")
    p.add_argument("--model", "-m", default="llama-3.1-8b")
    p.add_argument("--gpu", type=float, default=8.0)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--format", choices=["json", "yaml"], default="yaml")
    p.add_argument("--out", help="write to file instead of stdout")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("report", help="write a Markdown preflight report")
    _add_weigh_args(p)
    p.add_argument("--out", default="mizan-report.md")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("compare", help="compare two configs side by side")
    p.add_argument("configs", nargs=2, help="two JSON / TOML / YAML configs")
    p.add_argument("--gpu", type=float)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("calibrate", help="derive a calibration factor from a measured peak")
    _add_model_args(p)
    p.add_argument("--config", "-c")
    p.add_argument("--measured", type=float, required=True, help="measured peak in GiB")
    p.add_argument("--seq", "--seq-len", dest="seq_len", type=int)
    p.add_argument("--batch", "--batch-size", dest="batch_size", type=int)
    p.add_argument("--quant")
    p.add_argument("--lora-rank", type=int)
    p.add_argument("--gradient-checkpointing", action="store_true", default=None)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("sweep", help="sweep sequence lengths and batch sizes")
    _add_model_args(p)
    p.add_argument("--config", "-c")
    p.add_argument("--gpu", type=float)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--seqs", default="512,1024,2048", help="comma-separated sequence lengths")
    p.add_argument("--batches", default="1,2,4", help="comma-separated micro-batch sizes")
    p.add_argument("--quant", default="4bit")
    p.add_argument("--csv", help="write the table to a CSV file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_sweep)

    return parser


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else default


def _load_optional(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return load_config(path)
    except (OSError, ValueError) as exc:
        print(f"{C.RED}error:{C.RESET} {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _spec_from(args: argparse.Namespace, cfg: dict[str, Any]):
    mizan = _get(cfg, "mizan", {})
    model = getattr(args, "model", None) or _get(cfg, "base") or _get(cfg, "model")
    kwargs = dict(
        hidden_size=getattr(args, "hidden_size", None) or _get(mizan, "hidden_size"),
        intermediate_size=getattr(args, "intermediate_size", None) or _get(mizan, "intermediate_size"),
        num_hidden_layers=getattr(args, "num_hidden_layers", None) or _get(mizan, "num_hidden_layers"),
        num_attention_heads=getattr(args, "num_attention_heads", None) or _get(mizan, "num_attention_heads"),
        vocab_size=getattr(args, "vocab_size", None) or _get(mizan, "vocab_size"),
        head_dim=getattr(args, "head_dim", None) or _get(mizan, "head_dim"),
        num_key_value_heads=getattr(args, "num_key_value_heads", None) or _get(mizan, "num_key_value_heads"),
    )
    try:
        return resolve_model(model, **kwargs)
    except ValueError as exc:
        print(f"{C.RED}error:{C.RESET} {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _gpu_from(args: argparse.Namespace, cfg: dict[str, Any]) -> float | None:
    explicit = getattr(args, "gpu", None)
    if explicit is not None:
        return explicit if explicit > 0 else None
    declared = _get(_get(cfg, "mizan", {}), "gpu_gib")
    if declared is not None:
        try:
            return float(declared)
        except (TypeError, ValueError):
            pass
    env = configured_gpu_gib()
    if env is not None:
        return env
    gpus = detect_gpus()
    if gpus:
        index = getattr(args, "gpu_index", 0) or 0
        chosen = gpus[index] if index < len(gpus) else gpus[0]
        return (chosen.free_mib if getattr(args, "free", False) else chosen.total_mib) / 1024
    return None


def _weigh_input(args: argparse.Namespace, cfg: dict[str, Any], spec, **overrides) -> WeighInput:
    training = _get(cfg, "training", {})
    mizan = _get(cfg, "mizan", {})
    lora = _get(training, "lora", {})

    def pick(flag: str, *keys: tuple[dict, str], default: Any) -> Any:
        value = getattr(args, flag, None)
        if value is not None:
            return value
        for source, key in keys:
            found = _get(source, key)
            if found is not None:
                return found
        return default

    def positive_int(value: Any, fallback: int) -> int:
        try:
            n = int(value)
            return n if n > 0 else fallback
        except (TypeError, ValueError):
            return fallback

    batch_raw = pick("batch_size", (training, "micro_batch_size"), (training, "batch_size"), default=1)
    seq_raw = pick("seq_len", (training, "seq_len"), (training, "max_length"), (mizan, "seq_len"), default=1024)
    quant = pick("quant", (training, "quantization"), (mizan, "quant"), default="4bit")
    optimizer = pick("optimizer", (training, "optimizer"), (mizan, "optimizer"), default="adamw")
    full = pick("full_finetune", (training, "full_finetune"), default=False)
    ckpt = pick("gradient_checkpointing", (training, "gradient_checkpointing"), default=True)
    flash = pick("flash_attention", (training, "flash_attention"), (mizan, "flash_attention"), default=True)
    upcast = pick("loss_upcast_fp32", (mizan, "loss_upcast_fp32"), default=True)
    cal_raw = pick("calibration", (mizan, "calibration"), default=1.0)
    try:
        cal = float(cal_raw)
    except (TypeError, ValueError):
        print(f"{C.RED}error:{C.RESET} calibration must be a number", file=sys.stderr)
        raise SystemExit(2) from None

    values = dict(
        spec=spec,
        seq_len=positive_int(seq_raw, 1024),
        batch_size=positive_int(batch_raw, 1),
        grad_accum=positive_int(
            pick("grad_accum", (training, "gradient_accumulation"), (training, "grad_accum"), default=1), 1,
        ),
        quant=quant,
        lora_rank=positive_int(
            pick("lora_rank", (lora, "r"), default=16), 16,
        ),
        full_finetune=bool(full),
        gradient_checkpointing=bool(ckpt),
        flash_attention=bool(flash),
        loss_upcast_fp32=bool(upcast),
        optimizer=optimizer,
        calibration=cal,
    )
    for key in ("cuda_overhead_mb", "fragmentation"):
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    values.update(overrides)
    try:
        return WeighInput(**values)
    except ValueError as exc:
        print(f"{C.RED}error:{C.RESET} {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _model_label(spec) -> str:
    return f"{spec.name} {C.DIM}({spec.source}, {param_counts(spec)['total'] / 1e9:.2f}B params){C.RESET}"


def cmd_weigh(args: argparse.Namespace) -> None:
    cfg = _load_optional(getattr(args, "config", None))
    spec = _spec_from(args, cfg)
    weight = _weigh_input(args, cfg, spec)
    result = weigh(weight)
    gpu = _gpu_from(args, cfg)
    status, explanation = verdict(result, gpu, args.headroom)

    if args.json:
        payload = result.as_dict()
        payload.update({"gpu_gib": gpu, "verdict": status, "explanation": explanation})
        print(json.dumps(payload, indent=2))
        return

    print()
    print(f"{heading('MIZAN / WEIGH')}")
    print(f"  Model   {_model_label(spec)}")
    print(f"  Config  {weight.batch_size}x{weight.seq_len} tok  {C.DIM}|{C.RESET} "
          f"{weight.quant}  {C.DIM}|{C.RESET} "
          f"{'full fine-tune' if weight.full_finetune else f'LoRA r{weight.lora_rank}'}  "
          f"{C.DIM}|{C.RESET} ckpt {'on' if weight.gradient_checkpointing else 'off'}"
          + (f"{C.DIM}|{C.RESET} cal {weight.calibration:g}" if weight.calibration != 1.0 else ""))
    print()
    limit_mib = (gpu or 0) * 1024 * (1 - args.headroom)
    for key, value in result.components.items():
        share = value / result.total_mib if result.total_mib else 0
        suffix = f"  {bar(share, 18)}" if key != "fragmentation" else ""
        print(f"  {key:20} {gib(value)}{suffix}")
    print(f"  {C.DIM}{'-' * 62}{C.RESET}")
    print(f"  {C.BOLD}{'ESTIMATED PEAK':20} {gib(result.total_mib)}{C.RESET}")
    print()
    gpu_label = f"{gpu:.1f} GiB GPU" if gpu else "no GPU specified"
    print(f"  {C.BOLD}VERDICT{C.RESET}  {badge(status)}  {tip(f'({gpu_label})')}")
    print(f"         {explanation}")
    print(f"  {tip('Analytical estimate, expect +/-20%. Verify with a 20-step run before a long job.')}")
    print()
    if limit_mib and status == "OVER":
        print(f"  {C.YELLOW}Try:{C.RESET} mizan fit --model {spec.name} --gpu {gpu:g} --seq {weight.seq_len}")
        print()


def _fit_candidates(spec, *, gpu: float, seq_len: int, target_util: float,
                    effective_batch: int, optimizer: str,
                    calibration: float = 1.0) -> list[dict[str, Any]]:
    combos = [
        ("4bit", False), ("4bit", True),
        ("8bit", False), ("8bit", True),
        ("bf16", True), ("fp32", True),
    ]
    speed = {"4bit": 1.0, "8bit": 0.92, "bf16": 0.85, "fp32": 0.7}
    out: list[dict[str, Any]] = []
    for quant, ckpt in combos:
        for batch in (16, 8, 4, 2, 1):
            cfg = WeighInput(
                spec=spec, seq_len=seq_len, batch_size=batch,
                quant=quant, gradient_checkpointing=ckpt, optimizer=optimizer,
                calibration=calibration,
            )
            res = weigh(cfg)
            need = res.total_mib / 1024
            if need <= gpu * target_util:
                accum = max(1, effective_batch // batch)
                score = batch * speed[quant] * (0.78 if ckpt else 1.0)
                out.append({
                    "batch": batch, "quant": quant, "checkpointing": ckpt,
                    "grad_accum": accum, "need_gib": round(need, 3),
                    "headroom_gib": round(gpu - need, 3), "score": round(score, 3),
                })
                break
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def cmd_fit(args: argparse.Namespace) -> None:
    cfg = _load_optional(getattr(args, "config", None))
    spec = _spec_from(args, cfg)
    gpu = _gpu_from(args, cfg)
    if not gpu:
        print(f"{C.RED}error:{C.RESET} no GPU size known. Pass --gpu 8, set MIZAN_GPU_GIB=8, "
              "or run this on a machine with nvidia-smi.", file=sys.stderr)
        raise SystemExit(2)
    baseline = _weigh_input(args, cfg, spec)

    seq = getattr(args, "seq_len", None) or baseline.seq_len
    target = args.target_util
    if not 0 < target <= 1:
        print(f"{C.RED}error:{C.RESET} --target-util must be in (0, 1]", file=sys.stderr)
        raise SystemExit(2)

    candidates = _fit_candidates(
        spec, gpu=gpu, seq_len=seq, target_util=target,
        effective_batch=args.effective_batch, optimizer=baseline.optimizer,
        calibration=baseline.calibration,
    )
    if not candidates:
        print(f"{C.RED}No configuration fits {gpu:.1f} GiB at {seq} tokens "
              f"(target {target:.0%} utilization).{C.RESET}")
        print("  Try a shorter --seq, a smaller model, or offload/CPU training.")
        raise SystemExit(1)

    best = candidates[0]
    probe = WeighInput(
        spec=spec, seq_len=seq, batch_size=best["batch"], quant=best["quant"],
        gradient_checkpointing=best["checkpointing"], optimizer=baseline.optimizer,
        calibration=baseline.calibration,
    )
    result = weigh(probe)

    if args.json:
        print(json.dumps({"gpu_gib": gpu, "candidates": candidates, "best": best}, indent=2))
        return

    print()
    print(f"{heading('MIZAN / FIT')}  {spec.name} on {gpu:.1f} GiB, {seq} tokens")
    print()
    print(table(
        [[c["quant"], str(c["batch"]), "on" if c["checkpointing"] else "off",
          str(c["grad_accum"]), f"{c['need_gib']:.2f}", f"{c['headroom_gib']:.2f}"]
         for c in candidates[:6]],
        ["quant", "micro-batch", "ckpt", "grad-accum", "need GiB", "free GiB"],
    ))
    print()
    print(f"  {C.GREEN}Recommended (throughput heuristic):{C.RESET} "
          f"{best['quant']}, micro-batch {best['batch']}, "
          f"grad-accum {best['grad_accum']}, ckpt {'on' if best['checkpointing'] else 'off'}")
    print(f"  Estimated peak {result.total_mib / 1024:.2f} GiB — leave the rest for the runtime.")
    print()
    snippet = {
        "base": spec.name,
        "task": "sft",
        "training": {
            "batch_size": best["batch"],
            "gradient_accumulation": best["grad_accum"],
            "seq_len": seq,
            "quantization": best["quant"],
            "gradient_checkpointing": best["checkpointing"],
            "lora": {"r": 16},
        },
        "mizan": {"gpu_gib": gpu},
    }
    print(f"{C.DIM}paste into soup.yaml / mizan.json:{C.RESET}")
    print(json.dumps(snippet, indent=2))
    print()


def cmd_check(args: argparse.Namespace) -> None:
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"{C.RED}error:{C.RESET} {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    issues = lint_config(cfg, strict=args.strict)
    if args.json:
        print(json.dumps([i.__dict__ for i in issues], indent=2))
    else:
        print()
        print(heading("MIZAN / CHECK"))
        for issue in issues:
            color = {"error": C.RED, "warning": C.YELLOW, "info": C.BLUE}[issue.level]
            print(f"  {color}{issue.line()}{C.RESET}")
        errors = sum(i.level == "error" for i in issues)
        print(f"  {C.DIM}{'-' * 62}{C.RESET}")
        print(f"  {errors} error(s), {sum(i.level != 'error' for i in issues)} other note(s)")
        print()
    if any(i.level == "error" for i in issues):
        raise SystemExit(1)


def cmd_presets(args: argparse.Namespace) -> None:
    rows = []
    for name, spec in PRESETS.items():
        rows.append([name, f"{param_counts(spec)['total'] / 1e9:.2f}B",
                     str(spec.num_hidden_layers), str(spec.hidden_size),
                     str(spec.intermediate_size), str(spec.vocab_size)])
    if args.json:
        print(json.dumps([{"name": r[0], "params": r[1], "layers": r[2],
                           "hidden": r[3], "intermediate": r[4], "vocab": r[5]} for r in rows], indent=2))
        return
    print()
    print(heading("MIZAN / PRESETS"))
    print()
    print(table(rows, ["preset", "params", "layers", "hidden", "ffn", "vocab"]))
    print()
    print(tip("Approximate counts from architecture. Use a local model folder for exact config.json values."))
    print()


def cmd_doctor(args: argparse.Namespace) -> None:
    gpus = detect_gpus()
    env_gpu = configured_gpu_gib()
    payload = {
        "mizan_version": __version__,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "nvidia_smi": bool(gpus),
        "gpus": [g.__dict__ for g in gpus],
        "MIZAN_GPU_GIB": env_gpu,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print()
    print(heading("MIZAN / DOCTOR"))
    print(f"  mizan     {__version__}")
    print(f"  python    {sys.version.split()[0]} ({sys.platform})")
    if gpus:
        for g in gpus:
            print(f"  gpu{g.index}     {g.name} — {g.total_gib} GiB total, "
                  f"{g.free_mib / 1024:.1f} GiB free, driver {g.driver}")
    else:
        print(f"  gpu       {C.YELLOW}not visible{C.RESET} (no nvidia-smi on PATH or no driver)")
    if env_gpu:
        print(f"  override  MIZAN_GPU_GIB={env_gpu}")
    print()
    print(tip("No GPU here? Use --gpu 4 to model an RTX 3050 Laptop, or set MIZAN_GPU_GIB."))
    print()


def _dump_yaml(config: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit(obj: Any, indent: int = 0) -> None:
        pad = " " * indent
        for key, value in obj.items():
            if isinstance(value, dict):
                lines.append(f"{pad}{key}:")
                emit(value, indent + 2)
            elif isinstance(value, list):
                lines.append(f"{pad}{key}: [{', '.join(map(str, value))}]")
            elif isinstance(value, bool):
                lines.append(f"{pad}{key}: {'true' if value else 'false'}")
            else:
                lines.append(f"{pad}{key}: {value}")

    emit(config)
    return "\n".join(lines) + "\n"


def cmd_init(args: argparse.Namespace) -> None:
    config = {
        "base": args.model,
        "task": "sft",
        "data": {"train": "./data/train.jsonl", "format": "alpaca"},
        "training": {
            "epochs": 3,
            "lr": 2e-5,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "seq_len": args.seq,
            "quantization": "4bit",
            "gradient_checkpointing": True,
            "lora": {"r": 16, "target_modules": list(LORA_TARGETS)},
        },
        "output": "./output",
        "mizan": {"gpu_gib": args.gpu},
    }
    text = _dump_yaml(config) if args.format == "yaml" else json.dumps(config, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")


def cmd_report(args: argparse.Namespace) -> None:
    cfg = _load_optional(getattr(args, "config", None))
    spec = _spec_from(args, cfg)
    weight = _weigh_input(args, cfg, spec)
    result = weigh(weight)
    gpu = _gpu_from(args, cfg)
    status, explanation = verdict(result, gpu, args.headroom)
    issues = lint_config(cfg) if cfg else []
    candidates = (
        _fit_candidates(spec, gpu=gpu, seq_len=weight.seq_len, target_util=0.9,
                        effective_batch=16, optimizer=weight.optimizer,
                        calibration=weight.calibration)
        if gpu else []
    )

    lines = [
        f"# Mizan preflight — {spec.name}",
        "",
        f"Generated by `mizan {__version__}` · model source: `{spec.source}`",
        "",
        "## Verdict",
        "",
        f"**{status}** — {explanation}." if gpu else f"**{status}** — {explanation}.",
        "",
        f"Estimated peak: **{result.total_mib / 1024:.2f} GiB** "
        f"(analytical estimate, expect ±20%).",
        "",
        "## Breakdown",
        "",
        "| Component | GiB | Share |",
        "| --- | ---: | ---: |",
    ]
    for key, value in result.components.items():
        lines.append(f"| {key} | {value / 1024:.3f} | {value / result.total_mib:.1%} |")
    lines += [
        f"| **total** | **{result.total_mib / 1024:.3f}** | 100% |",
        "",
        "## Config",
        "",
        f"- micro-batch: {weight.batch_size}",
        f"- sequence length: {weight.seq_len}",
        f"- quantization: {weight.quant}",
        f"- optimizer: {weight.optimizer}",
        f"- gradient checkpointing: {weight.gradient_checkpointing}",
        f"- LoRA rank: {weight.lora_rank}" if not weight.full_finetune else "- full fine-tune",
        "",
    ]
    if issues:
        lines += ["## Config lint", ""]
        lines += [f"- {i.line()}" for i in issues] or ["- no issues"]
        lines.append("")
    if candidates:
        best = candidates[0]
        lines += [
            "## Suggested fit",
            "",
            f"{best['quant']}, micro-batch {best['batch']}, grad-accum {best['grad_accum']}, "
            f"checkpointing {'on' if best['checkpointing'] else 'off'} "
            f"({best['need_gib']:.2f} GiB estimated).",
            "",
        ]
    lines += [
        "## How to read this",
        "",
        "Mizan computes weights + LoRA states + activations + CUDA context + fragmentation",
        "from architecture integers. It never loads model weights or talks to a server.",
        "Confirm with a 20-step run before committing to a long training job.",
        "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} ({status})")


def _dummy_args(**changes: Any) -> argparse.Namespace:
    defaults = dict(
        model=None, hidden_size=None, intermediate_size=None, num_hidden_layers=None,
        num_attention_heads=None, num_key_value_heads=None, vocab_size=None, head_dim=None,
        gpu=None, free=False, gpu_index=0, seq_len=None, batch_size=None, grad_accum=None,
        quant=None, lora_rank=None, optimizer=None, full_finetune=None,
        gradient_checkpointing=None, flash_attention=None, loss_upcast_fp32=None,
        cuda_overhead_mb=None, fragmentation=None, calibration=None,
        headroom=0.05, json=False,
    )
    defaults.update(changes)
    return argparse.Namespace(**defaults)


def _parse_ints(text: str, flag: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            number = int(part)
        except ValueError:
            print(f"{C.RED}error:{C.RESET} {flag} expects integers, got {part!r}", file=sys.stderr)
            raise SystemExit(2) from None
        if number <= 0:
            print(f"{C.RED}error:{C.RESET} {flag} values must be positive", file=sys.stderr)
            raise SystemExit(2)
        values.append(number)
    if not values:
        print(f"{C.RED}error:{C.RESET} {flag} needs at least one value", file=sys.stderr)
        raise SystemExit(2)
    return values


def cmd_compare(args: argparse.Namespace) -> None:
    proxy = _dummy_args(gpu_index=args.gpu_index)
    gpu = args.gpu if args.gpu is not None else _gpu_from(proxy, {})
    summaries = []
    for path in args.configs:
        cfg = _load_optional(path)
        spec = _spec_from(proxy, cfg)
        w = _weigh_input(proxy, cfg, spec)
        r = weigh(w)
        status, note = verdict(r, gpu, 0.05)
        summaries.append({
            "config": path, "model": spec.name, "total_gib": round(r.total_mib / 1024, 3),
            "verdict": status, "note": note, "quant": w.quant,
            "batch": w.batch_size, "seq": w.seq_len,
            "components_gib": {k: round(v / 1024, 3) for k, v in r.components.items()},
        })
    if args.json:
        print(json.dumps(summaries, indent=2))
        return
    print()
    print(f"{heading('MIZAN / COMPARE')}  gpu={gpu if gpu else 'unknown'}")
    print()
    print(table(
        [[s["config"], s["model"], s["quant"], str(s["batch"]), str(s["seq"]),
          f"{s['total_gib']:.2f}", s["verdict"]] for s in summaries],
        ["config", "model", "quant", "batch", "seq", "GiB", "verdict"],
    ))
    print()
    if len(summaries) == 2 and gpu:
        delta = summaries[1]["total_gib"] - summaries[0]["total_gib"]
        word = "more" if delta > 0 else "less"
        print(f"  Second config needs {abs(delta):.2f} GiB {word} than the first.")
    print()


def cmd_calibrate(args: argparse.Namespace) -> None:
    if args.measured <= 0:
        print(f"{C.RED}error:{C.RESET} --measured must be positive", file=sys.stderr)
        raise SystemExit(2)
    cfg = _load_optional(getattr(args, "config", None))
    proxy = _dummy_args(
        model=args.model, seq_len=args.seq_len, batch_size=args.batch_size,
        quant=args.quant, lora_rank=args.lora_rank,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    spec = _spec_from(proxy, cfg)
    w = _weigh_input(proxy, cfg, spec)
    estimated = weigh(w).total_mib / 1024
    factor = round(args.measured / estimated, 3)
    clamped = round(max(0.5, min(3.0, factor)), 3)
    if args.json:
        print(json.dumps({
            "model": spec.name, "estimated_gib": round(estimated, 3),
            "measured_gib": args.measured, "factor": factor, "usable_factor": clamped,
        }, indent=2))
        return
    print()
    print(f"{heading('MIZAN / CALIBRATE')}  {spec.name}")
    print(f"  estimated  {estimated:.2f} GiB")
    print(f"  measured   {args.measured:.2f} GiB")
    print(f"  factor     {factor:g}")
    if factor != clamped:
        print(f"  {C.YELLOW}Clamped to {clamped:g} (allowed 0.5-3.0){C.RESET}")
    print()
    print(f"{C.DIM}add to your config:{C.RESET}")
    print(f"mizan:\n  calibration: {clamped}")
    print()
    print(tip("Factor scales weights, LoRA states and activations. CUDA context stays fixed."))
    print()


def cmd_sweep(args: argparse.Namespace) -> None:
    cfg = _load_optional(getattr(args, "config", None))
    gpu = args.gpu if args.gpu is not None else _gpu_from(args, cfg)
    if not gpu:
        print(f"{C.RED}error:{C.RESET} no GPU size known. Pass --gpu or set MIZAN_GPU_GIB.", file=sys.stderr)
        raise SystemExit(2)
    seqs = _parse_ints(args.seqs, "--seqs")
    batches = _parse_ints(args.batches, "--batches")
    spec = _spec_from(args, cfg)
    base = _weigh_input(args, cfg, spec)
    quant = args.quant or base.quant
    rows = []
    for seq in seqs:
        for batch in batches:
            w = WeighInput(
                spec=spec, seq_len=seq, batch_size=batch, quant=quant,
                gradient_checkpointing=base.gradient_checkpointing,
                optimizer=base.optimizer, calibration=base.calibration,
            )
            r = weigh(w)
            status, _ = verdict(r, gpu, 0.05)
            rows.append({"seq": seq, "batch": batch, "gib": round(r.total_mib / 1024, 3),
                         "verdict": status})
    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["seq", "batch", "gib", "verdict"])
            writer.writeheader()
            writer.writerows(rows)
        if not args.json:
            print(f"wrote {args.csv}")
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print()
    print(f"{heading('MIZAN / SWEEP')}  {spec.name} on {gpu:.1f} GiB, {quant}")
    print()
    print(table([[str(r["seq"]), str(r["batch"]), f"{r['gib']:.2f}", r["verdict"]]
                 for r in rows], ["seq", "batch", "GiB", "verdict"]))
    print()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0
