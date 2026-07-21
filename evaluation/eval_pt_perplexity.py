#!/usr/bin/env python3
"""按 PT 训练的 tokenization/packing 规则评估因果语言模型困惑度。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_FILENAMES = (
    "evaluation_config.json", "metrics.json", "block_metrics.jsonl", "run_summary.txt",
)
PACKING_BATCH_SIZE = 1000  # datasets.Dataset.map(batched=True) 的默认 batch_size。


class EvaluationError(ValueError):
    """评估配置、数据或模型输出不合法。"""


@dataclass
class DataStats:
    source_files: int = 0
    raw_records: int = 0
    valid_records: int = 0
    blank_lines: int = 0
    empty_text_records: int = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="公平评估 Base 或 Base+PT LoRA 在 PT validation 上的困惑度。",
        epilog="dry-run 只校验数据；增加 --dry_run_with_tokenizer 可用显式本地 tokenizer 检查 packing。",
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--tokenizer_name_or_path")
    parser.add_argument("--peft_path")
    parser.add_argument(
        "--validation_file_dir", type=Path,
        default=Path.home() / "datasets/medicalgpt/processed/pt/validation",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_blocks", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--torch_dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_with_tokenizer", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("block_size", "batch_size"):
        if getattr(args, name) <= 0:
            raise EvaluationError(f"--{name} 必须大于 0。")
    if args.max_blocks is not None and args.max_blocks <= 0:
        raise EvaluationError("--max_blocks 必须大于 0。")
    if args.num_workers < 0:
        raise EvaluationError("--num_workers 不能小于 0。")
    if args.dry_run_with_tokenizer and not args.dry_run:
        raise EvaluationError("--dry_run_with_tokenizer 只能与 --dry_run 一起使用。")


def load_pt_records(directory: Path) -> tuple[list[str], DataStats, list[Path]]:
    """递归、稳定地读取 PT JSONL；保留合法文本原样以匹配训练输入。"""
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PT validation 目录不存在：{root}")
    files = sorted(path for path in root.rglob("*.jsonl") if path.is_file())
    if not files:
        raise EvaluationError(f"PT validation 目录中没有 JSONL 文件：{root}")
    stats = DataStats(source_files=len(files))
    texts: list[str] = []
    for path in files:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    stats.blank_lines += 1
                    continue
                stats.raw_records += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(f"{path} 第 {line_number} 行不是合法 JSON：{exc}") from exc
                if not isinstance(record, dict):
                    raise EvaluationError(f"{path} 第 {line_number} 行必须是 JSON 对象。")
                if "text" not in record:
                    raise EvaluationError(f"{path} 第 {line_number} 行缺少必需字段 'text'。")
                text = record["text"]
                if not isinstance(text, str):
                    raise EvaluationError(f"{path} 第 {line_number} 行的 'text' 必须是字符串。")
                if not text.strip():
                    stats.empty_text_records += 1
                    continue
                texts.append(text)
                stats.valid_records += 1
    if not texts:
        raise EvaluationError("PT validation 没有可用于评估的非空 text 记录。")
    return texts, stats, files


def tokenize_and_pack(
    texts: Sequence[str], tokenizer: Any, block_size: int, *, packing_batch_size: int = PACKING_BATCH_SIZE,
) -> tuple[list[list[int]], dict[str, int | float]]:
    """复现 pretraining.py：默认编码、逐文档 EOS、map-batch 内拼接与尾部处理。"""
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise EvaluationError("tokenizer 缺少 eos_token_id，无法复现 PT packing。")
    blocks: list[list[int]] = []
    total_raw_tokens = 0
    packed_tokens = 0
    dropped = 0
    for start in range(0, len(texts), packing_batch_size):
        encoded = tokenizer(list(texts[start:start + packing_batch_size]))
        rows = encoded["input_ids"]
        joined: list[int] = []
        for ids in rows:
            values = list(ids)
            joined.extend(values)
            if not values or values[-1] != eos:
                joined.append(eos)
        total_raw_tokens += len(joined)
        # 与训练源码一致：达到一个 block 才向下取整；更短的整个 map-batch 会保留为短块。
        usable = (len(joined) // block_size) * block_size if len(joined) >= block_size else len(joined)
        dropped += len(joined) - usable
        for offset in range(0, usable, block_size):
            block = joined[offset:offset + block_size]
            if block:
                blocks.append(block)
                packed_tokens += len(block)
    return blocks, {
        "total_raw_tokens": total_raw_tokens,
        "packed_tokens": packed_tokens,
        "dropped_remainder_tokens": dropped,
        "dropped_remainder_ratio": dropped / total_raw_tokens if total_raw_tokens else 0.0,
    }


def apply_max_blocks(blocks: Sequence[list[int]], max_blocks: int | None) -> list[list[int]]:
    return list(blocks[:max_blocks] if max_blocks is not None else blocks)


def resolve_device(requested: str) -> str:
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        raise EvaluationError("指定了 --device cuda，但当前环境没有可用 CUDA。")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)


def resolve_torch_dtype(name: str) -> Any:
    import torch
    return name if name == "auto" else getattr(torch, name)


def _warn_adapter_base_mismatch(peft_path: str, model_name_or_path: str) -> None:
    config_path = Path(peft_path).expanduser() / "adapter_config.json"
    if not config_path.is_file():
        return
    try:
        configured = json.loads(config_path.read_text(encoding="utf-8")).get("base_model_name_or_path")
    except (OSError, json.JSONDecodeError):
        return
    if configured and Path(str(configured)).name != Path(model_name_or_path).name:
        warnings.warn(
            f"adapter 声明的基座 {configured!r} 与 --model_name_or_path {model_name_or_path!r} 不同，请确认兼容性。",
            UserWarning,
        )


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, str]:
    """只从基座路径加载完整模型，再可选叠加只读 adapter。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = resolve_device(args.device)
    common = {
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    tokenizer_source = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **common)
    if tokenizer.eos_token_id is None:
        raise EvaluationError("tokenizer 缺少 eos_token_id。")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **common,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
    )
    model = model.to(torch.device(device))
    if args.peft_path:
        _warn_adapter_base_mismatch(args.peft_path, args.model_name_or_path)
        model = PeftModel.from_pretrained(
            model, args.peft_path, is_trainable=False,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=args.local_files_only,
        )
    model.eval()
    return model, tokenizer, device


def safe_perplexity(mean_nll: float) -> float:
    try:
        return math.exp(mean_nll)
    except OverflowError:
        return math.inf


def score_blocks(
    model: Any, blocks: Sequence[list[int]], *, batch_size: int, pad_token_id: int, device: str,
    block_file: Any | None = None,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """手动对 shift 后的 logits 求 sum CE，再按有效目标 token 聚合。"""
    import torch
    import torch.nn.functional as functional
    total_nll = 0.0
    total_scored = 0
    rows: list[dict[str, float | int]] = []
    for start in range(0, len(blocks), batch_size):
        batch = blocks[start:start + batch_size]
        width = max(len(item) for item in batch)
        input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        for index, values in enumerate(batch):
            input_ids[index, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
            attention_mask[index, :len(values)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        for index, values in enumerate(batch):
            labels = input_ids[index, 1:].clone()
            valid = attention_mask[index, 1:].bool()
            labels[~valid] = -100
            scored = int(valid.sum().item())
            nll = float(functional.cross_entropy(
                logits[index, :-1, :].float(), labels, ignore_index=-100, reduction="sum"
            ).item()) if scored else 0.0
            mean_nll = nll / scored if scored else math.nan
            row = {
                "block_index": start + index, "block_tokens": len(values), "scored_tokens": scored,
                "negative_log_likelihood": nll, "mean_negative_log_likelihood": mean_nll,
                "perplexity": safe_perplexity(mean_nll) if scored else math.nan,
            }
            rows.append(row)
            if block_file is not None:
                json.dump(row, block_file, ensure_ascii=False, allow_nan=True)
                block_file.write("\n"); block_file.flush()
            total_nll += nll
            total_scored += scored
    if total_scored == 0:
        raise EvaluationError("total_scored_tokens 为 0，无法计算困惑度。")
    mean_nll = total_nll / total_scored
    return {
        "total_scored_tokens": total_scored,
        "total_negative_log_likelihood": total_nll,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": safe_perplexity(mean_nll),
    }, rows


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"输出目录非空：{output_dir}；请使用 --overwrite。")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    if overwrite:
        for path in paths.values():
            path.unlink(missing_ok=True)
    return paths


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=True)
            file.write("\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _redact_location(value: str | None) -> str | None:
    """清除 URL 用户信息、查询参数和片段，避免配置意外保存凭据。"""
    if value is None or "://" not in value:
        return value
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def build_config(args: argparse.Namespace, *, tokenizer: Any, model: Any, device: str, started_at: str) -> dict[str, Any]:
    import torch
    return {
        "git_commit": _git_commit(), "model_name_or_path": _redact_location(args.model_name_or_path),
        "tokenizer_name_or_path": _redact_location(args.tokenizer_name_or_path or args.model_name_or_path),
        "peft_path": _redact_location(args.peft_path),
        "validation_file_dir": str(args.validation_file_dir.expanduser().resolve()),
        "block_size": args.block_size, "batch_size": args.batch_size, "max_blocks": args.max_blocks,
        "device": device, "torch_dtype": args.torch_dtype, "seed": args.seed, "num_workers": args.num_workers,
        "python_version": platform.python_version(), "pytorch_version": torch.__version__,
        "transformers_version": _version("transformers"), "peft_version": _version("peft"),
        "tokenizer_class": type(tokenizer).__name__, "model_class": type(model).__name__,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "started_at": started_at, "ended_at": None,
    }


def run(args: argparse.Namespace, *, model: Any | None = None, tokenizer: Any | None = None) -> int:
    validate_args(args)
    random.seed(args.seed)
    texts, data_stats, files = load_pt_records(args.validation_file_dir)
    resolved = {
        **vars(args), "validation_file_dir": str(args.validation_file_dir.expanduser().resolve()),
        "output_dir": str(args.output_dir.expanduser().resolve()), "source_files": [str(path) for path in files],
        "raw_records": data_stats.raw_records, "valid_records": data_stats.valid_records,
        "blank_lines": data_stats.blank_lines, "empty_text_records": data_stats.empty_text_records,
    }
    if args.dry_run:
        print(json.dumps(resolved, ensure_ascii=False, indent=2, default=str))
        if args.dry_run_with_tokenizer:
            source = Path(args.tokenizer_name_or_path or args.model_name_or_path).expanduser()
            if not source.exists():
                raise EvaluationError("完整 packing dry-run 只允许显式本地 tokenizer 路径，拒绝联网解析 Hub ID。")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(source), cache_dir=str(args.cache_dir) if args.cache_dir else None,
                trust_remote_code=args.trust_remote_code, local_files_only=True,
            )
            blocks, packing = tokenize_and_pack(texts, tokenizer, args.block_size)
            print(json.dumps({**packing, "packed_blocks": len(blocks)}, ensure_ascii=False, indent=2))
        print("dry-run 完成：未加载模型，未创建正式 metrics 结果。")
        return 0

    paths = prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    if model is None or tokenizer is None:
        model, tokenizer, device = load_model_and_tokenizer(args)
    else:
        device = resolve_device(args.device)
        model.eval()
    if tokenizer.eos_token_id is None:
        raise EvaluationError("tokenizer 缺少 eos_token_id。")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    blocks, packing = tokenize_and_pack(texts, tokenizer, args.block_size)
    packed_blocks = len(blocks)
    selected = apply_max_blocks(blocks, args.max_blocks)
    if not selected:
        raise EvaluationError("packing 后没有可评估 block。")
    config = build_config(args, tokenizer=tokenizer, model=model, device=device, started_at=started_at)
    atomic_write_json(paths["evaluation_config.json"], config)
    try:
        with paths["block_metrics.jsonl"].open("w", encoding="utf-8", newline="\n") as block_file:
            score, _ = score_blocks(
                model, selected, batch_size=args.batch_size, pad_token_id=tokenizer.pad_token_id,
                device=device, block_file=block_file,
            )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError("CUDA 显存不足；请减小 --batch_size，并确认 --block_size 设置。") from exc
        raise
    elapsed = time.perf_counter() - started
    metrics = {
        "records": data_stats.valid_records, **asdict(data_stats), "block_size": args.block_size,
        "packed_blocks": packed_blocks, "evaluated_blocks": len(selected), **packing, **score,
        "batch_size": args.batch_size, "elapsed_seconds": elapsed,
        "blocks_per_second": len(selected) / elapsed if elapsed else math.inf,
        "scored_tokens_per_second": score["total_scored_tokens"] / elapsed if elapsed else math.inf,
    }
    atomic_write_json(paths["metrics.json"], metrics)
    config["ended_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(paths["evaluation_config.json"], config)
    summary = (
        "PT perplexity evaluation\n"
        f"records: {metrics['records']}\nevaluated_blocks: {metrics['evaluated_blocks']}\n"
        f"total_scored_tokens: {metrics['total_scored_tokens']}\n"
        f"mean_negative_log_likelihood: {metrics['mean_negative_log_likelihood']:.8f}\n"
        f"perplexity: {metrics['perplexity']:.8f}\n"
    )
    paths["run_summary.txt"].write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EvaluationError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
