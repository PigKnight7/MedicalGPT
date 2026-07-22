#!/usr/bin/env python3
"""可靠地评估 Base 或 Base+PEFT 在 CMExam official 划分上的 exact-set 表现。"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medicalgpt_ext.cmexam_utils import (
    exact_set_match,
    extract_predicted_labels,
    format_answer_labels,
    format_cmexam_prompt,
    normalize_answer_labels,
    validate_options,
)

LOGGER = logging.getLogger("eval_cmexam_v2")
OUTPUT_FILES = (
    "evaluation_config.json", "adapter_verification.json", "predictions.jsonl",
    "metrics.json", "error_cases.jsonl", "run_summary.txt",
)


class EvaluationError(ValueError):
    """评估输入、数据、模型状态或输出不满足协议。"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMExam v2：Base/PEFT 统一 exact-set 评估。")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--peft_path", type=Path)
    parser.add_argument("--data_root", type=Path, default=Path.home() / "datasets/medicalgpt/processed/cmexam/official")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow_test_evaluation", action="store_true")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verify_adapter_effect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter_check_samples", type=int, default=3)
    parser.add_argument("--adapter_check_atol", type=float, default=0.0)
    parser.add_argument("--skip_adapter_effect_check", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "max_input_length", "max_new_tokens", "adapter_check_samples"):
        if getattr(args, name) <= 0:
            raise EvaluationError(f"--{name} 必须大于 0。")
    if args.max_samples is not None and args.max_samples <= 0:
        raise EvaluationError("--max_samples 必须大于 0。")
    if not math.isfinite(args.adapter_check_atol) or args.adapter_check_atol < 0:
        raise EvaluationError("--adapter_check_atol 必须是有限非负数。")
    if args.split == "test" and not args.allow_test_evaluation:
        raise EvaluationError("CMExam test 是受保护的最终评估集；必须显式传入 --allow_test_evaluation。")


def resolve_data_file(data_root: Path, split: str) -> Path:
    root = data_root.expanduser().resolve()
    lowered = tuple(part.lower() for part in root.parts)
    if "decontaminated" in lowered or "train" in lowered or not lowered or lowered[-1] != "official":
        raise EvaluationError("v2 只允许 CMExam official/validation 或 official/test，禁止 train 数据。")
    return root / split / f"{split}.jsonl"


def load_cmexam_records(path: Path, split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"CMExam {split} 文件不存在：{path}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path.name} 第 {line_no} 行 JSON 非法：{exc}") from exc
            if not isinstance(raw, dict):
                raise EvaluationError(f"{path.name} 第 {line_no} 行必须是对象。")
            rid = raw.get("id")
            if not isinstance(rid, str) or not rid.strip():
                raise EvaluationError(f"{path.name} 第 {line_no} 行 id 必须是非空字符串。")
            if rid in seen:
                raise EvaluationError(f"{path.name} 第 {line_no} 行 id 重复：{rid}")
            question = raw.get("question")
            if not isinstance(question, str) or not question.strip():
                raise EvaluationError(f"记录 {rid} 缺少非空 question。")
            try:
                options = validate_options(raw.get("options"))
            except ValueError as exc:
                raise EvaluationError(f"记录 {rid} 的 options 非法：{exc}") from exc
            labels = normalize_answer_labels(raw.get("answer_labels"))
            if not labels:
                raise EvaluationError(f"记录 {rid} 的 answer_labels 非法。")
            option_labels = {item["label"] for item in options}
            if not set(labels) <= option_labels:
                raise EvaluationError(f"记录 {rid} 的答案标签不在 options 中。")
            if raw.get("split") not in (None, split):
                raise EvaluationError(f"记录 {rid} 的 split={raw.get('split')!r} 与 {split} 不一致。")
            item = dict(raw)
            item.update(id=rid, split=split, question=question.strip(), options=[dict(x) for x in options],
                        answer_labels=list(labels), is_multiple_choice=len(labels) > 1)
            records.append(item)
            seen.add(rid)
    if not records:
        raise EvaluationError(f"CMExam 文件为空：{path}")
    return records


def select_records(records: Sequence[dict[str, Any]], max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    if max_samples is None or max_samples >= len(records):
        return list(records)
    indices = sorted(random.Random(seed).sample(range(len(records)), max_samples))
    return [records[index] for index in indices]


class _ThinkingDisabledTokenizer:
    """为公共 Prompt 函数补充 Qwen chat template 的硬性 thinking 参数。"""
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template

    def apply_chat_template(self, conversation: Any, *, tokenize: bool, add_generation_prompt: bool) -> str:
        return self._tokenizer.apply_chat_template(
            conversation, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )


def build_prompt(tokenizer: Any, question: str) -> str:
    return format_cmexam_prompt(_ThinkingDisabledTokenizer(tokenizer), question, prompt_style="answer_only")


def resolve_torch_dtype(name: str) -> Any:
    import torch
    return {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir, trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only, use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise EvaluationError("tokenizer 缺少 eos_token/eos_token_id。")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise EvaluationError("tokenizer 缺少 chat_template。")
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(args: argparse.Namespace) -> Any:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=resolve_torch_dtype(args.torch_dtype), cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    return model


def _active_adapters(model: Any) -> list[str]:
    value = getattr(model, "active_adapters", getattr(model, "active_adapter", []))
    if callable(value):
        value = value()
    if isinstance(value, str):
        return [value]
    return list(value or [])


def validate_adapter_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir() or not (path / "adapter_config.json").is_file():
        raise EvaluationError(f"PEFT adapter 目录或 adapter_config.json 不存在：{path}")
    if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise EvaluationError(f"PEFT adapter 缺少 adapter_model.safetensors/adapter_model.bin：{path}")
    return path


def load_model_with_optional_adapter(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    base = load_base_model(args)
    info = {"base_model_class": type(base).__name__, "adapter_name": None, "active_adapters": []}
    if args.peft_path is None:
        info["final_model_class"] = type(base).__name__
        return base, info
    from peft import PeftModel
    path = validate_adapter_path(args.peft_path)
    model = PeftModel.from_pretrained(
        base, str(path), adapter_name="default", is_trainable=False,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only,
    )
    if not isinstance(model, PeftModel):
        raise EvaluationError("PeftModel.from_pretrained 未返回 PeftModel，拒绝退回 Base。")
    model.set_adapter("default", inference_mode=True)
    if "default" not in getattr(model, "peft_config", {}):
        raise EvaluationError("加载后的 PeftModel 不包含 default adapter。")
    if "default" not in _active_adapters(model):
        raise EvaluationError("default adapter 未处于活动状态。")
    model.eval()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable:
        raise EvaluationError(f"推理 adapter 仍有 {trainable} 个可训练参数，拒绝评估。")
    info.update(adapter_name="default", active_adapters=_active_adapters(model), final_model_class=type(model).__name__,
                trainable_parameters=trainable, total_parameters=total)
    return model, info


def _to_device(batch: Any, device: str) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def verify_adapter_changes_logits(model: Any, tokenizer: Any, records: Sequence[dict[str, Any]],
                                  *, device: str, sample_count: int, atol: float,
                                  adapter_path: str) -> dict[str, Any]:
    import torch
    details: list[dict[str, Any]] = []
    try:
        for record in records[:sample_count]:
            encoded = _to_device(tokenizer(build_prompt(tokenizer, record["question"]), return_tensors="pt"), device)
            with torch.inference_mode():
                with model.disable_adapter():
                    disabled = model(**encoded).logits[:, -1, :].float().cpu()
                model.set_adapter("default", inference_mode=True)
                enabled = model(**encoded).logits[:, -1, :].float().cpu()
            diff = (enabled - disabled).abs()
            details.append({"id": record["id"], "max_abs_diff": diff.max().item(),
                            "mean_abs_diff": diff.mean().item(), "nonzero_count": int((diff > atol).sum().item()),
                            "top1_token_changed": bool(enabled.argmax().item() != disabled.argmax().item())})
    finally:
        model.set_adapter("default", inference_mode=True)
    if "default" not in _active_adapters(model):
        raise EvaluationError("adapter logits 检查后 default adapter 未恢复。")
    passed = any(item["max_abs_diff"] > atol for item in details)
    result = {"requested": True, "passed": passed, "skipped": False, "adapter_path": adapter_path,
              "active_adapters": _active_adapters(model), "samples": details,
              "max_abs_diff_overall": max((x["max_abs_diff"] for x in details), default=0.0),
              "mean_abs_diff_overall": sum(x["mean_abs_diff"] for x in details) / len(details)}
    if not passed:
        raise EvaluationError(f"adapter 启用/关闭 logits 差异均未超过 atol={atol}，拒绝正式评估。")
    return result


def _atomic_json(path: Path, value: Any) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_output_dir(path: Path, overwrite: bool, *, create: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise EvaluationError(f"输出目录非空：{path}；如确认覆盖 v2 文件请传 --overwrite。")
    if create:
        path.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for name in OUTPUT_FILES:
                target = path / name
                if target.is_file():
                    target.unlink()
    return path


def _count_tokens(ids: Any, pad_id: int | None) -> int:
    values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    return sum(token != pad_id for token in values)


def generate_predictions(model: Any, tokenizer: Any, records: Sequence[dict[str, Any]], args: argparse.Namespace,
                         *, progress_factory: Any = None) -> list[dict[str, Any]]:
    import torch
    if progress_factory is None:
        from tqdm.auto import tqdm
        progress_factory = tqdm
    predictions: list[dict[str, Any]] = []
    batches = range(0, len(records), args.batch_size)
    for start in progress_factory(batches, total=math.ceil(len(records) / args.batch_size), desc="CMExam"):
        chunk = records[start:start + args.batch_size]
        prompts = [build_prompt(tokenizer, row["question"]) for row in chunk]
        untruncated = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=args.max_input_length)
        input_width = int(encoded["input_ids"].shape[1])
        encoded = _to_device(encoded, args.device)
        began = time.perf_counter()
        try:
            with torch.inference_mode():
                outputs = model.generate(**encoded, do_sample=False, num_beams=1,
                                         max_new_tokens=args.max_new_tokens, use_cache=True,
                                         pad_token_id=tokenizer.pad_token_id)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError("CMExam 生成发生 CUDA OOM；请显式调整 batch_size 或长度参数。") from exc
            raise
        elapsed = time.perf_counter() - began
        new_ids = outputs[:, input_width:]
        for offset, row in enumerate(chunk):
            raw_output = tokenizer.decode(new_ids[offset], skip_special_tokens=True)
            parsed = extract_predicted_labels(raw_output)
            gold = row["answer_labels"]
            predicted = list(parsed.labels or [])
            record = {
                "id": row["id"], "split": args.split, "question": row["question"], "options": row["options"],
                "gold_answer": format_answer_labels(gold), "gold_labels": list(gold), "raw_output": raw_output,
                "predicted_answer": format_answer_labels(predicted), "predicted_labels": predicted,
                "valid_prediction": parsed.valid, "format_valid": parsed.format_valid,
                "parse_source": parsed.source, "parse_error": parsed.error,
                "correct": exact_set_match(predicted, gold), "is_multiple_choice": len(gold) > 1,
                "input_tokens": _count_tokens(encoded["input_ids"][offset], tokenizer.pad_token_id),
                "generated_tokens": _count_tokens(new_ids[offset], tokenizer.pad_token_id),
                "input_truncated": untruncated[offset] > args.max_input_length,
                "latency_seconds": elapsed / len(chunk),
            }
            if args.peft_path is not None:
                record["peft_path"] = str(args.peft_path.expanduser().resolve())
            predictions.append(record)
    return predictions


def _group(rows: Sequence[dict[str, Any]], multi: bool | None = None) -> dict[str, Any]:
    subset = list(rows if multi is None else [r for r in rows if r["is_multiple_choice"] is multi])
    n = len(subset); correct = sum(r["correct"] for r in subset)
    return {"records": n, "correct": correct, "accuracy": correct / n if n else 0.0,
            "invalid_rate": sum(not r["valid_prediction"] for r in subset) / n if n else 0.0,
            "format_compliance_rate": sum(r["format_valid"] for r in subset) / n if n else 0.0}


def compute_metrics(rows: Sequence[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    overall = _group(rows); n = len(rows)
    metrics = {
        "records": n, "correct": overall["correct"], "exact_match_accuracy": overall["accuracy"],
        "valid_predictions": sum(r["valid_prediction"] for r in rows),
        "invalid_predictions": sum(not r["valid_prediction"] for r in rows),
        "invalid_prediction_rate": overall["invalid_rate"],
        "format_valid_predictions": sum(r["format_valid"] for r in rows),
        "format_compliance_rate": overall["format_compliance_rate"],
        "mean_input_tokens": sum(r["input_tokens"] for r in rows) / n,
        "mean_generated_tokens": sum(r["generated_tokens"] for r in rows) / n,
        "mean_latency_seconds": sum(r["latency_seconds"] for r in rows) / n,
        "elapsed_seconds": elapsed, "samples_per_second": n / elapsed if elapsed else 0.0,
        "truncated_input_count": sum(r["input_truncated"] for r in rows),
        "truncated_input_rate": sum(r["input_truncated"] for r in rows) / n,
        "single_choice": _group(rows, False), "multiple_choice": _group(rows, True),
        "by_gold_label_count": {},
        "gold_answer_combination_frequency": dict(Counter(r["gold_answer"] for r in rows)),
        "predicted_answer_combination_frequency": dict(Counter(r["predicted_answer"] for r in rows)),
        "invalid_error_distribution": dict(Counter(r["parse_error"] or "unknown" for r in rows if not r["valid_prediction"])),
    }
    metrics["single_choice"]["accuracy"] = metrics["single_choice"].pop("accuracy")
    multiple = metrics["multiple_choice"]
    multiple["exact_set_accuracy"] = multiple.pop("accuracy")
    for count in range(1, 6):
        subset = [r for r in rows if len(r["gold_labels"]) == count]
        metrics["by_gold_label_count"][str(count)] = _group(subset)
    return metrics


def _version(package: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    data_file = resolve_data_file(args.data_root, args.split)
    records = select_records(load_cmexam_records(data_file, args.split), args.max_samples, args.seed)
    output = prepare_output_dir(args.output_dir, args.overwrite, create=not args.dry_run)
    selected_ids = [row["id"] for row in records]
    if args.dry_run:
        print(json.dumps({"dry_run": True, "data_file": str(data_file), "records": len(records),
                          "selected_ids": selected_ids, "output_dir": str(output),
                          "enable_thinking": False}, ensure_ascii=False, indent=2))
        for row in records[:3]:
            print(f"[{row['id']}] {row['question']}")
        return 0
    started_at = datetime.now(timezone.utc).isoformat()
    tokenizer = load_tokenizer(args)
    model, model_info = load_model_with_optional_adapter(args)
    if args.peft_path is None:
        verification = {"requested": False, "passed": None, "skipped": False, "samples": []}
    elif args.skip_adapter_effect_check or not args.verify_adapter_effect:
        LOGGER.warning("强警告：用户显式跳过了 adapter logits 效果验证。")
        verification = {"requested": True, "passed": None, "skipped": True,
                        "reason": "explicit user request", "adapter_path": str(args.peft_path.expanduser().resolve()),
                        "active_adapters": _active_adapters(model), "samples": []}
    else:
        verification = verify_adapter_changes_logits(
            model, tokenizer, records, device=args.device, sample_count=args.adapter_check_samples,
            atol=args.adapter_check_atol, adapter_path=str(args.peft_path.expanduser().resolve()),
        )
    _atomic_json(output / "adapter_verification.json", verification)
    began = time.perf_counter()
    predictions = generate_predictions(model, tokenizer, records, args)
    elapsed = time.perf_counter() - began
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as pred_file, \
         (output / "error_cases.jsonl").open("w", encoding="utf-8") as error_file:
        for row in predictions:
            pred_file.write(json.dumps(row, ensure_ascii=False) + "\n"); pred_file.flush()
            if not row["correct"] or not row["valid_prediction"] or not row["format_valid"] or row["input_truncated"]:
                error_file.write(json.dumps(row, ensure_ascii=False) + "\n"); error_file.flush()
    metrics = compute_metrics(predictions, elapsed)
    _atomic_json(output / "metrics.json", metrics)
    import torch
    config = {"git_commit": _git_commit(), "model_name_or_path": args.model_name_or_path,
              "peft_path": str(args.peft_path.expanduser().resolve()) if args.peft_path else None,
              **model_info, "tokenizer_class": type(tokenizer).__name__, "data_file": str(data_file),
              "selected_ids": selected_ids, "split": args.split, "seed": args.seed, "batch_size": args.batch_size,
              "max_input_length": args.max_input_length, "max_new_tokens": args.max_new_tokens,
              "torch_dtype": args.torch_dtype, "device": args.device, "local_files_only": args.local_files_only,
              "trust_remote_code": args.trust_remote_code, "enable_thinking": False,
              "adapter_effect_check": {"verify": args.verify_adapter_effect, "samples": args.adapter_check_samples,
                                         "atol": args.adapter_check_atol, "skipped": args.skip_adapter_effect_check},
              "python_version": sys.version.split()[0], "pytorch_version": torch.__version__,
              "transformers_version": _version("transformers"), "peft_version": _version("peft"),
              "gpu_name": torch.cuda.get_device_name() if args.device.startswith("cuda") and torch.cuda.is_available() else None,
              "started_at": started_at, "ended_at": datetime.now(timezone.utc).isoformat()}
    _atomic_json(output / "evaluation_config.json", config)
    (output / "run_summary.txt").write_text(
        f"records={metrics['records']}\nexact_match_accuracy={metrics['exact_match_accuracy']:.6f}\n"
        f"model={model_info['final_model_class']}\nadapter={config['peft_path']}\n", encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        return run(parse_args(argv))
    except (EvaluationError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
