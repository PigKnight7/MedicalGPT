#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""使用统一 prompt、生成配置、解析器和指标评估 CMExam。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

# 直接执行脚本时将仓库根目录加入导入路径，不影响模块方式运行。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medicalgpt_ext.cmexam_utils import (  # noqa: E402
    exact_set_match,
    extract_predicted_labels,
    format_answer_labels,
    format_cmexam_prompt,
    normalize_answer_labels,
    validate_options,
)


OUTPUT_FILENAMES = (
    "evaluation_config.json",
    "predictions.jsonl",
    "metrics.json",
    "error_cases.jsonl",
    "run_summary.txt",
)
REQUIRED_RECORD_FIELDS = {
    "id", "split", "question", "stem", "options", "answer", "answer_labels",
    "is_multiple_choice", "metadata",
}


class EvaluationError(ValueError):
    """评估配置、数据或恢复文件不合法时抛出。"""


class DryRunTokenizer:
    """dry-run 专用模板渲染器，不访问模型仓库或网络。"""

    chat_template = "dry-run-chatml"

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        if tokenize:
            raise ValueError("DryRunTokenizer 仅用于 tokenize=False。")
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in conversation
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析统一评估 CLI 参数。"""

    parser = argparse.ArgumentParser(description="统一评估 CMExam 单选和多选题。")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--peft_path")
    parser.add_argument(
        "--data_root", type=Path,
        default=Path.home() / "datasets/medicalgpt/processed/cmexam/official",
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--allow_test_evaluation", action="store_true")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_style", choices=["answer_only", "reasoning_and_answer"], default="answer_only")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--torch_dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto"
    )
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--group_by", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """在任何模型加载前验证危险选项和数值参数。"""

    if args.split == "test" and not args.allow_test_evaluation:
        raise EvaluationError("test集只能用于最终配置评估，不应在开发阶段反复使用。")
    if args.resume and args.overwrite:
        raise EvaluationError("--resume 与 --overwrite 不能同时使用。")
    for name in ("batch_size", "max_input_length", "max_new_tokens", "num_beams"):
        if getattr(args, name) <= 0:
            raise EvaluationError(f"--{name} 必须大于 0。")
    if args.max_samples is not None and args.max_samples <= 0:
        raise EvaluationError("--max_samples 必须大于 0。")
    if args.do_sample and args.temperature <= 0:
        raise EvaluationError("采样生成时 --temperature 必须大于 0。")
    if not 0 < args.top_p <= 1:
        raise EvaluationError("--top_p 必须在 (0, 1] 范围内。")


def data_path_for(data_root: Path, split: str) -> Path:
    """只解析 official validation/test 的固定路径。"""

    return data_root.expanduser().resolve() / split / f"{split}.jsonl"


def load_cmexam_records(path: Path, *, expected_split: str) -> list[dict[str, Any]]:
    """逐行读取并严格校验 official CMExam JSONL。"""

    if not path.is_file():
        raise FileNotFoundError(f"CMExam 数据文件不存在：{path}")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise EvaluationError(f"{path} 第 {line_number} 行为空行。")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path} 第 {line_number} 行不是合法 JSON：{exc}") from exc
            if not isinstance(record, dict):
                raise EvaluationError(f"{path} 第 {line_number} 行必须是 JSON 对象。")
            missing = REQUIRED_RECORD_FIELDS - record.keys()
            if missing:
                raise EvaluationError(f"{path} 第 {line_number} 行缺少字段：{sorted(missing)}")
            record_id = record["id"]
            if not isinstance(record_id, str) or not record_id:
                raise EvaluationError(f"{path} 第 {line_number} 行 id 无效。")
            if record_id in seen_ids:
                raise EvaluationError(f"{path} 存在重复 id：{record_id}")
            seen_ids.add(record_id)
            if record["split"] != expected_split:
                raise EvaluationError(f"样本 {record_id} 的 split={record['split']!r}，预期为 {expected_split!r}。")
            if not isinstance(record["question"], str) or not record["question"].strip():
                raise EvaluationError(f"样本 {record_id} 的 question 为空。")
            options = validate_options(record["options"])
            gold = normalize_answer_labels(record["answer_labels"])
            answer_gold = normalize_answer_labels(record["answer"])
            if gold is None or gold != answer_gold:
                raise EvaluationError(f"样本 {record_id} 的 answer 与 answer_labels 不一致。")
            option_labels = {item["label"] for item in options}
            if not set(gold).issubset(option_labels):
                raise EvaluationError(f"样本 {record_id} 的答案标签不在实际选项中。")
            if not isinstance(record["metadata"], dict):
                raise EvaluationError(f"样本 {record_id} 的 metadata 必须是对象。")
            records.append(record)
    return records


def select_records(
    records: Sequence[dict[str, Any]], *, max_samples: int | None, seed: int
) -> list[dict[str, Any]]:
    """使用固定随机种子抽样，并按原始位置排序以便稳定恢复。"""

    if max_samples is None or max_samples >= len(records):
        return list(records)
    indices = sorted(random.Random(seed).sample(range(len(records)), max_samples))
    return [records[index] for index in indices]


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _basic_group(records: Sequence[Mapping[str, Any]], *, multiple: bool) -> dict[str, Any]:
    count = len(records)
    correct = sum(bool(item["correct"]) for item in records)
    invalid = sum(not bool(item["valid_prediction"]) for item in records)
    format_valid = sum(bool(item["format_valid"]) for item in records)
    result = {
        "records": count,
        "correct": correct,
        "invalid_rate": _ratio(invalid, count),
        "format_compliance_rate": _ratio(format_valid, count),
    }
    result["exact_set_accuracy" if multiple else "accuracy"] = _ratio(correct, count)
    return result


def _metadata_groups(
    predictions: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    grouped: dict[str, Any] = {}
    missing: list[str] = []
    for field in fields:
        if not any(field in item.get("metadata", {}) for item in predictions):
            missing.append(field)
            continue
        buckets: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in predictions:
            value = item.get("metadata", {}).get(field, "<missing>")
            buckets[str(value)].append(item)
        grouped[field] = {
            value: {
                "records": len(items),
                "correct": sum(bool(item["correct"]) for item in items),
                "accuracy": _ratio(sum(bool(item["correct"]) for item in items), len(items)),
                "invalid_rate": _ratio(sum(not bool(item["valid_prediction"]) for item in items), len(items)),
            }
            for value, items in sorted(buckets.items())
        }
    return grouped, missing


def calculate_metrics(
    predictions: Sequence[Mapping[str, Any]], *, group_by: Sequence[str] = ()
) -> dict[str, Any]:
    """从完整 predictions 重新计算全部统一指标。"""

    records = len(predictions)
    correct = sum(bool(item["correct"]) for item in predictions)
    valid = sum(bool(item["valid_prediction"]) for item in predictions)
    format_valid = sum(bool(item["format_valid"]) for item in predictions)
    generated_tokens = sum(int(item["generated_tokens"]) for item in predictions)
    latency = sum(float(item["latency_seconds"]) for item in predictions)
    single = [item for item in predictions if not item["is_multiple_choice"]]
    multiple = [item for item in predictions if item["is_multiple_choice"]]

    tp = fp = fn = 0
    omitted: Counter[str] = Counter()
    extra: Counter[str] = Counter()
    for item in multiple:
        gold = set(item["gold_labels"])
        predicted = set(item["predicted_labels"] or [])
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        if gold - predicted:
            omitted["".join(sorted(gold - predicted))] += 1
        if predicted - gold:
            extra["".join(sorted(predicted - gold))] += 1
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)
    multiple_metrics = _basic_group(multiple, multiple=True)
    multiple_metrics.update({"micro_precision": precision, "micro_recall": recall, "micro_f1": f1})

    by_count: dict[str, Any] = {}
    for label_count in range(1, 6):
        items = [item for item in predictions if len(item["gold_labels"]) == label_count]
        by_count[str(label_count)] = {
            "records": len(items),
            "correct": sum(bool(item["correct"]) for item in items),
            "accuracy": _ratio(sum(bool(item["correct"]) for item in items), len(items)),
            "invalid_rate": _ratio(sum(not bool(item["valid_prediction"]) for item in items), len(items)),
        }

    label_accuracy: dict[str, Any] = {}
    for label in "ABCDE":
        items = [item for item in single if item["gold_answer"] == label]
        label_accuracy[label] = {
            "records": len(items),
            "correct": sum(bool(item["correct"]) for item in items),
            "accuracy": _ratio(sum(bool(item["correct"]) for item in items), len(items)),
        }
    metadata_groups, missing_fields = _metadata_groups(predictions, group_by)
    invalid_reasons = Counter(
        str(item.get("parse_error") or "unknown") for item in predictions if not item["valid_prediction"]
    )
    return {
        "overall": {
            "records": records,
            "correct": correct,
            "exact_match_accuracy": _ratio(correct, records),
            "valid_predictions": valid,
            "invalid_predictions": records - valid,
            "invalid_prediction_rate": _ratio(records - valid, records),
            "format_valid_predictions": format_valid,
            "format_compliance_rate": _ratio(format_valid, records),
            "mean_input_tokens": mean([int(item["input_tokens"]) for item in predictions]) if records else 0.0,
            "mean_generated_tokens": mean([int(item["generated_tokens"]) for item in predictions]) if records else 0.0,
            "mean_latency_seconds": mean([float(item["latency_seconds"]) for item in predictions]) if records else 0.0,
            "samples_per_second": _ratio(records, latency),
            "generated_tokens_per_second": _ratio(generated_tokens, latency),
            "truncated_input_count": sum(bool(item["input_truncated"]) for item in predictions),
            "truncated_input_rate": _ratio(sum(bool(item["input_truncated"]) for item in predictions), records),
        },
        "single_choice": _basic_group(single, multiple=False),
        "multiple_choice": multiple_metrics,
        "by_answer_label_count": by_count,
        "metadata_groups": metadata_groups,
        "missing_group_by_fields": missing_fields,
        "frequencies": {
            "gold_answer_combinations": dict(sorted(Counter(str(item["gold_answer"]) for item in predictions).items())),
            "predicted_answer_combinations": dict(sorted(Counter(str(item["predicted_answer"] or "<invalid>") for item in predictions).items())),
            "invalid_reasons": dict(invalid_reasons.most_common()),
            "single_choice_label_accuracy": label_accuracy,
            "multiple_choice_common_omissions": dict(omitted.most_common(20)),
            "multiple_choice_common_extra_labels": dict(extra.most_common(20)),
        },
    }


def atomic_write_json(path: Path, content: Mapping[str, Any]) -> None:
    """原子写入 JSON 文件。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(content, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_predictions(path: Path) -> list[dict[str, Any]]:
    """读取恢复文件并拒绝损坏 JSON 或重复 ID。"""

    predictions: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path} 第 {line_number} 行损坏：{exc}") from exc
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise EvaluationError(f"{path} 第 {line_number} 行缺少有效 id。")
            if item["id"] in seen:
                raise EvaluationError(f"{path} 存在重复预测 id：{item['id']}")
            seen.add(item["id"])
            predictions.append(item)
    return predictions


def prepare_output_dir(output_dir: Path, *, resume: bool, overwrite: bool) -> dict[str, Path]:
    """应用拒绝覆盖、明确覆盖或恢复策略。"""

    output_dir = output_dir.expanduser().resolve()
    paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    if resume:
        if not paths["evaluation_config.json"].is_file() or not paths["predictions.jsonl"].is_file():
            raise EvaluationError("--resume 要求已有 evaluation_config.json 和 predictions.jsonl。")
        return paths
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"输出目录非空：{output_dir}；请使用 --overwrite 或 --resume。")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in paths.values():
            path.unlink(missing_ok=True)
    return paths


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, capture_output=True, check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _redact_location(value: str | None) -> str | None:
    """移除 URL 用户信息和查询参数，防止配置文件意外记录凭据。"""

    if value is None or "://" not in value:
        return value
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname += f":{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def generation_parameters(args: argparse.Namespace) -> dict[str, Any]:
    """生成传给 model.generate 的公平配置；确定性模式排除采样参数。"""

    parameters: dict[str, Any] = {
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }
    if args.do_sample:
        parameters.update({"temperature": args.temperature, "top_p": args.top_p})
    return parameters


def resume_signature(args: argparse.Namespace, data_path: Path, selected_ids: Sequence[str]) -> dict[str, Any]:
    """构造必须完全一致的恢复配置签名。"""

    return {
        "model_name_or_path": _redact_location(args.model_name_or_path),
        "peft_path": _redact_location(args.peft_path),
        "data_path": str(data_path),
        "split": args.split,
        "prompt_style": args.prompt_style,
        "batch_size": args.batch_size,
        "max_input_length": args.max_input_length,
        "generation": generation_parameters(args),
        "seed": args.seed,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "group_by": list(args.group_by),
        "selected_ids": list(selected_ids),
    }


def build_config(
    args: argparse.Namespace,
    *,
    data_path: Path,
    selected_ids: Sequence[str],
    started_at: str,
    tokenizer: Any | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """构造不含凭据的可复现实验配置。"""

    import platform
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        torch_version = torch.__version__
    except ImportError:
        gpu_name = None
        torch_version = None
    signature = resume_signature(args, data_path, selected_ids)
    return {
        "git_commit": _git_commit(),
        **signature,
        "cache_dir": str(args.cache_dir.expanduser().resolve()) if args.cache_dir else None,
        "max_samples": args.max_samples,
        "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
        "model_class": type(model).__name__ if model is not None else None,
        "versions": {
            "python": platform.python_version(), "torch": torch_version,
            "transformers": _version("transformers"), "peft": _version("peft"),
        },
        "gpu_name": gpu_name,
        "started_at": started_at,
        "ended_at": None,
        "resume_signature": signature,
    }


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    """加载基础/完整模型，并可选叠加只读 PEFT adapter。"""

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForConditionalGeneration, AutoTokenizer

    common = {
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left", **common)
    if not getattr(tokenizer, "chat_template", None):
        raise EvaluationError("tokenizer 没有 chat_template，无法执行统一 CMExam 评估。")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise EvaluationError("tokenizer 同时缺少 pad_token 和 eos_token。")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = args.torch_dtype if args.torch_dtype == "auto" else getattr(torch, args.torch_dtype)
    config = AutoConfig.from_pretrained(args.model_name_or_path, **common)
    architectures = getattr(config, "architectures", None) or []
    model_class = (
        AutoModelForConditionalGeneration
        if any("ConditionalGeneration" in architecture for architecture in architectures)
        else AutoModelForCausalLM
    )
    model_kwargs = {**common, "torch_dtype": dtype, "low_cpu_mem_usage": True}
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    model = model_class.from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise EvaluationError("指定了 --device cuda，但当前环境没有可用 CUDA。")
        model = model.to("cuda")
    elif args.device == "cpu":
        model = model.to("cpu")
    if args.peft_path:
        model = PeftModel.from_pretrained(
            model, args.peft_path, is_trainable=False,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=args.local_files_only,
        )
    model.eval()
    return model, tokenizer


def _model_input_device(model: Any, requested_device: str) -> Any:
    import torch
    if requested_device == "cuda":
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _prediction_record(
    source: Mapping[str, Any], raw_output: str, *, input_tokens: int,
    generated_tokens: int, input_truncated: bool, latency_seconds: float,
) -> dict[str, Any]:
    parsed = extract_predicted_labels(raw_output)
    gold = normalize_answer_labels(source["answer_labels"])
    assert gold is not None
    return {
        "id": source["id"], "split": source["split"], "stem": source["stem"],
        "options": source["options"], "gold_answer": format_answer_labels(gold),
        "gold_labels": list(gold), "raw_output": raw_output,
        "predicted_answer": format_answer_labels(parsed.labels) or None,
        "predicted_labels": list(parsed.labels) if parsed.labels else None,
        "valid_prediction": parsed.valid, "format_valid": parsed.format_valid,
        "parse_source": parsed.source, "parse_error": parsed.error,
        "correct": exact_set_match(parsed.labels, gold),
        "is_multiple_choice": len(gold) > 1, "metadata": source["metadata"],
        "input_tokens": input_tokens, "generated_tokens": generated_tokens,
        "input_truncated": input_truncated, "latency_seconds": latency_seconds,
    }


def evaluate_pending_records(
    records: Sequence[dict[str, Any]], *, model: Any, tokenizer: Any,
    args: argparse.Namespace, prediction_file: Any,
) -> list[dict[str, Any]]:
    """小批量生成；只解码输入张量宽度之后的新增 token。"""

    import torch
    prompts = [format_cmexam_prompt(tokenizer, item["question"], prompt_style=args.prompt_style) for item in records]
    predictions: list[dict[str, Any]] = []
    device = _model_input_device(model, args.device)
    generation = generation_parameters(args)
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start:start + args.batch_size]
        batch_prompts = prompts[start:start + args.batch_size]
        raw_lengths = [
            len(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
            for prompt in batch_prompts
        ]
        encoded = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_input_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_width = int(encoded["input_ids"].shape[1])
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                outputs = model.generate(**encoded, **generation)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA 显存不足。请显式减小 --batch_size、--max_input_length 或 --max_new_tokens 后重试。"
            ) from exc
        elapsed_per_sample = (time.perf_counter() - started) / len(batch_records)
        new_tokens = outputs[:, input_width:]
        for index, (record, token_row, raw_length) in enumerate(zip(batch_records, new_tokens, raw_lengths)):
            raw_output = tokenizer.decode(token_row, skip_special_tokens=True).strip()
            if getattr(tokenizer, "pad_token_id", None) is None:
                generated_count = int(token_row.shape[0])
            else:
                generated_count = int((token_row != tokenizer.pad_token_id).sum().item())
            input_count = int(encoded["attention_mask"][index].sum().item())
            prediction = _prediction_record(
                record, raw_output, input_tokens=input_count, generated_tokens=generated_count,
                input_truncated=raw_length > args.max_input_length,
                latency_seconds=elapsed_per_sample,
            )
            json.dump(prediction, prediction_file, ensure_ascii=False)
            prediction_file.write("\n")
            prediction_file.flush()
            predictions.append(prediction)
    return predictions


def write_error_cases(path: Path, predictions: Iterable[Mapping[str, Any]]) -> None:
    """原子写入错误、格式失败或截断样本。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            for item in predictions:
                if not item["correct"] or not item["valid_prediction"] or not item["format_valid"] or item["input_truncated"]:
                    json.dump(item, file, ensure_ascii=False)
                    file.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run_dry_run(
    args: argparse.Namespace,
    records: Sequence[dict[str, Any]],
    data_path: Path,
    *,
    all_records: Sequence[dict[str, Any]] | None = None,
) -> int:
    """不加载任何模型/tokenizer 权重地验证数据、prompt 和基础统计。"""

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空：{output_dir}；dry-run 请使用 --overwrite。")
    tokenizer = DryRunTokenizer()
    complete = list(all_records) if all_records is not None else list(records)
    complete_single = sum(not item["is_multiple_choice"] for item in complete)
    selected_single = sum(not item["is_multiple_choice"] for item in records)
    metadata = Counter(key for item in complete for key in item["metadata"])
    print(f"dry-run 数据：{data_path}")
    print(
        f"完整数据：{len(complete)}；单选：{complete_single}；"
        f"多选：{len(complete) - complete_single}"
    )
    print(
        f"抽样数据：{len(records)}；单选：{selected_single}；"
        f"多选：{len(records) - selected_single}"
    )
    print(f"metadata 字段：{dict(metadata)}")
    preview = list(records[:3])
    if not any(item["is_multiple_choice"] for item in preview):
        multiple_example = next((item for item in complete if item["is_multiple_choice"]), None)
        if multiple_example is not None:
            preview = preview[:2] + [multiple_example]
    for index, record in enumerate(preview, start=1):
        print(f"\n--- Prompt {index} ({record['id']}) ---")
        print(format_cmexam_prompt(tokenizer, record["question"], prompt_style=args.prompt_style))
    print("\ndry-run 完成：未加载模型或外部 tokenizer，未生成正式预测文件。")
    return 0


def run(args: argparse.Namespace, *, model: Any | None = None, tokenizer: Any | None = None) -> int:
    """执行 dry-run 或正式评估；注入模型仅供离线测试。"""

    validate_args(args)
    random.seed(args.seed)
    data_path = data_path_for(args.data_root, args.split)
    all_records = load_cmexam_records(data_path, expected_split=args.split)
    records = select_records(all_records, max_samples=args.max_samples, seed=args.seed)
    if args.dry_run:
        return run_dry_run(args, records, data_path, all_records=all_records)

    paths = prepare_output_dir(args.output_dir, resume=args.resume, overwrite=args.overwrite)
    selected_ids = [item["id"] for item in records]
    started_at = datetime.now(timezone.utc).isoformat()
    previous_predictions: list[dict[str, Any]] = []
    if args.resume:
        old_config = json.loads(paths["evaluation_config.json"].read_text(encoding="utf-8"))
        expected = resume_signature(args, data_path, selected_ids)
        if old_config.get("resume_signature") != expected:
            raise EvaluationError("当前配置与已有 evaluation_config.json 不兼容，拒绝 resume。")
        previous_predictions = read_predictions(paths["predictions.jsonl"])
        selected_set = set(selected_ids)
        unknown = [item["id"] for item in previous_predictions if item["id"] not in selected_set]
        if unknown:
            raise EvaluationError(f"恢复文件包含不属于当前抽样的数据 ID：{unknown[0]}")

    completed_ids = {item["id"] for item in previous_predictions}
    pending = [item for item in records if item["id"] not in completed_ids]
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(args)
    else:
        if not getattr(tokenizer, "chat_template", None):
            raise EvaluationError("tokenizer 没有 chat_template。")
        model.eval()
    config = build_config(
        args, data_path=data_path, selected_ids=selected_ids, started_at=started_at,
        tokenizer=tokenizer, model=model,
    )
    if not args.resume:
        atomic_write_json(paths["evaluation_config.json"], config)

    mode = "a" if args.resume else "w"
    with paths["predictions.jsonl"].open(mode, encoding="utf-8", newline="\n") as prediction_file:
        evaluate_pending_records(
            pending, model=model, tokenizer=tokenizer, args=args, prediction_file=prediction_file,
        )
    predictions = read_predictions(paths["predictions.jsonl"])
    if len(predictions) != len(records):
        raise EvaluationError(f"预测数量 {len(predictions)} 与选中样本数量 {len(records)} 不一致。")
    metrics = calculate_metrics(predictions, group_by=args.group_by)
    for field in metrics["missing_group_by_fields"]:
        print(f"警告：metadata 中不存在分组字段 {field!r}。", file=sys.stderr)
    atomic_write_json(paths["metrics.json"], metrics)
    write_error_cases(paths["error_cases.jsonl"], predictions)
    config["ended_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(paths["evaluation_config.json"], config)
    summary = (
        f"CMExam {args.split} evaluation\n"
        f"records: {metrics['overall']['records']}\n"
        f"correct: {metrics['overall']['correct']}\n"
        f"exact_match_accuracy: {metrics['overall']['exact_match_accuracy']:.6f}\n"
        f"invalid_prediction_rate: {metrics['overall']['invalid_prediction_rate']:.6f}\n"
        f"format_compliance_rate: {metrics['overall']['format_compliance_rate']:.6f}\n"
    )
    paths["run_summary.txt"].write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口，成功返回 0。"""

    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EvaluationError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
