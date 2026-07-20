#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""统计 MedicalGPT 处理后 PT 和 SFT 数据的 token 长度分布。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer


SPLITS = ("train", "validation", "test")
THRESHOLDS = (256, 512, 768, 1024, 1536, 2048, 4096)


class DataFormatError(ValueError):
    """JSONL 数据格式不符合预期时抛出。"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="统计 MedicalGPT 数据 token 长度。")
    parser.add_argument(
        "--tokenizer_name_or_path",
        default="Qwen/Qwen3.5-2B-Base",
        help="Hugging Face tokenizer 名称或本地路径。",
    )
    parser.add_argument(
        "--pt_dir",
        type=Path,
        default=Path.home() / "datasets" / "medicalgpt" / "processed" / "pt",
        help="处理后 PT 数据目录。",
    )
    parser.add_argument(
        "--sft_dir",
        type=Path,
        default=Path.home() / "datasets" / "medicalgpt" / "processed" / "sft",
        help="处理后 SFT 数据目录。",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("reports/data/token_length_statistics.json"),
        help="JSON 统计报告路径。",
    )
    parser.add_argument("--cache_dir", type=Path, default=None, help="tokenizer 缓存目录。")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="允许 tokenizer 加载远程自定义代码。"
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有报告。")
    return parser.parse_args(argv)


def percentile(values: Sequence[int | float], percent: int) -> int | float:
    """使用 ``(n - 1) * p`` 线性插值计算分位数。"""

    if not values:
        raise ValueError("不能对空序列计算分位数。")
    if not 0 <= percent <= 100:
        raise ValueError("分位数百分比必须位于0到100之间。")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    result = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return int(result) if result.is_integer() else result


def distribution_statistics(values: Sequence[int | float]) -> dict[str, int | float]:
    """计算一组长度（或比例）的基本分布统计。"""

    if not values:
        raise ValueError("数据文件中没有可统计的记录。")
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def threshold_statistics(lengths: Sequence[int]) -> dict[str, dict[str, int | float]]:
    """计算严格大于各阈值的记录数和比例。"""

    if not lengths:
        raise ValueError("不能对空序列计算阈值统计。")
    return {
        str(threshold): {
            "count": sum(length > threshold for length in lengths),
            "ratio": sum(length > threshold for length in lengths) / len(lengths),
        }
        for threshold in THRESHOLDS
    }


def summarize_pt(lengths: Sequence[int]) -> dict[str, Any]:
    """汇总 PT 记录的完整 token 长度。"""

    result: dict[str, Any] = {"records": len(lengths), **distribution_statistics(lengths)}
    result["total_tokens"] = sum(lengths)
    result["over_thresholds"] = threshold_statistics(lengths)
    return result


def summarize_sft(
    source_lengths: Sequence[int],
    target_lengths: Sequence[int],
    total_lengths: Sequence[int],
) -> dict[str, Any]:
    """汇总 SFT 的 source、target、总长度及 target 占比。"""

    if not (len(source_lengths) == len(target_lengths) == len(total_lengths)):
        raise ValueError("SFT 长度列表数量不一致。")
    ratios = [target / total for target, total in zip(target_lengths, total_lengths)]
    total_stats = distribution_statistics(total_lengths)
    total_stats["sum"] = sum(total_lengths)
    total_stats["over_thresholds"] = threshold_statistics(total_lengths)
    return {
        "records": len(total_lengths),
        "source_tokens": distribution_statistics(source_lengths),
        "target_tokens": distribution_statistics(target_lengths),
        "total_tokens": total_stats,
        "target_token_ratio": distribution_statistics(ratios),
    }


def _load_json_object(path: Path, line: str, line_no: int) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DataFormatError(f"{path} 第 {line_no} 行不是合法JSON：{exc}") from exc
    if not isinstance(record, dict):
        raise DataFormatError(f"{path} 第 {line_no} 行应为JSON对象。")
    return record


def analyze_pt_file(path: Path, tokenizer: Any) -> list[int]:
    """逐行统计单个 PT JSONL 文件，不截断也不 padding。"""

    lengths: list[int] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                raise DataFormatError(f"{path} 第 {line_no} 行是空白行。")
            record = _load_json_object(path, line, line_no)
            if "text" not in record:
                raise DataFormatError(f"{path} 第 {line_no} 行缺少必需字段 'text'。")
            if not isinstance(record["text"], str):
                raise DataFormatError(f"{path} 第 {line_no} 行的 'text' 必须是字符串。")
            token_ids = tokenizer.encode(record["text"], add_special_tokens=True)
            lengths.append(len(token_ids))
    if not lengths:
        raise DataFormatError(f"{path} 是空数据文件。")
    return lengths


def ensure_chat_template(tokenizer: Any) -> None:
    """确保 tokenizer 有可用的 chat template。"""

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer 没有 chat_template，无法复现 SFT 训练时的 prompt。")
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise ValueError("tokenizer 不支持 apply_chat_template。")


def _parse_sft_conversation(path: Path, record: dict[str, Any], line_no: int) -> tuple[str, str]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise DataFormatError(f"{path} 第 {line_no} 行的 'conversations' 必须是两条消息的列表。")
    expected_roles = ("human", "gpt")
    values: list[str] = []
    for index, (message, expected_role) in enumerate(zip(conversations, expected_roles), start=1):
        if not isinstance(message, dict):
            raise DataFormatError(f"{path} 第 {line_no} 行的第 {index} 条对话必须是JSON对象。")
        if message.get("from") != expected_role:
            raise DataFormatError(
                f"{path} 第 {line_no} 行的第 {index} 条对话 'from' 必须为 '{expected_role}'。"
            )
        if "value" not in message or not isinstance(message["value"], str):
            raise DataFormatError(f"{path} 第 {line_no} 行的第 {index} 条对话 'value' 必须是字符串。")
        values.append(message["value"])
    return values[0], values[1]


def analyze_sft_file(path: Path, tokenizer: Any) -> tuple[list[int], list[int], list[int]]:
    """逐行复现单轮 SFT 编码并返回三类长度。"""

    ensure_chat_template(tokenizer)
    source_lengths: list[int] = []
    target_lengths: list[int] = []
    total_lengths: list[int] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                raise DataFormatError(f"{path} 第 {line_no} 行是空白行。")
            record = _load_json_object(path, line, line_no)
            user_text, assistant_text = _parse_sft_conversation(path, record, line_no)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            source = tokenizer.encode(prompt, add_special_tokens=True)
            target = tokenizer.encode(assistant_text, add_special_tokens=False)
            source_lengths.append(len(source))
            target_lengths.append(len(target))
            total_lengths.append(len(source) + len(target) + 1)
    if not total_lengths:
        raise DataFormatError(f"{path} 是空数据文件。")
    return source_lengths, target_lengths, total_lengths


def dataset_paths(root: Path) -> dict[str, Path]:
    """生成处理后数据集的三个分割路径。"""

    paths = {
        "train": root / "train" / "train.jsonl",
        "validation": root / "validation" / "validation.jsonl",
        "test": root / "test" / "test.jsonl",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少以下数据文件：\n" + "\n".join(f"  - {path}" for path in missing))
    return paths


def analyze_datasets(pt_dir: Path, sft_dir: Path, tokenizer: Any) -> dict[str, Any]:
    """统计 PT/SFT 各分割以及两类数据的整体分布。"""

    ensure_chat_template(tokenizer)
    pt_results: dict[str, Any] = {}
    all_pt: list[int] = []
    for split, path in dataset_paths(pt_dir).items():
        print(f"正在处理 PT {split}：{path}")
        lengths = analyze_pt_file(path, tokenizer)
        pt_results[split] = summarize_pt(lengths)
        all_pt.extend(lengths)
        print(f"  完成：{len(lengths):,}条，最大 {max(lengths):,} tokens")
    pt_results["overall"] = summarize_pt(all_pt)

    sft_results: dict[str, Any] = {}
    all_source: list[int] = []
    all_target: list[int] = []
    all_total: list[int] = []
    for split, path in dataset_paths(sft_dir).items():
        print(f"正在处理 SFT {split}：{path}")
        source, target, total = analyze_sft_file(path, tokenizer)
        sft_results[split] = summarize_sft(source, target, total)
        all_source.extend(source)
        all_target.extend(target)
        all_total.extend(total)
        print(f"  完成：{len(total):,}条，最大 {max(total):,} tokens")
    sft_results["overall"] = summarize_sft(all_source, all_target, all_total)
    return {"pt": pt_results, "sft": sft_results}


def write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    """通过同目录临时文件原子写入 JSON 报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace, tokenizer: Any | None = None) -> dict[str, Any]:
    """执行分析并写入报告，可注入 tokenizer 以便离线测试。"""

    pt_dir = args.pt_dir.expanduser().resolve()
    sft_dir = args.sft_dir.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出报告已存在：{output_path}\n如需覆盖，请增加 --overwrite。")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            cache_dir=str(args.cache_dir.expanduser()) if args.cache_dir else None,
            trust_remote_code=args.trust_remote_code,
        )
    ensure_chat_template(tokenizer)
    statistics_report = analyze_datasets(pt_dir, sft_dir, tokenizer)
    report: dict[str, Any] = {
        "configuration": {
            "tokenizer_name_or_path": args.tokenizer_name_or_path,
            "pt_dir": str(pt_dir),
            "sft_dir": str(sft_dir),
            "output_path": str(output_path),
            "cache_dir": str(args.cache_dir.expanduser()) if args.cache_dir else None,
            "trust_remote_code": args.trust_remote_code,
            "percentile_method": "linear_interpolation_(n-1)*p",
            "thresholds": list(THRESHOLDS),
        },
        **statistics_report,
    }
    write_json_atomic(output_path, report)
    print("\nToken 长度统计完成：")
    print(f"  PT 记录：{report['pt']['overall']['records']:,}")
    print(f"  SFT 记录：{report['sft']['overall']['records']:,}")
    print(f"  报告：{output_path}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：成功返回0，可预期错误返回1。"""

    try:
        args = parse_args(argv)
        run(args)
        return 0
    except (DataFormatError, FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
