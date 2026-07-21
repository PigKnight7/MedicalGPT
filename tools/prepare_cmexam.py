#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""将官方 CMExam CSV 转换为 official 与去污染训练数据。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


VALID_ANSWER_LABELS = "ABCDE"
VALID_ANSWER_SET = frozenset(VALID_ANSWER_LABELS)
IGNORED_METADATA_COLUMNS = {"", "index", "unnamed: 0", "unnamed: 0.1"}

# 标签后必须有标点或空白，避免把普通正文开头的大写字母当成选项。
OPTION_PATTERN = re.compile(
    r"^\s*([A-Ea-e])(?:\s*[.．、:：)\]）]\s*|\s+)(.*)$"
)
ANSWER_PREFIX_PATTERN = re.compile(r"^\s*(?:答案|正确答案|选项)\s*[:：]?\s*", re.I)
ANSWER_SEPARATOR_PATTERN = re.compile(r"[\s,，、;；/|+]+")


class DataFormatError(ValueError):
    """CMExam 原始记录无法可靠恢复时抛出。"""


@dataclass
class SplitStats:
    """一个官方划分的处理统计。"""

    split: str
    source: str
    total_rows: int = 0
    written_records: int = 0
    single_answer_records: int = 0
    multiple_answer_records: int = 0
    invalid_rows: int = 0
    empty_question: int = 0
    empty_options: int = 0
    invalid_options: int = 0
    invalid_answer: int = 0
    answer_not_in_options: int = 0
    within_split_duplicates: int = 0
    answer_label_count_distribution: dict[str, int] = field(default_factory=dict)
    raw_answer_formats: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ColumnMapping:
    """官方 CSV 核心字段的实际列名。"""

    question: str
    options: str
    answer: str
    explanation: str | None
    record_id: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="处理官方 CMExam 数据。")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path.home() / "datasets/medicalgpt/raw/cmexam/CMExam/data",
        help="包含 train.csv、val.csv 和 test_with_annotations.csv 的目录。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.home() / "datasets/medicalgpt/processed/cmexam",
        help="处理后的 CMExam 输出目录。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="遇到第一条非法记录时立即终止。",
    )
    parser.add_argument(
        "--max_error_examples",
        type=int,
        default=10,
        help="报告中每个划分最多保留的非法记录示例数，默认 10。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有输出文件。",
    )
    parser.add_argument(
        "--drop_within_split_duplicates",
        action="store_true",
        help="删除各官方划分内部的精确重复；默认只统计并保留。",
    )
    return parser.parse_args(argv)


def normalize_text(value: str) -> str:
    """统一换行、删除 NUL 与行尾空白，不改写正文内容。"""

    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.strip() for line in value.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    normalized: list[str] = []
    for line in lines:
        if line or not normalized or normalized[-1]:
            normalized.append(line)
    return "\n".join(normalized)


def classify_answer_format(raw_value: str) -> str:
    """将原始答案归入稳定、可汇总的格式类别。"""

    value = normalize_text(raw_value)
    if not value:
        return "empty"
    if re.fullmatch(r"[A-Ea-e]", value):
        return "single_label"
    if re.fullmatch(r"[A-Ea-e]{2,}", value):
        return "concatenated_labels"
    if ANSWER_PREFIX_PATTERN.match(value):
        return "prefixed_labels"
    if re.search(r"[,，、;；/|+]", value):
        return "separated_labels"
    return "other"


def normalize_answer(value: str) -> str:
    """解析单选或多选答案，并按 A-E 排序、去重。"""

    normalized = normalize_text(value).upper()
    if not normalized:
        raise DataFormatError("答案为空。")

    payload = ANSWER_PREFIX_PATTERN.sub("", normalized, count=1).strip()
    payload = payload.strip(".．:：()（）[]【】")
    compact = ANSWER_SEPARATOR_PATTERN.sub("", payload)
    if not compact:
        raise DataFormatError("答案为空。")
    if not compact.isalpha() or any(label not in VALID_ANSWER_SET for label in compact):
        raise DataFormatError(f"答案包含 A-E 之外的标签或无法识别的文本：{value!r}")

    # 多个答案表达若可归一为同一标签集合才可接受；连续标签本身表示多选。
    clauses = [part for part in re.split(r"(?:答案|正确答案|选项)\s*[:：]?", normalized) if part.strip()]
    clause_sets: list[frozenset[str]] = []
    for clause in clauses:
        candidate = ANSWER_SEPARATOR_PATTERN.sub("", clause.strip(".．:：()（）[]【】 "))
        if candidate and candidate.isalpha() and all(ch in VALID_ANSWER_SET for ch in candidate):
            clause_sets.append(frozenset(candidate))
    if len(set(clause_sets)) > 1:
        raise DataFormatError(f"答案包含互相冲突的表达：{value!r}")

    return "".join(label for label in VALID_ANSWER_LABELS if label in compact)


def parse_options(value: str) -> list[dict[str, str]]:
    """解析逐行选项；未出现标签的后续行归入上一选项。"""

    normalized = normalize_text(value)
    if not normalized:
        raise DataFormatError("选项为空。")

    parsed: list[tuple[str, str]] = []
    current_label: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_label, current_lines
        if current_label is None:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            raise DataFormatError(f"选项 {current_label} 内容为空。")
        parsed.append((current_label, text))
        current_label = None
        current_lines = []

    for line in normalized.splitlines():
        if not line.strip():
            continue
        match = OPTION_PATTERN.match(line)
        if match:
            flush()
            current_label = match.group(1).upper()
            current_lines = [match.group(2).strip()]
        elif current_label is None:
            raise DataFormatError(f"首个选项之前存在无法解析的正文：{line!r}")
        else:
            current_lines.append(line.strip())
    flush()

    labels: set[str] = set()
    options: list[dict[str, str]] = []
    for label, text in parsed:
        if label in labels:
            raise DataFormatError(f"选项标签重复：{label}")
        labels.add(label)
        options.append({"label": label, "text": text})
    if len(options) < 2:
        raise DataFormatError(f"有效选项不足两个，实际为 {len(options)} 个。")
    return sorted(options, key=lambda option: VALID_ANSWER_LABELS.index(option["label"]))


def format_question(stem: str, options: Sequence[Mapping[str, str]]) -> str:
    """构造评估与训练共用的题目文本。"""

    return "\n".join([stem, *(f"{item['label']}. {item['text']}" for item in options)])


def question_digest(stem: str, options: Sequence[Mapping[str, str]]) -> str:
    """按规范化题干和结构化选项计算精确摘要。"""

    payload = json.dumps(
        {"stem": normalize_text(stem), "options": list(options)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_column_name(name: str) -> str:
    """规范化列名以支持大小写和分隔符差异。"""

    return re.sub(r"[\s_-]+", "", name.strip().lower())


def find_column(
    fieldnames: Sequence[str], aliases: Sequence[str], *, required: bool
) -> str | None:
    """按候选别名查找 CSV 列。"""

    columns = {normalize_column_name(name): name for name in fieldnames if name is not None}
    for alias in aliases:
        if normalize_column_name(alias) in columns:
            return columns[normalize_column_name(alias)]
    if required:
        raise DataFormatError(f"缺少必需 CSV 字段 {list(aliases)}；实际字段为 {list(fieldnames)}")
    return None


def resolve_columns(fieldnames: Sequence[str] | None) -> ColumnMapping:
    """解析官方 CSV 的核心字段。"""

    if not fieldnames:
        raise DataFormatError("CSV 文件没有表头。")
    return ColumnMapping(
        question=find_column(fieldnames, ["Question", "stem"], required=True),
        options=find_column(fieldnames, ["Options", "choices"], required=True),
        answer=find_column(fieldnames, ["Answer", "label"], required=True),
        explanation=find_column(fieldnames, ["Explanation", "analysis", "rationale"], required=False),
        record_id=find_column(fieldnames, ["ID", "QuestionID", "question_id"], required=False),
    )


def get_cell(row: Mapping[str, str | None], column: str | None) -> str:
    """安全读取并规范化一个 CSV 单元格。"""

    if column is None or row.get(column) is None:
        return ""
    return normalize_text(str(row[column]))


def build_metadata(row: Mapping[str, str | None], mapping: ColumnMapping) -> dict[str, str]:
    """保留核心字段之外的官方注释字段。"""

    core = {mapping.question, mapping.options, mapping.answer, mapping.explanation, mapping.record_id}
    ignored = {normalize_column_name(name) for name in IGNORED_METADATA_COLUMNS}
    metadata: dict[str, str] = {}
    for key, raw_value in row.items():
        if key is None or key in core or normalize_column_name(key) in ignored or raw_value is None:
            continue
        value = normalize_text(str(raw_value))
        if value:
            metadata[key.strip()] = value
    return metadata


def set_csv_field_size_limit() -> None:
    """在平台允许范围内提高 CSV 字段长度上限。"""

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _record_error(
    *,
    strict: bool,
    stats: SplitStats,
    examples: list[dict[str, object]],
    max_examples: int,
    source_path: Path,
    row_number: int,
    reason: str,
    raw_answer: str = "",
) -> None:
    """严格模式报错；非严格模式计数并保存有限示例。"""

    stats.invalid_rows += 1
    message = f"{source_path} 第 {row_number} 行：{reason}"
    if strict:
        raise DataFormatError(message)
    if len(examples) < max_examples:
        examples.append({"row_number": row_number, "reason": reason, "raw_answer": raw_answer})


def process_split(
    *,
    source_path: Path,
    split: str,
    strict: bool,
    max_error_examples: int,
    drop_within_split_duplicates: bool = False,
) -> tuple[list[dict[str, object]], set[str], SplitStats, list[dict[str, object]]]:
    """流式读取一个官方划分，并默认保留内部重复记录。"""

    stats = SplitStats(split=split, source=str(source_path))
    records: list[dict[str, object]] = []
    hashes: set[str] = set()
    examples: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    answer_formats: Counter[str] = Counter()

    with source_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        mapping = resolve_columns(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            stats.total_rows += 1
            stem = get_cell(row, mapping.question)
            if not stem:
                stats.empty_question += 1
                _record_error(strict=strict, stats=stats, examples=examples, max_examples=max_error_examples,
                              source_path=source_path, row_number=row_number, reason="题干为空。")
                continue

            raw_options = get_cell(row, mapping.options)
            if not raw_options:
                stats.empty_options += 1
                _record_error(strict=strict, stats=stats, examples=examples, max_examples=max_error_examples,
                              source_path=source_path, row_number=row_number, reason="选项为空。")
                continue
            try:
                options = parse_options(raw_options)
            except DataFormatError as exc:
                stats.invalid_options += 1
                _record_error(strict=strict, stats=stats, examples=examples, max_examples=max_error_examples,
                              source_path=source_path, row_number=row_number, reason=f"选项格式错误：{exc}")
                continue

            raw_answer = get_cell(row, mapping.answer)
            answer_formats[classify_answer_format(raw_answer)] += 1
            try:
                answer = normalize_answer(raw_answer)
            except DataFormatError as exc:
                stats.invalid_answer += 1
                _record_error(strict=strict, stats=stats, examples=examples, max_examples=max_error_examples,
                              source_path=source_path, row_number=row_number, reason=str(exc), raw_answer=raw_answer)
                continue

            answer_labels = list(answer)
            option_map = {item["label"]: item["text"] for item in options}
            missing = [label for label in answer_labels if label not in option_map]
            if missing:
                stats.answer_not_in_options += 1
                _record_error(strict=strict, stats=stats, examples=examples, max_examples=max_error_examples,
                              source_path=source_path, row_number=row_number,
                              reason=f"答案标签 {missing} 不存在于实际选项中。", raw_answer=raw_answer)
                continue

            digest = question_digest(stem, options)
            if digest in hashes:
                stats.within_split_duplicates += 1
                if drop_within_split_duplicates:
                    continue
            hashes.add(digest)
            raw_id = get_cell(row, mapping.record_id)
            record_id = raw_id or f"cmexam-{split}-{row_number - 1:06d}"
            answer_texts = [option_map[label] for label in answer_labels]
            records.append(
                {
                    "id": record_id,
                    "split": split,
                    "question": format_question(stem, options),
                    "stem": stem,
                    "options": options,
                    "answer": answer,
                    "answer_labels": answer_labels,
                    "answer_text": "；".join(answer_texts),
                    "answer_texts": answer_texts,
                    "is_multiple_choice": len(answer_labels) > 1,
                    "explanation": get_cell(row, mapping.explanation),
                    "metadata": build_metadata(row, mapping),
                }
            )
            label_counts[str(len(answer_labels))] += 1
            if len(answer_labels) == 1:
                stats.single_answer_records += 1
            else:
                stats.multiple_answer_records += 1

    stats.written_records = len(records)
    stats.answer_label_count_distribution = dict(sorted(label_counts.items()))
    stats.raw_answer_formats = dict(sorted(answer_formats.items()))
    return records, hashes, stats, examples


def validate_input_files(raw_dir: Path) -> dict[str, Path]:
    """检查三个官方输入文件。"""

    paths = {
        "train": raw_dir / "train.csv",
        "validation": raw_dir / "val.csv",
        "test": raw_dir / "test_with_annotations.csv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少以下 CMExam 原始文件：\n" + "\n".join(f"  - {p}" for p in missing))
    return paths


def prepare_output_paths(output_dir: Path, *, overwrite: bool) -> dict[str, Path]:
    """构造输出路径并实施默认禁止覆盖策略。"""

    paths = {
        "official_train": output_dir / "official/train/train.jsonl",
        "official_validation": output_dir / "official/validation/validation.jsonl",
        "official_test": output_dir / "official/test/test.jsonl",
        "decontaminated_train": output_dir / "decontaminated/train/train.jsonl",
        "report": output_dir / "processing_report.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "以下输出文件已经存在：\n" + "\n".join(f"  - {path}" for path in existing)
            + "\n如需重新生成，请增加 --overwrite。"
        )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    """先写同目录临时文件，成功后原子替换 JSONL。"""

    temp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                json.dump(record, file, ensure_ascii=False)
                file.write("\n")
                count += 1
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return count


def write_json_atomic(path: Path, content: Mapping[str, object]) -> None:
    """原子写入 JSON 报告。"""

    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(content, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _count_overlap(records: Sequence[Mapping[str, object]], other_hashes: set[str]) -> int:
    """统计一个划分中与另一划分重叠的记录数。"""

    return sum(question_digest(str(item["stem"]), item["options"]) in other_hashes for item in records)


def main(argv: Sequence[str] | None = None) -> int:
    """执行 CMExam 转换，成功返回 0。"""

    args = parse_args(argv)
    if args.max_error_examples < 0:
        raise ValueError("--max_error_examples 不能小于 0。")
    set_csv_field_size_limit()
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    input_paths = validate_input_files(raw_dir)
    output_paths = prepare_output_paths(output_dir, overwrite=args.overwrite)

    processed: dict[str, tuple[list[dict[str, object]], set[str], SplitStats, list[dict[str, object]]]] = {}
    for split in ("train", "validation", "test"):
        processed[split] = process_split(
            source_path=input_paths[split], split=split, strict=args.strict,
            max_error_examples=args.max_error_examples,
            drop_within_split_duplicates=args.drop_within_split_duplicates,
        )
    train_records, train_hashes, train_stats, train_errors = processed["train"]
    validation_records, validation_hashes, validation_stats, validation_errors = processed["validation"]
    test_records, test_hashes, test_stats, test_errors = processed["test"]

    evaluation_hashes = validation_hashes | test_hashes
    decontaminated_train = [
        item for item in train_records
        if question_digest(str(item["stem"]), item["options"]) not in evaluation_hashes
    ]
    overlaps = {
        "train_validation": _count_overlap(train_records, validation_hashes),
        "train_test": _count_overlap(train_records, test_hashes),
        "validation_test": _count_overlap(validation_records, test_hashes),
    }
    counts = {
        "official_train": write_jsonl_atomic(output_paths["official_train"], train_records),
        "official_validation": write_jsonl_atomic(output_paths["official_validation"], validation_records),
        "official_test": write_jsonl_atomic(output_paths["official_test"], test_records),
        "decontaminated_train": write_jsonl_atomic(output_paths["decontaminated_train"], decontaminated_train),
    }

    total_label_distribution = Counter[str]()
    total_answer_formats = Counter[str]()
    for stats in (train_stats, validation_stats, test_stats):
        total_label_distribution.update(stats.answer_label_count_distribution)
        total_answer_formats.update(stats.raw_answer_formats)
    report: dict[str, object] = {
        "configuration": {
            "raw_dir": str(raw_dir), "output_dir": str(output_dir), "strict": args.strict,
            "max_error_examples": args.max_error_examples, "overwrite": args.overwrite,
            "drop_within_split_duplicates": args.drop_within_split_duplicates,
        },
        "input_files": {key: str(path) for key, path in input_paths.items()},
        "official_splits": {
            "train": asdict(train_stats), "validation": asdict(validation_stats), "test": asdict(test_stats)
        },
        "cross_split_overlaps": overlaps,
        "official_output_counts": {
            "train": counts["official_train"], "validation": counts["official_validation"],
            "test": counts["official_test"],
        },
        "decontaminated_train_output_count": counts["decontaminated_train"],
        "decontaminated_train_removed_count": len(train_records) - len(decontaminated_train),
        "answer_label_count_distribution": dict(sorted(total_label_distribution.items())),
        "raw_answer_format_counts": dict(sorted(total_answer_formats.items())),
        "invalid_record_examples": {
            "train": train_errors, "validation": validation_errors, "test": test_errors,
        },
        "output_files": {key: str(path) for key, path in output_paths.items()},
    }
    write_json_atomic(output_paths["report"], report)

    print("CMExam 数据处理完成：")
    print(f"  official/train：{counts['official_train']:,} 条")
    print(f"  official/validation：{counts['official_validation']:,} 条")
    print(f"  official/test：{counts['official_test']:,} 条")
    print(f"  decontaminated/train：{counts['decontaminated_train']:,} 条")
    print(f"  处理报告：{output_paths['report']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (DataFormatError, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
