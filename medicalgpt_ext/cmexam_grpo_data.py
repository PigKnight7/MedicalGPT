"""CMExam 去污染训练集的 GRPO 数据读取、校验、抽样与转换。"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset

from medicalgpt_ext.cmexam_utils import (
    format_answer_labels,
    normalize_answer_labels,
    validate_options,
)


DEFAULT_CMEXAM_GRPO_TRAIN_FILE = (
    Path.home() / "datasets/medicalgpt/processed/cmexam/decontaminated/train/train.jsonl"
)
CMEXAM_GRPO_SYSTEM_PROMPT = (
    "你是一名医学考试答题助手。请根据题目选择所有正确选项。"
    "单选题只输出一个字母，多选题按A到E顺序输出全部字母。"
    "最终答案必须严格放在<answer></answer>标签中。"
    "例如：<answer>A</answer>或<answer>AE</answer>。"
    "不要输出解释、分析过程、思维过程或其他文字。"
)


class CMExamGRPODataError(ValueError):
    """CMExam GRPO 数据或配置不符合固定实验协议。"""


@dataclass(frozen=True)
class CMExamGRPODataConfig:
    """CMExam GRPO 数据准备的不可变配置。"""

    train_file: Path = DEFAULT_CMEXAM_GRPO_TRAIN_FILE
    max_samples: int | None = None
    seed: int = 42
    prompt_style: str = "answer_only"
    shuffle_before_select: bool = True
    require_decontaminated_train: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.train_file, (str, Path)):
            raise TypeError("train_file 必须是 pathlib.Path 或路径字符串。")
        object.__setattr__(self, "train_file", Path(self.train_file).expanduser())
        if self.prompt_style != "answer_only":
            raise CMExamGRPODataError("prompt_style 当前只支持 answer_only。")
        if self.max_samples is not None and (
            isinstance(self.max_samples, bool)
            or not isinstance(self.max_samples, int)
            or self.max_samples <= 0
        ):
            raise CMExamGRPODataError("max_samples 必须是 None 或正整数。")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed 必须是整数。")
        if not isinstance(self.shuffle_before_select, bool):
            raise TypeError("shuffle_before_select 必须是布尔值。")
        if not isinstance(self.require_decontaminated_train, bool):
            raise TypeError("require_decontaminated_train 必须是布尔值。")


@dataclass(frozen=True)
class CMExamGRPODataSummary:
    """不含题目正文和答案文本的可复现数据摘要。"""

    source_file: str
    raw_records: int
    valid_records: int
    selected_records: int
    blank_lines: int
    single_choice_records: int
    multiple_choice_records: int
    answer_label_count_distribution: dict[str, int]
    selected_ids: tuple[str, ...]
    seed: int
    max_samples: int | None
    shuffled_before_select: bool
    requested_samples_exceeded_available: bool

    def to_dict(self) -> dict[str, object]:
        """转换为仅含普通 JSON 可序列化类型的字典。"""

        result = asdict(self)
        result["selected_ids"] = list(self.selected_ids)
        return result


def _validate_train_path(path: Path) -> None:
    parts = tuple(part.casefold() for part in path.parts)
    forbidden = {"validation", "test"}
    if forbidden.intersection(parts):
        raise CMExamGRPODataError(
            "validation 和 test 不能用于 GRPO 训练；只能使用去污染 train 数据。"
        )
    has_required_pair = any(
        parts[index : index + 2] == ("decontaminated", "train")
        for index in range(max(0, len(parts) - 1))
    )
    if not has_required_pair or path.name.casefold() != "train.jsonl":
        raise CMExamGRPODataError(
            "GRPO 训练文件必须位于连续目录 decontaminated/train 下且名为 train.jsonl；"
            "validation 和 test 不能用于 GRPO 训练。"
        )


def validate_cmexam_grpo_record(
    record: Mapping[str, object], *, line_number: int | None = None
) -> dict[str, object]:
    """严格校验并复制一条记录，不修改调用方对象。"""

    where = f"第 {line_number} 行" if line_number is not None else "记录"
    if not isinstance(record, Mapping):
        raise CMExamGRPODataError(f"{where}必须是 JSON 对象。")
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise CMExamGRPODataError(f"{where}缺少非空字符串 id。")
    context = f"{where}（id={record_id!r}）"

    split = record.get("split")
    if split is not None and (not isinstance(split, str) or split.casefold() != "train"):
        raise CMExamGRPODataError(
            f"{context} split 必须是 train；validation 和 test 不能用于 GRPO 训练。"
        )
    question = record.get("stem", record.get("question"))
    if not isinstance(question, str) or not question.strip():
        raise CMExamGRPODataError(f"{context}缺少非空 question（或等价 stem）字段。")
    if "options" not in record:
        raise CMExamGRPODataError(f"{context}缺少 options。")
    try:
        options = validate_options(record["options"])
    except ValueError as exc:
        raise CMExamGRPODataError(f"{context} options 非法：{exc}") from exc
    if "answer_labels" not in record:
        raise CMExamGRPODataError(f"{context}缺少 answer_labels。")
    raw_labels = record["answer_labels"]
    if not isinstance(raw_labels, (str, list, tuple)):
        raise CMExamGRPODataError(f"{context} answer_labels 必须是字符串或字符串序列。")
    if not isinstance(raw_labels, str) and not all(isinstance(item, str) for item in raw_labels):
        raise CMExamGRPODataError(f"{context} answer_labels 必须只包含字符串。")
    labels = normalize_answer_labels(raw_labels)
    if labels is None:
        raise CMExamGRPODataError(f"{context} answer_labels 非法，标签只能为 A-E。")
    option_labels = {item["label"] for item in options}
    missing = [label for label in labels if label not in option_labels]
    if missing:
        raise CMExamGRPODataError(f"{context}答案标签 {missing} 不存在于 options 中。")
    if "is_multiple_choice" not in record:
        raise CMExamGRPODataError(f"{context}缺少 is_multiple_choice。")
    multiple = record["is_multiple_choice"]
    expected_multiple = len(labels) >= 2
    if not isinstance(multiple, bool) or multiple is not expected_multiple:
        raise CMExamGRPODataError(
            f"{context} is_multiple_choice 与 answer_labels 数量不一致。"
        )
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CMExamGRPODataError(f"{context} metadata 必须是对象。")

    normalized = copy.deepcopy(dict(record))
    normalized["id"] = record_id.strip()
    normalized["options"] = [dict(item) for item in options]
    normalized["answer_labels"] = list(labels)
    normalized["answer"] = format_answer_labels(labels)
    normalized["is_multiple_choice"] = expected_multiple
    normalized["metadata"] = copy.deepcopy(metadata)
    return normalized


def read_cmexam_grpo_jsonl(
    path: str | Path, *, require_decontaminated_train: bool = True
) -> tuple[list[dict[str, object]], int]:
    """逐行读取 JSONL，返回规范化记录与跳过的空白行数。"""

    source = Path(path).expanduser()
    if require_decontaminated_train:
        _validate_train_path(source)
    if not source.exists():
        raise FileNotFoundError(f"CMExam GRPO 数据文件不存在：{source}")
    if not source.is_file():
        raise CMExamGRPODataError(f"CMExam GRPO 数据路径不是文件：{source}")
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    blank_lines = 0
    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                blank_lines += 1
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CMExamGRPODataError(
                    f"{source.name} 第 {line_number} 行 JSON 非法：{exc.msg}。"
                ) from exc
            if not isinstance(raw, dict):
                raise CMExamGRPODataError(f"{source.name} 第 {line_number} 行必须是 JSON 对象。")
            record = validate_cmexam_grpo_record(raw, line_number=line_number)
            record_id = str(record["id"])
            if record_id in seen_ids:
                raise CMExamGRPODataError(
                    f"{source.name} 第 {line_number} 行存在重复 id：{record_id!r}。"
                )
            seen_ids.add(record_id)
            records.append(record)
    if not records:
        raise CMExamGRPODataError(f"CMExam GRPO 数据文件为空或只包含空白行：{source}")
    return records, blank_lines


def build_cmexam_question_text(record: Mapping[str, object]) -> str:
    """仅用题干和结构化选项构造用户消息，不泄露答案或 metadata。"""

    stem = record.get("stem", record.get("question"))
    if not isinstance(stem, str) or not stem.strip():
        raise CMExamGRPODataError("题干 question（或等价 stem）不能为空。")
    try:
        options = validate_options(record.get("options"))
    except ValueError as exc:
        raise CMExamGRPODataError(f"options 非法：{exc}") from exc
    return "\n".join(
        [stem.strip(), *(f"{item['label']}. {item['text'].strip()}" for item in options)]
    )


def build_cmexam_grpo_prompt(
    record: Mapping[str, object], *, prompt_style: str = "answer_only"
) -> list[dict[str, str]]:
    """构造由 TRL 1.8 在 Trainer 内应用 chat template 的对话 prompt。"""

    if prompt_style != "answer_only":
        raise CMExamGRPODataError("prompt_style 当前只支持 answer_only。")
    return [
        {"role": "system", "content": CMEXAM_GRPO_SYSTEM_PROMPT},
        {"role": "user", "content": build_cmexam_question_text(record)},
    ]


def prepare_cmexam_grpo_examples(
    config: CMExamGRPODataConfig,
) -> tuple[list[dict[str, object]], CMExamGRPODataSummary]:
    """读取、校验、确定性抽样并构造可序列化的 TRL 样本。"""

    records, blank_lines = read_cmexam_grpo_jsonl(
        config.train_file,
        require_decontaminated_train=config.require_decontaminated_train,
    )
    indexed = list(enumerate(records))
    limit = config.max_samples
    did_shuffle_select = bool(
        limit is not None and limit < len(indexed) and config.shuffle_before_select
    )
    if limit is not None and limit < len(indexed):
        if config.shuffle_before_select:
            indexed = sorted(random.Random(config.seed).sample(indexed, limit), key=lambda item: item[0])
        else:
            indexed = indexed[:limit]
    selected = [record for _, record in indexed]
    examples: list[dict[str, object]] = []
    for record in selected:
        examples.append(
            {
                "id": str(record["id"]),
                "prompt": build_cmexam_grpo_prompt(record, prompt_style=config.prompt_style),
                "answer": str(record["answer"]),
                "answer_labels": list(record["answer_labels"]),
                "is_multiple_choice": bool(record["is_multiple_choice"]),
                "metadata": copy.deepcopy(record["metadata"]),
                "question": build_cmexam_question_text(record),
                "options": copy.deepcopy(record["options"]),
            }
        )
    summary = summarize_cmexam_grpo_examples(
        examples,
        source_file=config.train_file,
        raw_records=len(records),
        valid_records=len(records),
        blank_lines=blank_lines,
        seed=config.seed,
        max_samples=config.max_samples,
        shuffled_before_select=did_shuffle_select,
        requested_samples_exceeded_available=(
            config.max_samples is not None and config.max_samples > len(records)
        ),
    )
    return examples, summary


def summarize_cmexam_grpo_examples(
    examples: Sequence[Mapping[str, object]],
    *,
    source_file: str | Path,
    raw_records: int | None = None,
    valid_records: int | None = None,
    blank_lines: int = 0,
    seed: int = 42,
    max_samples: int | None = None,
    shuffled_before_select: bool = False,
    requested_samples_exceeded_available: bool = False,
) -> CMExamGRPODataSummary:
    """汇总处理后样本，统计口径均针对最终选中记录。"""

    count = len(examples)
    multiple = sum(bool(item["is_multiple_choice"]) for item in examples)
    distribution = Counter(str(len(item["answer_labels"])) for item in examples)
    return CMExamGRPODataSummary(
        source_file=str(Path(source_file).expanduser()),
        raw_records=count if raw_records is None else raw_records,
        valid_records=count if valid_records is None else valid_records,
        selected_records=count,
        blank_lines=blank_lines,
        single_choice_records=count - multiple,
        multiple_choice_records=multiple,
        answer_label_count_distribution=dict(sorted(distribution.items())),
        selected_ids=tuple(str(item["id"]) for item in examples),
        seed=seed,
        max_samples=max_samples,
        shuffled_before_select=shuffled_before_select,
        requested_samples_exceeded_available=requested_samples_exceeded_available,
    )


def build_cmexam_grpo_dataset(
    config: CMExamGRPODataConfig,
) -> Dataset:
    """构造无需 Hub 访问、可直接交给 TRL GRPOTrainer 的 Dataset。"""

    examples, _ = prepare_cmexam_grpo_examples(config)
    return Dataset.from_list(examples)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验并预览 CMExam GRPO 去污染训练数据。")
    parser.add_argument("--train_file", type=Path, default=DEFAULT_CMEXAM_GRPO_TRAIN_FILE)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--allow_non_decontaminated_train", action="store_true")
    parser.add_argument("--show_examples", type=int, nargs="?", const=3, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """只执行本地数据校验、Dataset 构造和安全预览。"""

    args = parse_args(argv)
    if args.show_examples < 0 or args.show_examples > 3:
        raise CMExamGRPODataError("--show_examples 必须在 0 到 3 之间。")
    config = CMExamGRPODataConfig(
        train_file=args.train_file,
        max_samples=args.max_samples,
        seed=args.seed,
        shuffle_before_select=not args.no_shuffle,
        require_decontaminated_train=not args.allow_non_decontaminated_train,
    )
    examples, summary = prepare_cmexam_grpo_examples(config)
    dataset = Dataset.from_list(examples)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    for index in range(min(args.show_examples, len(dataset))):
        safe = {"id": dataset[index]["id"], "prompt": dataset[index]["prompt"]}
        print(f"\n--- Prompt 示例 {index + 1} ---")
        print(json.dumps(safe, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "CMEXAM_GRPO_SYSTEM_PROMPT",
    "CMExamGRPODataConfig",
    "CMExamGRPODataError",
    "CMExamGRPODataSummary",
    "build_cmexam_grpo_dataset",
    "build_cmexam_grpo_prompt",
    "build_cmexam_question_text",
    "prepare_cmexam_grpo_examples",
    "read_cmexam_grpo_jsonl",
    "summarize_cmexam_grpo_examples",
    "validate_cmexam_grpo_record",
]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CMExamGRPODataError, FileNotFoundError, OSError, TypeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
