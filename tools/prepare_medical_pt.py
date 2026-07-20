#输入pretrain中的4个json文件，完成格式化转换。输出到processed。
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 shibing624/medical 的预训练数据转换为 MedicalGPT 可读取的 JSONL 格式。

原始输入文件：
    raw_dir/
    └── pretrain/
        ├── train_encyclopedia.json
        ├── medical_book_zh.json
        ├── valid_encyclopedia.json
        └── test_encyclopedia.json

处理后输出文件：
    output_dir/
    ├── train/
    │   └── train.jsonl
    ├── validation/
    │   └── validation.jsonl
    ├── test/
    │   └── test.jsonl
    └── processing_report.json

每行输出格式：
    {"text": "医疗领域文本"}

默认处理策略：
1. 保留全部合法、非重复的医学教材数据；
2. 从医疗百科训练集中使用蓄水池采样抽取 20000 条；
3. 保留官方验证集和测试集；
4. 删除空文本、过短文本和精确重复文本；
5. 避免训练集与验证集、测试集发生精确文本重叠；
6. 固定随机种子，保证抽样结果可以复现。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class SourceStats:
    """保存单个原始数据文件的处理统计信息。"""

    source: str
    total_lines: int = 0
    clean_records: int = 0
    skipped_blank_lines: int = 0
    skipped_empty_text: int = 0
    skipped_short_text: int = 0
    skipped_duplicate: int = 0
    skipped_overlap: int = 0


class DataFormatError(ValueError):
    """原始数据格式不符合预期时抛出的异常。"""


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="处理 MedicalGPT 的医疗领域 PT 数据。"
    )

    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=(
            Path.home()
            / "datasets"
            / "medicalgpt"
            / "raw"
            / "shibing624_medical"
        ),
        help="原始 shibing624/medical 数据集根目录。",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=(
            Path.home()
            / "datasets"
            / "medicalgpt"
            / "processed"
            / "pt"
        ),
        help="处理后 PT 数据的输出目录。",
    )

    parser.add_argument(
        "--encyclopedia_samples",
        type=int,
        default=20_000,
        help=(
            "从 train_encyclopedia.json 中抽取的样本数量，"
            "默认 20000。"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42。",
    )

    parser.add_argument(
        "--min_chars",
        type=int,
        default=20,
        help=(
            "文本去除首尾空白后的最少字符数。"
            "不足该长度的文本会被过滤，默认 20。"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的处理结果。",
    )

    return parser.parse_args()


def normalize_text(text: str) -> str:
    """
    对文本执行保守清洗。

    只完成以下操作：
    1. 将不同系统的换行符统一为 \\n；
    2. 删除 NUL 字符；
    3. 删除文本首尾空白。

    不随意压缩正文内部空格，避免破坏医学文本结构。
    """

    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\x00", "")
        .strip()
    )


def text_digest(text: str) -> str:
    """
    计算规范化文本的SHA-256摘要。

    摘要用于精确去重，避免在内存中保存多份完整文本。
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_clean_texts(
    path: Path,
    *,
    min_chars: int,
    stats: SourceStats,
) -> Iterator[str]:
    """
    逐行读取原始文件，并产出清洗后的text字段。

    原始文件虽然扩展名是.json，但实际是一行一个JSON对象，
    因此必须逐行读取，不能使用json.load()一次性加载整个文件。
    """

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stats.total_lines += 1

            if not line.strip():
                stats.skipped_blank_lines += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataFormatError(
                    f"{path} 第 {line_no} 行不是合法JSON：{exc}"
                ) from exc

            if not isinstance(record, dict):
                raise DataFormatError(
                    f"{path} 第 {line_no} 行应为JSON对象，"
                    f"实际类型为 {type(record).__name__}。"
                )

            if "text" not in record:
                raise DataFormatError(
                    f"{path} 第 {line_no} 行缺少必需字段 'text'。"
                )

            raw_text = record["text"]

            if not isinstance(raw_text, str):
                raise DataFormatError(
                    f"{path} 第 {line_no} 行的 'text' 必须是字符串，"
                    f"实际类型为 {type(raw_text).__name__}。"
                )

            text = normalize_text(raw_text)

            if not text:
                stats.skipped_empty_text += 1
                continue

            if len(text) < min_chars:
                stats.skipped_short_text += 1
                continue

            stats.clean_records += 1
            yield text


def load_unique_records(
    path: Path,
    *,
    min_chars: int,
    blocked_hashes: set[str] | None = None,
) -> tuple[list[str], set[str], SourceStats]:
    """
    读取较小的数据文件，并进行精确去重。

    blocked_hashes表示禁止出现的文本摘要，例如：
    - 测试集不得出现验证集中的文本；
    - 训练集不得出现验证集和测试集中的文本。
    """

    stats = SourceStats(source=str(path))
    records: list[str] = []
    hashes: set[str] = set()

    blocked = (
        blocked_hashes
        if blocked_hashes is not None
        else set()
    )

    for text in iter_clean_texts(
        path,
        min_chars=min_chars,
        stats=stats,
    ):
        digest = text_digest(text)

        if digest in blocked:
            stats.skipped_overlap += 1
            continue

        if digest in hashes:
            stats.skipped_duplicate += 1
            continue

        hashes.add(digest)
        records.append(text)

    return records, hashes, stats


def reservoir_sample_unique(
    path: Path,
    *,
    sample_size: int,
    seed: int,
    min_chars: int,
    blocked_hashes: set[str],
) -> tuple[list[str], SourceStats, int]:
    """
    使用蓄水池采样从大型百科文件中等概率抽取固定数量的样本。

    优点：
    - 不需要一次性把36万条百科数据全部载入内存；
    - 内存中只保存最终需要的sample_size条文本；
    - 固定随机种子后，抽样结果可以复现。
    """

    stats = SourceStats(source=str(path))
    rng = random.Random(seed)

    reservoir: list[str] = []
    seen_hashes: set[str] = set()
    eligible_count = 0

    for text in iter_clean_texts(
        path,
        min_chars=min_chars,
        stats=stats,
    ):
        digest = text_digest(text)

        # 排除与验证集、测试集和医学教材重复的文本。
        if digest in blocked_hashes:
            stats.skipped_overlap += 1
            continue

        # 排除百科训练集内部的重复文本。
        if digest in seen_hashes:
            stats.skipped_duplicate += 1
            continue

        seen_hashes.add(digest)
        eligible_count += 1

        if len(reservoir) < sample_size:
            reservoir.append(text)
            continue

        # Algorithm R：
        # 第eligible_count个合格样本，以sample_size/eligible_count
        # 的概率进入蓄水池，保证所有合格样本被等概率抽中。
        index = rng.randrange(eligible_count)

        if index < sample_size:
            reservoir[index] = text

    if eligible_count < sample_size:
        raise ValueError(
            f"{path} 清洗和去重后只有 {eligible_count} 条可用样本，"
            f"不足以抽取 {sample_size} 条。"
        )

    return reservoir, stats, eligible_count


def validate_input_files(raw_dir: Path) -> dict[str, Path]:
    """检查四个预期的原始PT数据文件是否全部存在。"""

    pretrain_dir = raw_dir / "pretrain"

    input_files = {
        "encyclopedia_train": (
            pretrain_dir / "train_encyclopedia.json"
        ),
        "medical_book": (
            pretrain_dir / "medical_book_zh.json"
        ),
        "validation": (
            pretrain_dir / "valid_encyclopedia.json"
        ),
        "test": (
            pretrain_dir / "test_encyclopedia.json"
        ),
    }

    missing_files = [
        path
        for path in input_files.values()
        if not path.is_file()
    ]

    if missing_files:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "缺少以下原始数据文件：\n"
            f"{formatted}"
        )

    return input_files


def prepare_output_paths(
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, Path]:
    """
    创建输出目录。

    默认禁止覆盖已经存在的处理结果，避免误删此前的实验数据。
    """

    output_paths = {
        "train": (
            output_dir
            / "train"
            / "train.jsonl"
        ),
        "validation": (
            output_dir
            / "validation"
            / "validation.jsonl"
        ),
        "test": (
            output_dir
            / "test"
            / "test.jsonl"
        ),
        "report": (
            output_dir
            / "processing_report.json"
        ),
    }

    existing_files = [
        path
        for path in output_paths.values()
        if path.exists()
    ]

    if existing_files and not overwrite:
        formatted = "\n".join(
            f"  - {path}"
            for path in existing_files
        )

        raise FileExistsError(
            "以下输出文件已经存在：\n"
            f"{formatted}\n"
            "如确认需要覆盖，请增加 --overwrite 参数。"
        )

    output_paths["train"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_paths["validation"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_paths["test"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_paths


def write_jsonl_atomic(
    path: Path,
    texts: Iterable[str],
) -> int:
    """
    以原子方式写入JSONL文件。

    先写入临时文件，全部成功后再替换正式文件。
    如果程序中途失败，不会留下内容不完整的正式训练文件。
    """

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )
    count = 0

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            for text in texts:
                json.dump(
                    {"text": text},
                    file,
                    ensure_ascii=False,
                )
                file.write("\n")
                count += 1

        temp_path.replace(path)

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return count


def write_report_atomic(
    path: Path,
    report: dict,
) -> None:
    """以原子方式写入数据处理报告。"""

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

        temp_path.replace(path)

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    """执行PT数据处理流程。"""

    args = parse_args()

    if args.encyclopedia_samples <= 0:
        raise ValueError(
            "--encyclopedia_samples必须大于0。"
        )

    if args.min_chars <= 0:
        raise ValueError(
            "--min_chars必须大于0。"
        )

    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    input_files = validate_input_files(raw_dir)

    output_paths = prepare_output_paths(
        output_dir,
        overwrite=args.overwrite,
    )

    print("=" * 72)
    print("开始处理MedicalGPT预训练数据")
    print(f"原始数据目录：{raw_dir}")
    print(f"输出数据目录：{output_dir}")
    print(
        "百科抽样数量："
        f"{args.encyclopedia_samples:,}"
    )
    print(f"最少字符数量：{args.min_chars}")
    print(f"随机种子：{args.seed}")
    print("=" * 72)

    # 先处理验证集。
    validation_records, validation_hashes, validation_stats = (
        load_unique_records(
            input_files["validation"],
            min_chars=args.min_chars,
        )
    )

    # 测试集不得与验证集发生精确重复。
    test_records, test_hashes, test_stats = (
        load_unique_records(
            input_files["test"],
            min_chars=args.min_chars,
            blocked_hashes=validation_hashes,
        )
    )

    evaluation_hashes = (
        validation_hashes
        | test_hashes
    )

    # 医学教材全部保留，但排除与验证集、测试集重复的文本。
    book_records, book_hashes, book_stats = (
        load_unique_records(
            input_files["medical_book"],
            min_chars=args.min_chars,
            blocked_hashes=evaluation_hashes,
        )
    )

    # 从医疗百科训练集中等概率抽取指定数量的样本。
    encyclopedia_records, encyclopedia_stats, eligible_count = (
        reservoir_sample_unique(
            input_files["encyclopedia_train"],
            sample_size=args.encyclopedia_samples,
            seed=args.seed,
            min_chars=args.min_chars,
            blocked_hashes=(
                evaluation_hashes
                | book_hashes
            ),
        )
    )

    # 将医学教材和医疗百科样本混合，避免按来源整块排列。
    train_records = (
        book_records
        + encyclopedia_records
    )

    random.Random(args.seed).shuffle(
        train_records
    )

    train_count = write_jsonl_atomic(
        output_paths["train"],
        train_records,
    )

    validation_count = write_jsonl_atomic(
        output_paths["validation"],
        validation_records,
    )

    test_count = write_jsonl_atomic(
        output_paths["test"],
        test_records,
    )

    report = {
        "configuration": {
            "raw_dir": str(raw_dir),
            "output_dir": str(output_dir),
            "encyclopedia_samples": (
                args.encyclopedia_samples
            ),
            "seed": args.seed,
            "min_chars": args.min_chars,
        },
        "input_files": {
            key: str(value)
            for key, value in input_files.items()
        },
        "source_statistics": {
            "encyclopedia_train": asdict(
                encyclopedia_stats
            ),
            "medical_book": asdict(
                book_stats
            ),
            "validation": asdict(
                validation_stats
            ),
            "test": asdict(
                test_stats
            ),
        },
        "sampling": {
            "encyclopedia_eligible_unique_records": (
                eligible_count
            ),
            "encyclopedia_selected_records": len(
                encyclopedia_records
            ),
            "medical_book_selected_records": len(
                book_records
            ),
        },
        "outputs": {
            "train": {
                "path": str(
                    output_paths["train"]
                ),
                "records": train_count,
            },
            "validation": {
                "path": str(
                    output_paths["validation"]
                ),
                "records": validation_count,
            },
            "test": {
                "path": str(
                    output_paths["test"]
                ),
                "records": test_count,
            },
        },
    }

    write_report_atomic(
        output_paths["report"],
        report,
    )

    print("\n处理完成：")
    print(f"  训练集：{train_count:,}条")
    print(
        "    - 医学教材："
        f"{len(book_records):,}条"
    )
    print(
        "    - 医疗百科："
        f"{len(encyclopedia_records):,}条"
    )
    print(
        f"  验证集：{validation_count:,}条"
    )
    print(
        f"  测试集：{test_count:,}条"
    )
    print(
        "  处理报告："
        f"{output_paths['report']}"
    )

    print(
        "\n注意：数据生成成功不代表可以直接开始正式训练，"
        "还需要检查行数、样本内容和token长度分布。"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except (
        DataFormatError,
        FileNotFoundError,
        FileExistsError,
        ValueError,
    ) as exc:
        print(
            f"错误：{exc}",
            file=sys.stderr,
        )
        sys.exit(1)