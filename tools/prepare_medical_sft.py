#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""将 shibing624/medical 的微调数据转换为 MedicalGPT ShareGPT JSONL。

原始 ``.json`` 文件实际上是每行一个 JSON 对象。训练集使用蓄水池
抽样，处理过程不会将整个大文件载入内存。
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


@dataclass(frozen=True)
class SFTRecord:
    """一条规范化后的单轮 SFT 记录。"""

    user_text: str
    assistant_text: str


@dataclass
class SourceStats:
    """保存单个原始文件的处理统计。"""

    source: str
    total_lines: int = 0
    clean_records: int = 0
    skipped_blank_lines: int = 0
    skipped_empty_user: int = 0
    skipped_empty_assistant: int = 0
    skipped_short_record: int = 0
    skipped_duplicate: int = 0
    skipped_overlap: int = 0


class DataFormatError(ValueError):
    """原始数据格式不符合预期时抛出。"""


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="处理 MedicalGPT 的医疗 SFT 数据。")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path.home() / "datasets" / "medicalgpt" / "raw" / "shibing624_medical",
        help="原始 shibing624/medical 数据集根目录。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.home() / "datasets" / "medicalgpt" / "processed" / "sft",
        help="处理后 SFT 数据的输出目录。",
    )
    parser.add_argument(
        "--train_samples", type=int, default=30_000, help="训练集抽样数量，默认 30000。"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42。")
    parser.add_argument(
        "--min_user_chars", type=int, default=2, help="用户文本最少字符数，默认 2。"
    )
    parser.add_argument(
        "--min_assistant_chars", type=int, default=2, help="回答最少字符数，默认 2。"
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有处理结果。")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """统一换行符并删除首尾空白，不改动正文内容。"""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def record_digest(user_text: str, assistant_text: str) -> str:
    """计算用户文本和回答联合的 SHA-256 摘要。"""

    # JSON 数组编码可避免不同字段分割方式产生拼接歧义。
    payload = json.dumps(
        [user_text, assistant_text], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# 为调用方提供与 PT 脚本一致的摘要函数名。
text_digest = record_digest


def iter_clean_records(
    path: Path,
    *,
    min_user_chars: int,
    min_assistant_chars: int,
    stats: SourceStats,
) -> Iterator[SFTRecord]:
    """逐行读取、校验原始文件，并产出规范化后的合法记录。"""

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stats.total_lines += 1
            if not line.strip():
                stats.skipped_blank_lines += 1
                continue

            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataFormatError(
                    f"{path} 第 {line_no} 行不是合法JSON：{exc}"
                ) from exc

            if not isinstance(raw_record, dict):
                raise DataFormatError(
                    f"{path} 第 {line_no} 行应为JSON对象，"
                    f"实际类型为 {type(raw_record).__name__}。"
                )

            required_fields = ("instruction", "input", "output")
            for field in required_fields:
                if field not in raw_record:
                    raise DataFormatError(
                        f"{path} 第 {line_no} 行缺少必需字段 '{field}'。"
                    )
                if not isinstance(raw_record[field], str):
                    raise DataFormatError(
                        f"{path} 第 {line_no} 行的 '{field}' 必须是字符串，"
                        f"实际类型为 {type(raw_record[field]).__name__}。"
                    )

            instruction = normalize_text(raw_record["instruction"])
            input_text = normalize_text(raw_record["input"])
            assistant_text = normalize_text(raw_record["output"])
            user_text = instruction if not input_text else f"{instruction}\n{input_text}"

            if not user_text:
                stats.skipped_empty_user += 1
                continue
            if not assistant_text:
                stats.skipped_empty_assistant += 1
                continue
            if len(user_text) < min_user_chars or len(assistant_text) < min_assistant_chars:
                stats.skipped_short_record += 1
                continue

            stats.clean_records += 1
            yield SFTRecord(user_text=user_text, assistant_text=assistant_text)


def load_unique_records(
    path: Path,
    *,
    min_user_chars: int,
    min_assistant_chars: int,
    blocked_hashes: set[str] | None = None,
) -> tuple[list[SFTRecord], set[str], SourceStats]:
    """读取较小的数据集，并过滤内部重复及跨集合重叠。"""

    stats = SourceStats(source=str(path))
    records: list[SFTRecord] = []
    hashes: set[str] = set()
    blocked = blocked_hashes if blocked_hashes is not None else set()

    for record in iter_clean_records(
        path,
        min_user_chars=min_user_chars,
        min_assistant_chars=min_assistant_chars,
        stats=stats,
    ):
        digest = record_digest(record.user_text, record.assistant_text)
        if digest in blocked:
            stats.skipped_overlap += 1
            continue
        if digest in hashes:
            stats.skipped_duplicate += 1
            continue
        hashes.add(digest)
        records.append(record)

    return records, hashes, stats


def reservoir_sample_unique(
    path: Path,
    *,
    sample_size: int,
    seed: int,
    min_user_chars: int,
    min_assistant_chars: int,
    blocked_hashes: set[str],
) -> tuple[list[SFTRecord], SourceStats, int]:
    """流式去重后使用 Algorithm R 等概率抽取训练样本。"""

    stats = SourceStats(source=str(path))
    rng = random.Random(seed)
    reservoir: list[SFTRecord] = []
    seen_hashes: set[str] = set()
    eligible_count = 0

    for record in iter_clean_records(
        path,
        min_user_chars=min_user_chars,
        min_assistant_chars=min_assistant_chars,
        stats=stats,
    ):
        digest = record_digest(record.user_text, record.assistant_text)
        if digest in blocked_hashes:
            stats.skipped_overlap += 1
            continue
        if digest in seen_hashes:
            stats.skipped_duplicate += 1
            continue
        seen_hashes.add(digest)
        eligible_count += 1

        if len(reservoir) < sample_size:
            reservoir.append(record)
        else:
            index = rng.randrange(eligible_count)
            if index < sample_size:
                reservoir[index] = record

    if eligible_count < sample_size:
        raise ValueError(
            f"{path} 清洗和去重后只有 {eligible_count} 条可用训练样本，"
            f"不足以抽取 {sample_size} 条。"
        )
    return reservoir, stats, eligible_count


def validate_input_files(raw_dir: Path) -> dict[str, Path]:
    """检查三个预期的原始 SFT 文件。"""

    finetune_dir = raw_dir / "finetune"
    input_files = {
        "train": finetune_dir / "train_zh_0.json",
        "validation": finetune_dir / "valid_zh_0.json",
        "test": finetune_dir / "test_zh_0.json",
    }
    missing = [path for path in input_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少以下原始数据文件：\n" + "\n".join(f"  - {path}" for path in missing)
        )
    return input_files


def prepare_output_paths(output_dir: Path, *, overwrite: bool) -> dict[str, Path]:
    """检查覆盖策略并创建输出目录。"""

    paths = {
        "train": output_dir / "train" / "train.jsonl",
        "validation": output_dir / "validation" / "validation.jsonl",
        "test": output_dir / "test" / "test.jsonl",
        "report": output_dir / "processing_report.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "以下输出文件已经存在：\n"
            + "\n".join(f"  - {path}" for path in existing)
            + "\n如确认需要覆盖，请增加 --overwrite 参数。"
        )
    for key in ("train", "validation", "test"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _sharegpt_record(record: SFTRecord) -> dict[str, list[dict[str, str]]]:
    """构造严格的 MedicalGPT ShareGPT 对象。"""

    return {
        "conversations": [
            {"from": "human", "value": record.user_text},
            {"from": "gpt", "value": record.assistant_text},
        ]
    }


def write_jsonl_atomic(path: Path, records: Iterable[SFTRecord]) -> int:
    """先写同目录临时文件，成功后原子替换正式 JSONL。"""

    temp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                json.dump(_sharegpt_record(record), file, ensure_ascii=False)
                file.write("\n")
                count += 1
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return count


def write_report_atomic(path: Path, report: dict[str, object]) -> None:
    """原子写入处理报告。"""

    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    """执行 SFT 数据处理流程，成功时返回 0。"""

    args = parse_args()
    if args.train_samples <= 0:
        raise ValueError("--train_samples必须大于0。")
    if args.min_user_chars <= 0:
        raise ValueError("--min_user_chars必须大于0。")
    if args.min_assistant_chars <= 0:
        raise ValueError("--min_assistant_chars必须大于0。")

    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    input_files = validate_input_files(raw_dir)
    output_paths = prepare_output_paths(output_dir, overwrite=args.overwrite)

    validation_records, validation_hashes, validation_stats = load_unique_records(
        input_files["validation"],
        min_user_chars=args.min_user_chars,
        min_assistant_chars=args.min_assistant_chars,
    )
    test_records, test_hashes, test_stats = load_unique_records(
        input_files["test"],
        min_user_chars=args.min_user_chars,
        min_assistant_chars=args.min_assistant_chars,
        blocked_hashes=validation_hashes,
    )
    train_records, train_stats, eligible_count = reservoir_sample_unique(
        input_files["train"],
        sample_size=args.train_samples,
        seed=args.seed,
        min_user_chars=args.min_user_chars,
        min_assistant_chars=args.min_assistant_chars,
        blocked_hashes=validation_hashes | test_hashes,
    )

    train_count = write_jsonl_atomic(output_paths["train"], train_records)
    validation_count = write_jsonl_atomic(output_paths["validation"], validation_records)
    test_count = write_jsonl_atomic(output_paths["test"], test_records)
    report: dict[str, object] = {
        "configuration": {
            "raw_dir": str(raw_dir),
            "output_dir": str(output_dir),
            "train_samples": args.train_samples,
            "seed": args.seed,
            "min_user_chars": args.min_user_chars,
            "min_assistant_chars": args.min_assistant_chars,
            "overwrite": args.overwrite,
        },
        "input_files": {key: str(value) for key, value in input_files.items()},
        "source_statistics": {
            "train": asdict(train_stats),
            "validation": asdict(validation_stats),
            "test": asdict(test_stats),
        },
        "sampling": {
            "train_eligible_unique_records": eligible_count,
            "train_selected_records": len(train_records),
        },
        "outputs": {
            "train": {"path": str(output_paths["train"]), "records": train_count},
            "validation": {
                "path": str(output_paths["validation"]),
                "records": validation_count,
            },
            "test": {"path": str(output_paths["test"]), "records": test_count},
            "report": {"path": str(output_paths["report"])},
        },
    }
    write_report_atomic(output_paths["report"], report)

    print("医疗 SFT 数据处理完成：")
    print(f"  训练集：{train_count:,}条")
    print(f"  验证集：{validation_count:,}条")
    print(f"  测试集：{test_count:,}条")
    print(f"  处理报告：{output_paths['report']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (DataFormatError, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
