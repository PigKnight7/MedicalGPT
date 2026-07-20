# -*- coding: utf-8 -*-

"""
tools/prepare_medical_pt.py 的单元测试。

测试内容：
1. 文本规范化；
2. 原始JSONL读取和过滤；
3. 非法数据格式检测；
4. 精确去重及集合重叠过滤；
5. 蓄水池采样的数量、去重和可复现性；
6. 完整数据处理流程；
7. 默认禁止覆盖已有输出文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.prepare_medical_pt import (
    DataFormatError,
    SourceStats,
    iter_clean_texts,
    load_unique_records,
    main,
    normalize_text,
    reservoir_sample_unique,
    text_digest,
)


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    """测试辅助函数：将若干JSON对象写成JSONL文件。"""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in records:
            json.dump(
                record,
                file,
                ensure_ascii=False,
            )
            file.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    """测试辅助函数：读取JSONL文件。"""

    records: list[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


def create_mock_raw_dataset(
    raw_dir: Path,
) -> None:
    """
    创建一个小型模拟PT数据集。

    数据中故意包含：
    - 训练集内部重复；
    - 验证集与测试集重叠；
    - 教材与验证集重叠；
    - 百科与教材、测试集重叠。
    """

    pretrain_dir = raw_dir / "pretrain"

    write_jsonl(
        pretrain_dir
        / "valid_encyclopedia.json",
        [
            {"text": "验证集医疗文本一"},
            {"text": "验证集医疗文本二"},
        ],
    )

    write_jsonl(
        pretrain_dir
        / "test_encyclopedia.json",
        [
            {"text": "测试集医疗文本一"},
            # 与验证集重叠，应当被过滤。
            {"text": "验证集医疗文本一"},
            {"text": "测试集医疗文本二"},
        ],
    )

    write_jsonl(
        pretrain_dir
        / "medical_book_zh.json",
        [
            {"text": "医学教材有效文本一"},
            # 教材内部重复，应当被过滤。
            {"text": "医学教材有效文本一"},
            # 与验证集重叠，应当被过滤。
            {"text": "验证集医疗文本二"},
            {"text": "医学教材有效文本二"},
        ],
    )

    encyclopedia_records = [
        {
            "text": f"医疗百科有效训练文本{i}"
        }
        for i in range(10)
    ]

    encyclopedia_records.extend(
        [
            # 与教材重叠，应当被过滤。
            {"text": "医学教材有效文本一"},
            # 与测试集重叠，应当被过滤。
            {"text": "测试集医疗文本一"},
            # 百科内部重复，应当被过滤。
            {"text": "医疗百科有效训练文本0"},
        ]
    )

    write_jsonl(
        pretrain_dir
        / "train_encyclopedia.json",
        encyclopedia_records,
    )


def test_normalize_text() -> None:
    """测试换行符、NUL字符和首尾空白清理。"""

    raw_text = (
        "  第一行\r\n"
        "第二行\r"
        "\x00第三行  "
    )

    result = normalize_text(raw_text)

    assert result == (
        "第一行\n"
        "第二行\n"
        "第三行"
    )


def test_iter_clean_texts_filters_invalid_records(
    tmp_path: Path,
) -> None:
    """测试空行、空文本和过短文本过滤。"""

    data_path = tmp_path / "sample.json"

    with data_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n")
        file.write(
            json.dumps(
                {"text": "   "},
                ensure_ascii=False,
            )
            + "\n"
        )
        file.write(
            json.dumps(
                {"text": "太短"},
                ensure_ascii=False,
            )
            + "\n"
        )
        file.write(
            json.dumps(
                {"text": "这是一条长度足够的医疗文本"},
                ensure_ascii=False,
            )
            + "\n"
        )

    stats = SourceStats(
        source=str(data_path)
    )

    records = list(
        iter_clean_texts(
            data_path,
            min_chars=5,
            stats=stats,
        )
    )

    assert records == [
        "这是一条长度足够的医疗文本"
    ]

    assert stats.total_lines == 4
    assert stats.clean_records == 1
    assert stats.skipped_blank_lines == 1
    assert stats.skipped_empty_text == 1
    assert stats.skipped_short_text == 1


def test_iter_clean_texts_rejects_missing_text(
    tmp_path: Path,
) -> None:
    """缺少text字段时应当明确报错。"""

    data_path = tmp_path / "missing_text.json"

    write_jsonl(
        data_path,
        [
            {
                "content": "字段名称错误"
            }
        ],
    )

    stats = SourceStats(
        source=str(data_path)
    )

    with pytest.raises(
        DataFormatError,
        match="缺少必需字段 'text'",
    ):
        list(
            iter_clean_texts(
                data_path,
                min_chars=1,
                stats=stats,
            )
        )


def test_iter_clean_texts_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """非法JSON行应当触发DataFormatError。"""

    data_path = tmp_path / "invalid.json"

    data_path.write_text(
        '{"text": "合法文本"}\n'
        '{"text": 非法JSON}\n',
        encoding="utf-8",
    )

    stats = SourceStats(
        source=str(data_path)
    )

    with pytest.raises(
        DataFormatError,
        match="不是合法JSON",
    ):
        list(
            iter_clean_texts(
                data_path,
                min_chars=1,
                stats=stats,
            )
        )


def test_load_unique_records_filters_duplicates_and_overlap(
    tmp_path: Path,
) -> None:
    """测试文件内部去重和跨集合重叠过滤。"""

    data_path = tmp_path / "records.json"

    write_jsonl(
        data_path,
        [
            {"text": "有效医疗文本一"},
            {"text": "有效医疗文本一"},
            {"text": "禁止出现的医疗文本"},
            {"text": "有效医疗文本二"},
        ],
    )

    blocked_hashes = {
        text_digest(
            "禁止出现的医疗文本"
        )
    }

    records, hashes, stats = (
        load_unique_records(
            data_path,
            min_chars=1,
            blocked_hashes=blocked_hashes,
        )
    )

    assert records == [
        "有效医疗文本一",
        "有效医疗文本二",
    ]

    assert len(hashes) == 2
    assert stats.skipped_duplicate == 1
    assert stats.skipped_overlap == 1


def test_reservoir_sample_is_reproducible(
    tmp_path: Path,
) -> None:
    """固定随机种子时，蓄水池采样结果必须一致。"""

    data_path = tmp_path / "encyclopedia.json"

    write_jsonl(
        data_path,
        [
            {
                "text": f"医疗百科训练文本{i}"
            }
            for i in range(100)
        ],
    )

    sample_1, stats_1, eligible_1 = (
        reservoir_sample_unique(
            data_path,
            sample_size=10,
            seed=42,
            min_chars=1,
            blocked_hashes=set(),
        )
    )

    sample_2, stats_2, eligible_2 = (
        reservoir_sample_unique(
            data_path,
            sample_size=10,
            seed=42,
            min_chars=1,
            blocked_hashes=set(),
        )
    )

    assert sample_1 == sample_2
    assert len(sample_1) == 10
    assert len(set(sample_1)) == 10

    assert eligible_1 == 100
    assert eligible_2 == 100

    assert stats_1.skipped_duplicate == 0
    assert stats_2.skipped_duplicate == 0


def test_reservoir_sample_excludes_blocked_text(
    tmp_path: Path,
) -> None:
    """被blocked_hashes标记的样本不能进入采样结果。"""

    data_path = tmp_path / "encyclopedia.json"

    blocked_text = "禁止进入训练集的文本"

    records = [
        {"text": blocked_text},
    ]

    records.extend(
        {
            "text": f"正常医疗训练文本{i}"
        }
        for i in range(20)
    )

    write_jsonl(
        data_path,
        records,
    )

    sample, stats, eligible_count = (
        reservoir_sample_unique(
            data_path,
            sample_size=5,
            seed=42,
            min_chars=1,
            blocked_hashes={
                text_digest(blocked_text)
            },
        )
    )

    assert blocked_text not in sample
    assert stats.skipped_overlap == 1
    assert eligible_count == 20


def test_reservoir_sample_rejects_oversized_request(
    tmp_path: Path,
) -> None:
    """请求数量超过可用样本数时应当报错。"""

    data_path = tmp_path / "small.json"

    write_jsonl(
        data_path,
        [
            {"text": "医疗文本一"},
            {"text": "医疗文本二"},
        ],
    )

    with pytest.raises(
        ValueError,
        match="不足以抽取",
    ):
        reservoir_sample_unique(
            data_path,
            sample_size=3,
            seed=42,
            min_chars=1,
            blocked_hashes=set(),
        )


def test_main_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试完整数据处理流程。

    验证：
    - 输出文件全部生成；
    - 数据数量正确；
    - 输出格式正确；
    - 训练、验证和测试之间不存在精确重叠；
    - 处理报告内容正确。
    """

    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    create_mock_raw_dataset(raw_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_medical_pt.py",
            "--raw_dir",
            str(raw_dir),
            "--output_dir",
            str(output_dir),
            "--encyclopedia_samples",
            "3",
            "--min_chars",
            "1",
            "--seed",
            "42",
        ],
    )

    return_code = main()

    assert return_code == 0

    train_path = (
        output_dir
        / "train"
        / "train.jsonl"
    )

    validation_path = (
        output_dir
        / "validation"
        / "validation.jsonl"
    )

    test_path = (
        output_dir
        / "test"
        / "test.jsonl"
    )

    report_path = (
        output_dir
        / "processing_report.json"
    )

    assert train_path.is_file()
    assert validation_path.is_file()
    assert test_path.is_file()
    assert report_path.is_file()

    train_records = read_jsonl(train_path)
    validation_records = read_jsonl(
        validation_path
    )
    test_records = read_jsonl(test_path)

    # 两条教材数据 + 三条百科抽样数据。
    assert len(train_records) == 5

    # 验证集两条全部保留。
    assert len(validation_records) == 2

    # 测试集中有一条与验证集重叠，因此保留两条。
    assert len(test_records) == 2

    for record in (
        train_records
        + validation_records
        + test_records
    ):
        assert set(record.keys()) == {
            "text"
        }
        assert isinstance(
            record["text"],
            str,
        )
        assert record["text"]

    train_texts = {
        record["text"]
        for record in train_records
    }

    validation_texts = {
        record["text"]
        for record in validation_records
    }

    test_texts = {
        record["text"]
        for record in test_records
    }

    assert train_texts.isdisjoint(
        validation_texts
    )
    assert train_texts.isdisjoint(
        test_texts
    )
    assert validation_texts.isdisjoint(
        test_texts
    )

    with report_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    assert (
        report["configuration"][
            "encyclopedia_samples"
        ]
        == 3
    )

    assert (
        report["outputs"]["train"][
            "records"
        ]
        == 5
    )

    assert (
        report["outputs"]["validation"][
            "records"
        ]
        == 2
    )

    assert (
        report["outputs"]["test"][
            "records"
        ]
        == 2
    )


def test_main_refuses_to_overwrite_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未提供--overwrite时，不允许覆盖既有处理结果。"""

    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    create_mock_raw_dataset(raw_dir)

    arguments = [
        "prepare_medical_pt.py",
        "--raw_dir",
        str(raw_dir),
        "--output_dir",
        str(output_dir),
        "--encyclopedia_samples",
        "3",
        "--min_chars",
        "1",
        "--seed",
        "42",
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        arguments,
    )

    assert main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        arguments,
    )

    with pytest.raises(
        FileExistsError,
        match="已经存在",
    ):
        main()