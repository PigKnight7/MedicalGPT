"""tools.analyze_token_lengths 的离线单元测试。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import tools.analyze_token_lengths as analyzer
from tools.analyze_token_lengths import (
    DataFormatError,
    analyze_datasets,
    analyze_pt_file,
    analyze_sft_file,
    distribution_statistics,
    main,
    parse_args,
    percentile,
    run,
    summarize_pt,
    summarize_sft,
)


class FakeTokenizer:
    """按字符计数的 tokenizer，专用于无网络测试。"""

    chat_template = "fake-template"
    eos_token_id = 99

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [1] * (len(text) + (2 if add_special_tokens else 0))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return f"U:{messages[0]['content']}|A:"


def write_jsonl(path: Path, records: list[object | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            if isinstance(record, str):
                file.write(record)
            else:
                json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def sft(user: object, answer: object) -> dict[str, object]:
    return {
        "conversations": [
            {"from": "human", "value": user},
            {"from": "gpt", "value": answer},
        ]
    }


def create_processed_data(pt_dir: Path, sft_dir: Path) -> None:
    split_values = {"train": ("abc", "xy"), "validation": ("abcd", "xyz"), "test": ("abcde", "wxyz")}
    for split, (text, answer) in split_values.items():
        write_jsonl(pt_dir / split / f"{split}.jsonl", [{"text": text}])
        write_jsonl(sft_dir / split / f"{split}.jsonl", [sft(text, answer)])


def arguments(pt_dir: Path, sft_dir: Path, output: Path, *, overwrite: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        tokenizer_name_or_path="fake",
        pt_dir=pt_dir,
        sft_dir=sft_dir,
        output_path=output,
        cache_dir=None,
        trust_remote_code=False,
        overwrite=overwrite,
    )


def test_pt_length_statistics_are_correct(tmp_path: Path) -> None:
    path = tmp_path / "pt.jsonl"
    write_jsonl(path, [{"text": "a"}, {"text": "abcd"}])
    lengths = analyze_pt_file(path, FakeTokenizer())
    assert lengths == [3, 6]
    result = summarize_pt(lengths)
    assert result["records"] == 2
    assert result["min"] == 3
    assert result["max"] == 6
    assert result["mean"] == 4.5
    assert result["total_tokens"] == 9


def test_sft_source_target_total_and_eos_are_correct(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    write_jsonl(path, [sft("abc", "xy")])
    source, target, total = analyze_sft_file(path, FakeTokenizer())
    # prompt 为 U:abc|A: 共8字符，source再加2个特殊 token。
    assert source == [10]
    assert target == [2]
    assert total == [13]
    assert total[0] == source[0] + target[0] + 1  # EOS


def test_percentiles_use_linear_interpolation() -> None:
    values = [1, 2, 3, 4, 100]
    assert percentile(values, 50) == 3
    assert percentile(values, 75) == 4
    assert percentile(values, 90) == pytest.approx(61.6)
    stats = distribution_statistics(values)
    assert stats["median"] == stats["p50"] == 3
    assert stats["p99"] == pytest.approx(96.16)


def test_threshold_counts_and_ratios_are_strictly_greater() -> None:
    result = summarize_pt([256, 257, 512, 513])
    assert result["over_thresholds"]["256"] == {"count": 3, "ratio": 0.75}
    assert result["over_thresholds"]["512"] == {"count": 1, "ratio": 0.25}
    assert result["over_thresholds"]["4096"] == {"count": 0, "ratio": 0.0}


def test_sft_ratio_distribution() -> None:
    result = summarize_sft([2, 4], [2, 1], [5, 6])
    assert result["records"] == 2
    assert result["target_token_ratio"]["min"] == pytest.approx(1 / 6)
    assert result["target_token_ratio"]["max"] == pytest.approx(2 / 5)
    assert result["total_tokens"]["sum"] == 11


def test_each_split_and_overall_are_reported(tmp_path: Path) -> None:
    pt_dir, sft_dir = tmp_path / "pt", tmp_path / "sft"
    create_processed_data(pt_dir, sft_dir)
    result = analyze_datasets(pt_dir, sft_dir, FakeTokenizer())
    assert set(result["pt"]) == {"train", "validation", "test", "overall"}
    assert set(result["sft"]) == {"train", "validation", "test", "overall"}
    assert result["pt"]["overall"]["records"] == 3
    assert result["pt"]["overall"]["total_tokens"] == 18
    assert result["sft"]["overall"]["records"] == 3
    assert result["sft"]["overall"]["total_tokens"]["sum"] == 45


def test_invalid_json_reports_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    write_jsonl(path, [{"text": "ok"}, "{bad"])
    with pytest.raises(DataFormatError, match=r"bad\.jsonl 第 2 行不是合法JSON"):
        analyze_pt_file(path, FakeTokenizer())


def test_pt_missing_or_wrong_text_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    write_jsonl(missing, [{"content": "x"}])
    with pytest.raises(DataFormatError, match="缺少必需字段 'text'"):
        analyze_pt_file(missing, FakeTokenizer())
    wrong = tmp_path / "wrong.jsonl"
    write_jsonl(wrong, [{"text": 1}])
    with pytest.raises(DataFormatError, match="'text' 必须是字符串"):
        analyze_pt_file(wrong, FakeTokenizer())


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"conversations": []},
        {"conversations": [{"from": "gpt", "value": "x"}, {"from": "human", "value": "y"}]},
        sft("x", 1),
    ],
)
def test_bad_sft_conversation_is_rejected(tmp_path: Path, record: dict[str, object]) -> None:
    path = tmp_path / "bad-sft.jsonl"
    write_jsonl(path, [record])
    with pytest.raises(DataFormatError, match="conversations|from|value"):
        analyze_sft_file(path, FakeTokenizer())


def test_missing_chat_template_is_rejected(tmp_path: Path) -> None:
    class NoTemplateTokenizer(FakeTokenizer):
        chat_template = None

    path = tmp_path / "sft.jsonl"
    write_jsonl(path, [sft("x", "y")])
    with pytest.raises(ValueError, match="没有 chat_template"):
        analyze_sft_file(path, NoTemplateTokenizer())


@pytest.mark.parametrize("kind", ["pt", "sft"])
def test_empty_data_file_is_rejected(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DataFormatError, match="空数据文件"):
        if kind == "pt":
            analyze_pt_file(path, FakeTokenizer())
        else:
            analyze_sft_file(path, FakeTokenizer())


def test_report_structure_and_overwrite_policy(tmp_path: Path) -> None:
    pt_dir, sft_dir, output = tmp_path / "pt", tmp_path / "sft", tmp_path / "report.json"
    create_processed_data(pt_dir, sft_dir)
    report = run(arguments(pt_dir, sft_dir, output), FakeTokenizer())
    assert output.is_file()
    assert set(report) == {"configuration", "pt", "sft"}
    assert set(report["pt"]) == {"train", "validation", "test", "overall"}
    assert "source_tokens" in report["sft"]["train"]
    assert "target_tokens" in report["sft"]["train"]
    assert "total_tokens" in report["sft"]["train"]
    assert "target_token_ratio" in report["sft"]["train"]
    with pytest.raises(FileExistsError, match="--overwrite"):
        run(arguments(pt_dir, sft_dir, output), FakeTokenizer())
    overwritten = run(arguments(pt_dir, sft_dir, output, overwrite=True), FakeTokenizer())
    assert overwritten["pt"]["overall"]["records"] == 3


def test_end_to_end_cli_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pt_dir, sft_dir, output = tmp_path / "pt", tmp_path / "sft", tmp_path / "reports" / "stats.json"
    create_processed_data(pt_dir, sft_dir)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_from_pretrained(name: str, **kwargs: object) -> FakeTokenizer:
        calls.append((name, kwargs))
        return FakeTokenizer()

    monkeypatch.setattr(analyzer.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    exit_code = main(
        [
            "--tokenizer_name_or_path",
            "fake/local",
            "--pt_dir",
            str(pt_dir),
            "--sft_dir",
            str(sft_dir),
            "--output_path",
            str(output),
            "--trust_remote_code",
        ]
    )
    assert exit_code == 0
    assert calls == [("fake/local", {"cache_dir": None, "trust_remote_code": True})]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["configuration"]["tokenizer_name_or_path"] == "fake/local"
    assert report["pt"]["overall"]["records"] == 3


def test_main_returns_one_and_writes_stderr_on_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--pt_dir", str(tmp_path / "missing"), "--sft_dir", str(tmp_path / "missing")])
    assert code == 1
    assert "错误：" in capsys.readouterr().err


def test_cli_defaults() -> None:
    args = parse_args([])
    assert args.tokenizer_name_or_path == "Qwen/Qwen3.5-2B-Base"
    assert args.output_path == Path("reports/data/token_length_statistics.json")
