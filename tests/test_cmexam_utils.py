"""CMExam 公共答案解析和 prompt 工具测试。"""

from __future__ import annotations

import pytest

from medicalgpt_ext.cmexam_utils import (
    exact_set_match,
    extract_predicted_labels,
    format_answer_labels,
    format_cmexam_prompt,
    normalize_answer_labels,
    validate_options,
)


class PromptTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(f"{item['role']}:{item['content']}" for item in conversation) + "\nassistant:"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A", ("A",)),
        ("AE", ("A", "E")),
        ("A,E", ("A", "E")),
        ("A、E", ("A", "E")),
        ("答案：A和E", ("A", "E")),
        ("选项A和E", ("A", "E")),
        ("<answer>AE</answer>", ("A", "E")),
        ("AAE", ("A", "E")),
        ("ECA", ("A", "C", "E")),
    ],
)
def test_normalize_answer_labels(raw: str, expected: tuple[str, ...]) -> None:
    assert normalize_answer_labels(raw) == expected


@pytest.mark.parametrize("raw", ["", "F", "<answer></answer>", "答案可能是A或C"])
def test_normalize_rejects_invalid_or_uncertain(raw: str) -> None:
    assert normalize_answer_labels(raw) is None


def test_answer_tag_priority_and_same_tags() -> None:
    parsed = extract_predicted_labels("分析中提到B。<answer>EA</answer><answer>AE</answer>")
    assert parsed.labels == ("A", "E")
    assert parsed.valid and parsed.format_valid and parsed.source == "answer_tag"


@pytest.mark.parametrize(
    ("text", "error_part"),
    [
        ("<answer></answer>", "为空"),
        ("<answer>A</answer><answer>C</answer>", "冲突"),
        ("<answer>F</answer>", "非法"),
        ("可能是A或C", "不确定"),
        ("患者使用维生素A和维生素E后症状改善。", "未找到"),
    ],
)
def test_extract_rejects_ambiguous_outputs(text: str, error_part: str) -> None:
    parsed = extract_predicted_labels(text)
    assert not parsed.valid and not parsed.format_valid
    assert error_part in (parsed.error or "")


def test_conflicting_explicit_final_answers_are_rejected() -> None:
    parsed = extract_predicted_labels("答案：A。重新考虑后，最终答案：C。")
    assert not parsed.valid and "冲突" in (parsed.error or "")


def test_explicit_final_and_answer_only_are_valid_but_format_invalid() -> None:
    final = extract_predicted_labels("简要分析。最终答案：E、A")
    bare = extract_predicted_labels("E,A")
    assert final.labels == bare.labels == ("A", "E")
    assert final.valid and bare.valid
    assert not final.format_valid and not bare.format_valid
    assert final.source == "final_expression" and bare.source == "answer_only"


def test_exact_set_match_and_format() -> None:
    assert exact_set_match(("E", "A"), ("A", "E"))
    assert not exact_set_match(("A",), ("A", "E"))
    assert not exact_set_match(None, ("A",))
    assert format_answer_labels(("E", "A", "A")) == "AE"


def test_prompt_uses_chat_template_and_requires_answer_tag() -> None:
    prompt = format_cmexam_prompt(PromptTokenizer(), "题干\nA. 甲\nB. 乙")
    assert "system:" in prompt and "user:题干" in prompt
    assert "<answer>AE</answer>" in prompt and prompt.endswith("assistant:")
    reasoning = format_cmexam_prompt(PromptTokenizer(), "题目", prompt_style="reasoning_and_answer")
    assert "可以先简要分析" in reasoning and "<answer>" in reasoning


def test_prompt_rejects_tokenizer_without_chat_template() -> None:
    tokenizer = PromptTokenizer()
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="chat_template"):
        format_cmexam_prompt(tokenizer, "题目")


def test_validate_options() -> None:
    result = validate_options([{"label": "B", "text": "乙"}, {"label": "A", "text": "甲"}])
    assert [item["label"] for item in result] == ["A", "B"]
    with pytest.raises(ValueError, match="重复"):
        validate_options([{"label": "A", "text": "甲"}, {"label": "A", "text": "乙"}])
