"""CMExam 预处理、评估和奖励函数共用的严格工具。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


LABEL_ORDER = "ABCDE"
LABEL_SET = frozenset(LABEL_ORDER)
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
SEPARATOR_PATTERN = re.compile(r"[\s,，、;；/|+和及与]+")
FINAL_EXPRESSION_ANY_PATTERN = re.compile(
    r"(?:最终答案|答案(?:为|是)?|正确选项(?:为|是)?|我选择)\s*[:：]?\s*"
    r"([A-Za-z](?:[\s,，、;；/|+和及与]*[A-Za-z])*)"
    r"(?=\s*(?:[。.!！；;\n]|$))",
    re.I,
)
ANSWER_ONLY_PATTERN = re.compile(r"^[\s,，、;；/|+和及与A-Za-z]+$")


class ChatTokenizer(Protocol):
    """format_cmexam_prompt 所需的最小 tokenizer 接口。"""

    chat_template: object

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


@dataclass(frozen=True)
class ParsedAnswer:
    """模型答案解析结果。"""

    labels: tuple[str, ...] | None
    valid: bool
    format_valid: bool
    source: str
    error: str | None = None


def _normalize_label_payload(value: str) -> tuple[str, ...] | None:
    """仅解析已经确定为答案区域的内容。"""

    payload = value.strip()
    if not payload:
        return None
    compact = SEPARATOR_PATTERN.sub("", payload.upper())
    if not compact or not compact.isalpha():
        return None
    if any(character not in LABEL_SET for character in compact):
        return None
    return tuple(label for label in LABEL_ORDER if label in compact)


def normalize_answer_labels(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """将明确答案规范化为按 A-E 排序、去重的标签元组。"""

    if value is None:
        return None
    if not isinstance(value, str):
        value = "".join(str(item) for item in value)
    text = value.strip()
    if not text:
        return None

    tags = ANSWER_TAG_PATTERN.findall(text)
    if tags:
        parsed_tags = [_normalize_label_payload(item) for item in tags]
        if any(item is None for item in parsed_tags) or len(set(parsed_tags)) != 1:
            return None
        return parsed_tags[0]

    text = re.sub(r"^\s*(?:答案|正确答案)\s*[:：]?\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*选项\s*", "", text, flags=re.I)
    return _normalize_label_payload(text)


def extract_predicted_labels(text: str | None) -> ParsedAnswer:
    """按 answer 标签、末尾答案表达、纯标签输出的优先级解析生成结果。"""

    if text is None or not text.strip():
        return ParsedAnswer(None, False, False, "none", "没有答案")
    output = text.strip()

    tags = ANSWER_TAG_PATTERN.findall(output)
    if tags:
        parsed = [_normalize_label_payload(item) for item in tags]
        if any(item is None for item in parsed):
            return ParsedAnswer(None, False, False, "answer_tag", "answer 标签为空或包含非法标签")
        if len(set(parsed)) != 1:
            return ParsedAnswer(None, False, False, "answer_tag", "存在多个内容冲突的 answer 标签")
        return ParsedAnswer(parsed[0], True, True, "answer_tag")
    if re.search(r"</?answer\b", output, re.I):
        return ParsedAnswer(None, False, False, "answer_tag", "answer 标签不完整")

    matches = FINAL_EXPRESSION_ANY_PATTERN.findall(output)
    if matches:
        parsed = [_normalize_label_payload(item) for item in matches]
        if any(item is None for item in parsed):
            return ParsedAnswer(None, False, False, "final_expression", "最终答案包含非法标签")
        if len(set(parsed)) != 1:
            return ParsedAnswer(None, False, False, "final_expression", "存在互相冲突的最终答案")
        # 可解析但未遵循要求的 answer 标签格式。
        return ParsedAnswer(parsed[0], True, False, "final_expression")

    if ANSWER_ONLY_PATTERN.fullmatch(output):
        labels = _normalize_label_payload(output)
        if labels is not None:
            return ParsedAnswer(labels, True, False, "answer_only")
        return ParsedAnswer(None, False, False, "answer_only", "纯答案输出包含非法标签")

    if re.search(r"(?:可能|也许|或许).*[A-E].*(?:或|或者).*[A-E]", output, re.I):
        return ParsedAnswer(None, False, False, "none", "答案表达不确定")
    return ParsedAnswer(None, False, False, "none", "未找到明确的最终答案")


def exact_set_match(
    predicted: Sequence[str] | None,
    gold: Sequence[str] | None,
) -> bool:
    """单选和多选统一按标签集合完全相等计分。"""

    if not predicted or not gold:
        return False
    return set(predicted) == set(gold)


def format_answer_labels(labels: Sequence[str] | None) -> str:
    """将标签序列规范化为连续的 A-E 答案字符串。"""

    normalized = normalize_answer_labels(labels)
    return "" if normalized is None else "".join(normalized)


def validate_options(options: object) -> tuple[dict[str, str], ...]:
    """校验结构化选项，返回按 A-E 排序的不可变结果。"""

    if not isinstance(options, list) or len(options) < 2:
        raise ValueError("CMExam options 必须是至少包含两个选项的列表。")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, option in enumerate(options):
        if not isinstance(option, dict):
            raise ValueError(f"options[{index}] 必须是对象。")
        label = option.get("label")
        text = option.get("text")
        if not isinstance(label, str) or label.upper() not in LABEL_SET:
            raise ValueError(f"options[{index}] 的 label 必须是 A-E。")
        label = label.upper()
        if label in seen:
            raise ValueError(f"选项标签重复：{label}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"选项 {label} 的内容为空。")
        seen.add(label)
        normalized.append({"label": label, "text": text})
    return tuple(sorted(normalized, key=lambda item: LABEL_ORDER.index(item["label"])))


def format_cmexam_prompt(
    tokenizer: ChatTokenizer,
    question: str,
    *,
    prompt_style: str = "answer_only",
) -> str:
    """使用 tokenizer 自带 chat template 构造统一 CMExam prompt。"""

    if prompt_style not in {"answer_only", "reasoning_and_answer"}:
        raise ValueError(f"不支持的 prompt_style：{prompt_style}")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer 没有 chat_template，无法构造公平一致的评估 prompt。")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("CMExam question 不能为空。")

    if prompt_style == "answer_only":
        system = (
            "你是一名医学考试答题助手。请根据题目选择所有正确选项。"
            "单选题只输出一个字母，多选题按A到E顺序输出全部字母。"
            "最终答案必须放在<answer></answer>标签中。"
            "例如：<answer>A</answer>或<answer>AE</answer>。"
            "不要在标签中输出其他文字，也不要输出解释。"
        )
    else:
        system = (
            "你是一名医学考试答题助手。请根据题目选择所有正确选项，可以先简要分析。"
            "最后必须按A到E顺序把全部正确选项放在<answer></answer>标签中，"
            "例如：<answer>A</answer>或<answer>AE</answer>。标签中不要输出其他文字。"
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


__all__ = [
    "ParsedAnswer",
    "exact_set_match",
    "extract_predicted_labels",
    "format_answer_labels",
    "format_cmexam_prompt",
    "normalize_answer_labels",
    "validate_options",
]
