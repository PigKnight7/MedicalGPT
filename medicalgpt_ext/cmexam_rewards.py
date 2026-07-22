"""CMExam 的纯函数奖励核心与 TRL GRPO 奖励适配器。"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable

from medicalgpt_ext.cmexam_utils import (
    exact_set_match,
    extract_predicted_labels,
    normalize_answer_labels,
)


_ANSWER_ONLY_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class CMExamRewardConfig:
    """CMExam 三个奖励分项的不可变配置。"""

    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    format_reward: float = 0.1
    invalid_penalty: float = -0.1

    def __post_init__(self) -> None:
        for field_name in (
            "correct_reward",
            "incorrect_reward",
            "format_reward",
            "invalid_penalty",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} 必须是有限浮点数。")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} 必须是有限浮点数。")
            object.__setattr__(self, field_name, value)
        if self.correct_reward <= self.incorrect_reward:
            raise ValueError("correct_reward 必须大于 incorrect_reward。")
        if self.format_reward > self.correct_reward:
            raise ValueError("format_reward 不得大于 correct_reward。")
        if self.invalid_penalty > 0:
            raise ValueError("invalid_penalty 必须小于等于 0。")


DEFAULT_REWARD_CONFIG = CMExamRewardConfig()


@dataclass(frozen=True)
class CMExamRewardBreakdown:
    """单条 completion 的解析结果和三个奖励分项。"""

    predicted_labels: tuple[str, ...] | None
    gold_labels: tuple[str, ...]
    valid_prediction: bool
    format_valid: bool
    exact_match: bool
    parse_source: str
    parse_error: str | None
    correctness_reward: float
    format_reward: float
    invalid_penalty: float
    total_reward: float


def completion_to_text(completion: object) -> str:
    """提取 TRL 1.8 普通文本或对话 completion 中的 assistant 文本。"""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        messages: Sequence[object] = (completion,)
    elif isinstance(completion, Sequence) and not isinstance(completion, (bytes, bytearray)):
        messages = completion
    else:
        raise TypeError(
            "completion 必须是字符串、role/content 消息对象或消息列表，"
            f"实际为 {type(completion).__name__}。"
        )
    if not messages:
        raise ValueError("对话 completion 不能为空消息列表。")

    assistant_contents: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"completion 消息 {message_index} 必须是 role/content 对象。")
        role = message.get("role")
        if role != "assistant":
            raise ValueError(f"completion 消息 {message_index} 的 role 必须是 assistant。")
        if "content" not in message:
            raise ValueError(f"completion 消息 {message_index} 缺少 content。")
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError(f"completion 消息 {message_index} 的 content 必须是字符串。")
        assistant_contents.append(content)

    if len(assistant_contents) != 1:
        raise ValueError(
            "对话 completion 必须恰好包含一个 assistant 生成内容，"
            f"实际为 {len(assistant_contents)} 个。"
        )
    return assistant_contents[0]


def is_answer_only_format(completion: object) -> bool:
    """判断输出是否仅由一个内容合法的 ``<answer>`` 标签组成。"""

    try:
        text = completion_to_text(completion).strip()
    except (TypeError, ValueError):
        return False
    match = _ANSWER_ONLY_TAG_PATTERN.fullmatch(text)
    return match is not None and normalize_answer_labels(match.group(1)) is not None


def _normalize_gold_labels(gold_labels: object) -> tuple[str, ...]:
    """严格校验数据集标准答案，避免把数据错误当作模型错误。"""

    if isinstance(gold_labels, str):
        candidate: str | Sequence[str] = gold_labels
    elif isinstance(gold_labels, (list, tuple)) and all(isinstance(item, str) for item in gold_labels):
        candidate = gold_labels
    else:
        raise ValueError("标准答案必须是 A-E 字符串或字符串 list/tuple。")
    normalized = normalize_answer_labels(candidate)
    if normalized is None:
        raise ValueError(f"标准答案非法：{gold_labels!r}。")
    return normalized


def score_cmexam_completion(
    completion: object,
    gold_labels: object,
    config: CMExamRewardConfig = DEFAULT_REWARD_CONFIG,
) -> CMExamRewardBreakdown:
    """按 exact-set、answer-only 格式和无效预测三个分项评分。"""

    gold = _normalize_gold_labels(gold_labels)
    try:
        text = completion_to_text(completion)
    except (TypeError, ValueError) as exc:
        predicted = None
        valid = False
        source = "completion_structure"
        error = str(exc)
    else:
        parsed = extract_predicted_labels(text)
        predicted = parsed.labels
        valid = parsed.valid
        source = parsed.source
        error = parsed.error

    format_valid = is_answer_only_format(completion)
    matched = valid and exact_set_match(predicted, gold)
    correctness = config.correct_reward if matched else config.incorrect_reward
    format_component = config.format_reward if format_valid else 0.0
    invalid_component = 0.0 if valid else config.invalid_penalty
    total = correctness + format_component + invalid_component
    return CMExamRewardBreakdown(
        predicted_labels=predicted,
        gold_labels=gold,
        valid_prediction=valid,
        format_valid=format_valid,
        exact_match=matched,
        parse_source=source,
        parse_error=error,
        correctness_reward=correctness,
        format_reward=format_component,
        invalid_penalty=invalid_component,
        total_reward=total,
    )


def score_cmexam_batch(
    completions: Sequence[object],
    answer_labels: Sequence[object],
    config: CMExamRewardConfig = DEFAULT_REWARD_CONFIG,
    *,
    ids: Sequence[object] | None = None,
) -> list[CMExamRewardBreakdown]:
    """批量评分，并为标准答案错误补充样本索引和可选 id。"""

    if len(completions) != len(answer_labels):
        raise ValueError(
            "completions 与 answer_labels 长度不一致："
            f"{len(completions)} != {len(answer_labels)}。"
        )
    if ids is not None and len(ids) != len(completions):
        raise ValueError(f"id 与 completions 长度不一致：{len(ids)} != {len(completions)}。")
    results: list[CMExamRewardBreakdown] = []
    for index, (completion, gold) in enumerate(zip(completions, answer_labels, strict=True)):
        try:
            results.append(score_cmexam_completion(completion, gold, config))
        except (TypeError, ValueError) as exc:
            sample_id = "" if ids is None else f"，id={ids[index]!r}"
            raise ValueError(f"CMExam 样本索引 {index}{sample_id}：{exc}") from exc
    return results


def _ids_from_kwargs(kwargs: Mapping[str, object]) -> Sequence[object] | None:
    ids = kwargs.get("id")
    if ids is None:
        return None
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
        raise ValueError("TRL 传入的 id 字段必须是逐样本序列。")
    return ids


def exact_set_correctness_reward(
    completions: Sequence[object],
    answer_labels: Sequence[object],
    **kwargs: object,
) -> list[float]:
    """TRL 批量 exact-set 正确性奖励。"""

    return [
        item.correctness_reward
        for item in score_cmexam_batch(completions, answer_labels, ids=_ids_from_kwargs(kwargs))
    ]


def answer_only_format_reward(
    completions: Sequence[object],
    **kwargs: object,
) -> list[float]:
    """TRL 批量 answer-only 格式奖励。"""

    del kwargs
    return [DEFAULT_REWARD_CONFIG.format_reward if is_answer_only_format(item) else 0.0 for item in completions]


def invalid_answer_penalty(
    completions: Sequence[object],
    **kwargs: object,
) -> list[float]:
    """TRL 批量无效答案惩罚。"""

    del kwargs
    rewards: list[float] = []
    for completion in completions:
        try:
            parsed = extract_predicted_labels(completion_to_text(completion))
        except (TypeError, ValueError):
            rewards.append(DEFAULT_REWARD_CONFIG.invalid_penalty)
        else:
            rewards.append(0.0 if parsed.valid else DEFAULT_REWARD_CONFIG.invalid_penalty)
    return rewards


RewardFunction = Callable[..., list[float]]


def build_cmexam_reward_functions(
    config: CMExamRewardConfig = DEFAULT_REWARD_CONFIG,
) -> list[RewardFunction]:
    """构造可直接传给 TRL 1.8 ``GRPOTrainer`` 的三个奖励函数。"""

    def correctness(
        completions: Sequence[object], answer_labels: Sequence[object], **kwargs: object
    ) -> list[float]:
        return [
            item.correctness_reward
            for item in score_cmexam_batch(
                completions, answer_labels, config, ids=_ids_from_kwargs(kwargs)
            )
        ]

    def format_component(completions: Sequence[object], **kwargs: object) -> list[float]:
        del kwargs
        return [config.format_reward if is_answer_only_format(item) else 0.0 for item in completions]

    def invalid_component(completions: Sequence[object], **kwargs: object) -> list[float]:
        del kwargs
        rewards: list[float] = []
        for completion in completions:
            try:
                valid = extract_predicted_labels(completion_to_text(completion)).valid
            except (TypeError, ValueError):
                valid = False
            rewards.append(0.0 if valid else config.invalid_penalty)
        return rewards

    correctness.__name__ = "exact_set_correctness_reward"
    format_component.__name__ = "answer_only_format_reward"
    invalid_component.__name__ = "invalid_answer_penalty"
    return [correctness, format_component, invalid_component]


__all__ = [
    "CMExamRewardBreakdown",
    "CMExamRewardConfig",
    "DEFAULT_REWARD_CONFIG",
    "answer_only_format_reward",
    "build_cmexam_reward_functions",
    "completion_to_text",
    "exact_set_correctness_reward",
    "invalid_answer_penalty",
    "is_answer_only_format",
    "score_cmexam_batch",
    "score_cmexam_completion",
]
