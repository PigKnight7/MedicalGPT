"""CMExam 公共奖励模块测试；不加载模型或访问网络。"""

from __future__ import annotations

import math
from dataclasses import fields

import pytest

from medicalgpt_ext.cmexam_rewards import (
    CMExamRewardBreakdown,
    CMExamRewardConfig,
    answer_only_format_reward,
    build_cmexam_reward_functions,
    completion_to_text,
    exact_set_correctness_reward,
    invalid_answer_penalty,
    is_answer_only_format,
    score_cmexam_completion,
)


@pytest.mark.parametrize(
    ("completion", "gold", "expected"),
    [
        ("<answer>A</answer>", ["A"], True),
        ("<answer>B</answer>", ["A"], False),
        ("<answer>AE</answer>", ["A", "E"], True),
        ("<answer>EA</answer>", ["A", "E"], True),
        ("<answer>A</answer>", ["A", "E"], False),
        ("<answer>ABE</answer>", ["A", "E"], False),
        ("<answer>AAE</answer>", ["A", "E"], True),
        ("", ["A"], False),
        ("<answer>F</answer>", ["A"], False),
    ],
)
def test_single_completion_exact_set(completion: str, gold: list[str], expected: bool) -> None:
    result = score_cmexam_completion(completion, gold)
    assert result.exact_match is expected
    assert result.correctness_reward == (1.0 if expected else 0.0)


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("<answer>A</answer>", True),
        ("<answer>AE</answer>", True),
        ("  \n<answer>A,E</answer>\t", True),
        ("A", False),
        ("答案：A", False),
        ("解释。<answer>A</answer>", False),
        ("<answer>A</answer>谢谢", False),
        ("```<answer>A</answer>```", False),
        ("<answer></answer>", False),
        ("<answer>A</answer><answer>A</answer>", False),
        ("<answer>A</answer><answer>C</answer>", False),
        ("<answer>F</answer>", False),
    ],
)
def test_answer_only_format(completion: str, expected: bool) -> None:
    assert is_answer_only_format(completion) is expected
    assert answer_only_format_reward([completion]) == ([0.1] if expected else [0.0])


@pytest.mark.parametrize(
    "completion",
    ["", "<answer>F</answer>", "可能是A或C", "患者使用维生素A和维生素E后症状改善。"],
)
def test_invalid_prediction_is_penalized(completion: str) -> None:
    result = score_cmexam_completion(completion, ["A"])
    assert not result.valid_prediction
    assert result.invalid_penalty == -0.1
    assert invalid_answer_penalty([completion]) == [-0.1]


@pytest.mark.parametrize("completion", ["<answer>B</answer>", "<answer>A</answer>"])
def test_valid_prediction_has_no_invalid_penalty(completion: str) -> None:
    assert score_cmexam_completion(completion, ["A"]).invalid_penalty == 0.0
    assert invalid_answer_penalty([completion]) == [0.0]


@pytest.mark.parametrize(
    ("completion", "expected_parts"),
    [
        ("<answer>A</answer>", (1.0, 0.1, 0.0, 1.1)),
        ("最终答案：A", (1.0, 0.0, 0.0, 1.0)),
        ("<answer>B</answer>", (0.0, 0.1, 0.0, 0.1)),
        ("无明确答案", (0.0, 0.0, -0.1, -0.1)),
    ],
)
def test_reward_components_and_total(
    completion: str, expected_parts: tuple[float, float, float, float]
) -> None:
    result = score_cmexam_completion(completion, ["A"])
    actual = (
        result.correctness_reward,
        result.format_reward,
        result.invalid_penalty,
        result.total_reward,
    )
    assert actual == pytest.approx(expected_parts)
    assert result.total_reward == pytest.approx(sum(actual[:3]))
    assert {item.name for item in fields(CMExamRewardBreakdown)} == {
        "predicted_labels", "gold_labels", "valid_prediction", "format_valid", "exact_match",
        "parse_source", "parse_error", "correctness_reward", "format_reward",
        "invalid_penalty", "total_reward",
    }


def test_batch_reward_functions_mixed_and_finite() -> None:
    completions = ["<answer>A</answer>", "<answer>EA</answer>", "<answer>B</answer>"]
    gold = [["A"], ("A", "E"), "A"]
    assert exact_set_correctness_reward(completions, gold) == [1.0, 1.0, 0.0]
    assert answer_only_format_reward(completions) == [0.1, 0.1, 0.1]
    assert invalid_answer_penalty(completions) == [0.0, 0.0, 0.0]
    all_rewards = [
        value
        for index, function in enumerate(build_cmexam_reward_functions())
        for value in (
            function(completions=completions, answer_labels=gold)
            if index == 0
            else function(completions=completions)
        )
    ]
    assert all(isinstance(value, float) and math.isfinite(value) for value in all_rewards)


def test_batch_length_mismatch_and_invalid_gold_include_context() -> None:
    with pytest.raises(ValueError, match="长度不一致"):
        exact_set_correctness_reward(["A"], [["A"], ["B"]])
    with pytest.raises(ValueError, match=r"索引 1.*id='bad-id'.*标准答案非法"):
        exact_set_correctness_reward(
            ["A", "B"], [["A"], ["F"]], id=["good-id", "bad-id"]
        )


def test_batch_does_not_modify_inputs() -> None:
    completions = [[{"role": "assistant", "content": "<answer>A</answer>"}]]
    gold = [["A"]]
    original_completions = [[dict(completions[0][0])]]
    original_gold = [list(gold[0])]
    assert exact_set_correctness_reward(completions, gold) == [1.0]
    assert completions == original_completions and gold == original_gold


def test_built_functions_names_order_and_custom_values() -> None:
    config = CMExamRewardConfig(2, -0.5, 0.25, -0.2)
    functions = build_cmexam_reward_functions(config)
    assert [function.__name__ for function in functions] == [
        "exact_set_correctness_reward", "answer_only_format_reward", "invalid_answer_penalty"
    ]
    assert len(functions) == 3 and all(callable(function) for function in functions)
    assert functions[0](completions=["<answer>B</answer>"], answer_labels=[["A"]]) == [-0.5]
    assert functions[1](completions=["<answer>B</answer>"]) == [0.25]
    assert functions[2](completions=[""]) == [-0.2]


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("<answer>A</answer>", "<answer>A</answer>"),
        ([{"role": "assistant", "content": "<answer>AE</answer>"}], "<answer>AE</answer>"),
        ({"role": "assistant", "content": "A"}, "A"),
    ],
)
def test_completion_to_text_supported_structures(completion: object, expected: str) -> None:
    assert completion_to_text(completion) == expected


@pytest.mark.parametrize(
    "completion",
    [
        [{"role": "assistant"}],
        123,
        [{"role": "assistant", "content": "A"}, {"role": "assistant", "content": "C"}],
    ],
)
def test_invalid_completion_structure_becomes_invalid_prediction(completion: object) -> None:
    result = score_cmexam_completion(completion, ["A"])
    assert not result.valid_prediction
    assert result.parse_source == "completion_structure"
    assert result.parse_error
    assert result.invalid_penalty == -0.1


def test_default_and_custom_config() -> None:
    assert CMExamRewardConfig() == CMExamRewardConfig(1.0, 0.0, 0.1, -0.1)
    custom = CMExamRewardConfig(2, -1, 0.5, 0)
    assert (custom.correct_reward, custom.incorrect_reward, custom.format_reward, custom.invalid_penalty) == (
        2.0, -1.0, 0.5, 0.0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"correct_reward": float("nan")},
        {"format_reward": float("inf")},
        {"correct_reward": 0, "incorrect_reward": 0},
        {"format_reward": 2},
        {"invalid_penalty": 0.1},
    ],
)
def test_invalid_config_raises(kwargs: dict[str, float]) -> None:
    with pytest.raises((TypeError, ValueError)):
        CMExamRewardConfig(**kwargs)
