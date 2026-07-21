"""PT 困惑度评估测试；不访问网络或真实模型。"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluation.eval_pt_perplexity import (
    EvaluationError,
    apply_max_blocks,
    load_model_and_tokenizer,
    load_pt_records,
    parse_args,
    prepare_output_dir,
    resolve_device,
    resolve_torch_dtype,
    run,
    safe_perplexity,
    score_blocks,
    tokenize_and_pack,
)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def write_records(path: Path, texts: list[str]) -> None:
    write_lines(path, [json.dumps({"text": text}, ensure_ascii=False) for text in texts])


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[ord(char) % 5 + 2 for char in text] for text in texts]}


class FakeCausalLM:
    def __init__(self, vocab_size: int = 8):
        self.vocab_size = vocab_size
        self.eval_calls = 0
        self.to_calls: list[str] = []

    def to(self, device):
        self.to_calls.append(str(device))
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, input_ids, attention_mask):
        # 每个位置固定 logits，便于人工用 cross_entropy 校验。
        batch, width = input_ids.shape
        logits = torch.zeros(batch, width, self.vocab_size, device=input_ids.device)
        logits[..., 1] = 1.0
        return SimpleNamespace(logits=logits)


def args(data: Path, out: Path, *extra: str):
    return parse_args([
        "--model_name_or_path", "fake-base", "--validation_file_dir", str(data),
        "--output_dir", str(out), "--device", "cpu", "--block_size", "4", *extra,
    ])


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_read_single_file_blank_and_empty(tmp_path: Path) -> None:
    path = tmp_path / "validation.jsonl"
    write_lines(path, [json.dumps({"text": "甲"}), "", json.dumps({"text": "  "}), json.dumps({"text": "乙"})])
    texts, stats, files = load_pt_records(tmp_path)
    assert texts == ["甲", "乙"] and files == [path]
    assert stats.raw_records == 3 and stats.valid_records == 2
    assert stats.blank_lines == 1 and stats.empty_text_records == 1


def test_recursive_files_are_sorted(tmp_path: Path) -> None:
    write_records(tmp_path / "z/b.jsonl", ["乙"])
    write_records(tmp_path / "a.jsonl", ["甲"])
    texts, stats, files = load_pt_records(tmp_path)
    assert texts == ["甲", "乙"]
    assert files == sorted(files) and stats.source_files == 2


@pytest.mark.parametrize(
    ("line", "message"),
    [("{bad", "不是合法 JSON"), ("[]", "JSON 对象"), ('{"other": 1}', "缺少必需字段"), ('{"text": 1}', "必须是字符串")],
)
def test_invalid_records_report_file_and_line(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "bad.jsonl"
    write_lines(path, [line])
    with pytest.raises(EvaluationError, match=message) as caught:
        load_pt_records(tmp_path)
    assert str(path) in str(caught.value) and "第 1 行" in str(caught.value)


def test_empty_directory_and_empty_dataset_fail(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="没有 JSONL"):
        load_pt_records(tmp_path)
    write_records(tmp_path / "empty.jsonl", [""])
    with pytest.raises(EvaluationError, match="没有可用于评估"):
        load_pt_records(tmp_path)


class ExactTokenizer(FakeTokenizer):
    def __call__(self, texts):
        mapping = {"a": [2, 3], "b": [4, 1], "c": [5]}
        return {"input_ids": [mapping[text] for text in texts]}


def test_eos_and_packing_match_training_tail_rule() -> None:
    blocks, stats = tokenize_and_pack(["a", "b", "c"], ExactTokenizer(), 4)
    # a 追加EOS；b已有EOS；c追加EOS => 3+2+2=7，保留首4、丢3。
    assert blocks == [[2, 3, 1, 4]]
    assert stats == {
        "total_raw_tokens": 7, "packed_tokens": 4,
        "dropped_remainder_tokens": 3, "dropped_remainder_ratio": pytest.approx(3 / 7),
    }
    short, short_stats = tokenize_and_pack(["c"], ExactTokenizer(), 4)
    assert short == [[5, 1]] and short_stats["dropped_remainder_tokens"] == 0


def test_map_batch_boundary_and_max_blocks_after_packing() -> None:
    blocks, _ = tokenize_and_pack(["c", "c"], ExactTokenizer(), 2, packing_batch_size=1)
    assert blocks == [[5, 1], [5, 1]]
    assert apply_max_blocks(blocks, 1) == [[5, 1]]
    assert apply_max_blocks(blocks, None) == blocks


def expected_row_nll(values: list[int], vocab: int = 8) -> float:
    logits = torch.zeros(len(values) - 1, vocab)
    logits[:, 1] = 1.0
    return float(torch.nn.functional.cross_entropy(logits, torch.tensor(values[1:]), reduction="sum"))


def test_shift_weighted_nll_perplexity_and_padding() -> None:
    blocks = [[2, 1, 3, 4], [2, 3]]
    totals, rows = score_blocks(FakeCausalLM(), blocks, batch_size=2, pad_token_id=0, device="cpu")
    expected = expected_row_nll(blocks[0]) + expected_row_nll(blocks[1])
    assert [row["scored_tokens"] for row in rows] == [3, 1]
    assert totals["total_scored_tokens"] == 4
    assert totals["total_negative_log_likelihood"] == pytest.approx(expected)
    assert totals["mean_negative_log_likelihood"] == pytest.approx(expected / 4)
    assert totals["perplexity"] == pytest.approx(math.exp(expected / 4))


def test_different_batch_sizes_have_identical_weighted_result() -> None:
    blocks = [[2, 3, 4, 1], [5, 2], [3, 4, 5]]
    one, _ = score_blocks(FakeCausalLM(), blocks, batch_size=1, pad_token_id=0, device="cpu")
    three, _ = score_blocks(FakeCausalLM(), blocks, batch_size=3, pad_token_id=0, device="cpu")
    assert one["total_scored_tokens"] == three["total_scored_tokens"]
    assert one["total_negative_log_likelihood"] == pytest.approx(three["total_negative_log_likelihood"])
    assert one["mean_negative_log_likelihood"] == pytest.approx(three["mean_negative_log_likelihood"])


def test_scoring_uses_inference_mode(monkeypatch) -> None:
    entered = []
    original = torch.inference_mode
    class TrackingContext:
        def __enter__(self):
            entered.append(True)
            self.context = original()
            return self.context.__enter__()
        def __exit__(self, *exc):
            return self.context.__exit__(*exc)
    monkeypatch.setattr(torch, "inference_mode", TrackingContext)
    score_blocks(FakeCausalLM(), [[2, 3]], batch_size=1, pad_token_id=0, device="cpu")
    assert entered == [True]


def test_zero_scored_tokens_and_overflow() -> None:
    with pytest.raises(EvaluationError, match="total_scored_tokens 为 0"):
        score_blocks(FakeCausalLM(), [[2]], batch_size=1, pad_token_id=0, device="cpu")
    assert safe_perplexity(10000.0) == math.inf


def test_device_and_dtype(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu" and resolve_device("cpu") == "cpu"
    with pytest.raises(EvaluationError, match="没有可用 CUDA"):
        resolve_device("cuda")
    assert resolve_torch_dtype("auto") == "auto"
    assert resolve_torch_dtype("bfloat16") is torch.bfloat16
    assert resolve_torch_dtype("float16") is torch.float16
    assert resolve_torch_dtype("float32") is torch.float32


def test_model_and_adapter_loading(monkeypatch, tmp_path: Path) -> None:
    import peft
    import transformers
    tokenizer, base, wrapped = FakeTokenizer(), FakeCausalLM(), FakeCausalLM()
    calls: dict[str, object] = {}
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda source, **kw: calls.setdefault("tokenizer", source) and tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda source, **kw: calls.setdefault("model", source) and base)
    def load_adapter(model, path, **kwargs):
        calls.update(adapter=path, adapter_base=model, adapter_kwargs=kwargs)
        return wrapped
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", load_adapter)
    parsed = parse_args([
        "--model_name_or_path", "the-base", "--tokenizer_name_or_path", "the-tokenizer",
        "--peft_path", str(tmp_path / "adapter"), "--validation_file_dir", str(tmp_path),
        "--output_dir", str(tmp_path / "out"), "--device", "cpu", "--local_files_only",
    ])
    model, result_tokenizer, device = load_model_and_tokenizer(parsed)
    assert calls["model"] == "the-base" and calls["tokenizer"] == "the-tokenizer"
    assert calls["adapter_base"] is base and calls["adapter"] == str(tmp_path / "adapter")
    assert calls["adapter_kwargs"]["is_trainable"] is False
    assert wrapped.eval_calls == 1 and model is wrapped and result_tokenizer is tokenizer and device == "cpu"


def test_base_loading_does_not_treat_adapter_as_model(monkeypatch, tmp_path: Path) -> None:
    import transformers
    calls = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda source, **kw: calls.append(source) or FakeCausalLM())
    parsed = args(tmp_path, tmp_path / "out")
    model, _, _ = load_model_and_tokenizer(parsed)
    assert calls == ["fake-base"] and model.eval_calls == 1


def test_output_overwrite_policy(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir(); (out / "metrics.json").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_output_dir(out, overwrite=False)
    paths = prepare_output_dir(out, overwrite=True)
    assert not paths["metrics.json"].exists()


def test_full_fake_run_metrics_blocks_config_and_overwrite(tmp_path: Path) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    write_records(data / "validation.jsonl", ["ab", "c"])
    tokenizer, model = FakeTokenizer(), FakeCausalLM()
    parsed = args(data, out, "--batch_size", "2", "--overwrite")
    assert run(parsed, model=model, tokenizer=tokenizer) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    required = {
        "records", "source_files", "raw_records", "valid_records", "blank_lines", "empty_text_records",
        "block_size", "packed_blocks", "evaluated_blocks", "total_raw_tokens", "packed_tokens",
        "dropped_remainder_tokens", "dropped_remainder_ratio", "total_scored_tokens",
        "total_negative_log_likelihood", "mean_negative_log_likelihood", "perplexity", "batch_size",
        "elapsed_seconds", "blocks_per_second", "scored_tokens_per_second",
    }
    assert required <= metrics.keys()
    rows = read_jsonl(out / "block_metrics.jsonl")
    assert rows and {"block_index", "block_tokens", "scored_tokens", "negative_log_likelihood", "mean_negative_log_likelihood", "perplexity"} <= rows[0].keys()
    config_text = (out / "evaluation_config.json").read_text(encoding="utf-8").lower()
    assert "hf_token" not in config_text and "api_key" not in config_text
    (out / "block_metrics.jsonl").write_text('{"old": true}\n', encoding="utf-8")
    assert run(parsed, model=FakeCausalLM(), tokenizer=tokenizer) == 0
    assert "old" not in (out / "block_metrics.jsonl").read_text(encoding="utf-8")


def test_dry_run_does_not_load_model_or_create_metrics(tmp_path: Path, monkeypatch, capsys) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    write_records(data / "validation.jsonl", ["医学文本"])
    monkeypatch.setattr("evaluation.eval_pt_perplexity.load_model_and_tokenizer", lambda args: pytest.fail("不应加载"))
    assert run(args(data, out, "--dry_run", "--overwrite")) == 0
    assert "raw_records" in capsys.readouterr().out and not (out / "metrics.json").exists()


def test_cli_dry_run_end_to_end(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_records(data / "validation.jsonl", ["医学文本"])
    script = Path(__file__).parents[1] / "evaluation/eval_pt_perplexity.py"
    result = subprocess.run([
        sys.executable, str(script), "--model_name_or_path", "never-loaded",
        "--validation_file_dir", str(data), "--output_dir", str(tmp_path / "out"), "--dry_run",
    ], text=True, capture_output=True, check=False)
    assert result.returncode == 0 and "未加载模型" in result.stdout


def test_cuda_oom_message(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_records(data / "validation.jsonl", ["abcd"])
    class OOM(FakeCausalLM):
        def __call__(self, **kwargs):
            raise RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="减小 --batch_size.*--block_size"):
        run(args(data, tmp_path / "out", "--overwrite"), model=OOM(), tokenizer=FakeTokenizer())
