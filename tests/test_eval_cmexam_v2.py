from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluation import eval_cmexam_v2 as subject


def record(rid: str = "q1", labels: list[str] | None = None, split: str = "validation") -> dict:
    labels = labels or ["A"]
    return {"id": rid, "split": split, "question": "题干\nA. 甲\nB. 乙",
            "options": [{"label": "A", "text": "甲"}, {"label": "B", "text": "乙"}],
            "answer_labels": labels, "metadata": {"source": "fixture"}}


def write_split(tmp_path: Path, split: str, rows: list[dict]) -> Path:
    root = tmp_path / "official"
    path = root / split / f"{split}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return root


def args(tmp_path: Path, **changes) -> argparse.Namespace:
    values = dict(model_name_or_path="base", peft_path=None, data_root=tmp_path / "official",
                  split="validation", allow_test_evaluation=False, output_dir=tmp_path / "out",
                  max_samples=None, seed=42, batch_size=1, max_input_length=16, max_new_tokens=4,
                  device="cpu", torch_dtype="float32", cache_dir=None, trust_remote_code=False,
                  local_files_only=True, overwrite=False, dry_run=False, verify_adapter_effect=True,
                  adapter_check_samples=3, adapter_check_atol=0.0, skip_adapter_effect_check=False)
    values.update(changes)
    return argparse.Namespace(**values)


def test_validation_read_and_choice_types(tmp_path):
    root = write_split(tmp_path, "validation", [record(), record("q2", ["A", "B"])])
    rows = subject.load_cmexam_records(subject.resolve_data_file(root, "validation"), "validation")
    assert [r["is_multiple_choice"] for r in rows] == [False, True]
    assert rows[1]["answer_labels"] == ["A", "B"]


def test_test_is_gated_and_train_root_rejected(tmp_path):
    with pytest.raises(subject.EvaluationError, match="allow_test"):
        subject.validate_args(args(tmp_path, split="test"))
    subject.validate_args(args(tmp_path, split="test", allow_test_evaluation=True))
    with pytest.raises(subject.EvaluationError, match="禁止 train"):
        subject.resolve_data_file(tmp_path / "official" / "train", "validation")


def test_duplicate_id_and_bad_json_report_line(tmp_path):
    root = write_split(tmp_path, "validation", [record(), record()])
    with pytest.raises(subject.EvaluationError, match="id 重复"):
        subject.load_cmexam_records(subject.resolve_data_file(root, "validation"), "validation")
    path = root / "validation" / "validation.jsonl"
    path.write_text("{}\n{bad\n", encoding="utf-8")
    with pytest.raises(subject.EvaluationError, match="第 1 行|id"):
        subject.load_cmexam_records(path, "validation")


def test_sampling_is_seeded_without_replacement():
    rows = [record(str(i)) for i in range(20)]
    one = subject.select_records(rows, 5, 7)
    two = subject.select_records(rows, 5, 7)
    assert [x["id"] for x in one] == [x["id"] for x in two]
    assert len({x["id"] for x in one}) == 5
    assert [rows.index(x) for x in one] == sorted(rows.index(x) for x in one)


class FakeTokenizer:
    chat_template = "template"
    eos_token_id = 9
    eos_token = "</s>"
    pad_token_id = 0
    padding_side = "right"

    def __init__(self):
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "PROMPT:" + messages[-1]["content"]

    def __call__(self, text, **kwargs):
        batch = text if isinstance(text, list) else [text]
        width = min(kwargs.get("max_length", 4), 4)
        ids = torch.tensor([[1, 2, 3, 4][-width:] for _ in batch])
        result = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return result if kwargs.get("return_tensors") == "pt" else {"input_ids": ids[0].tolist()}

    def decode(self, ids, **kwargs):
        return "<answer>A</answer>"


def test_prompt_reuses_public_formatter_and_disables_thinking(monkeypatch):
    tokenizer = FakeTokenizer()
    called = {}
    real = subject.format_cmexam_prompt

    def spy(tok, question, **kwargs):
        called["yes"] = True
        return real(tok, question, **kwargs)

    monkeypatch.setattr(subject, "format_cmexam_prompt", spy)
    prompt = subject.build_prompt(tokenizer, "题目（不含答案）")
    assert called == {"yes": True}
    assert tokenizer.template_kwargs == {"tokenize": False, "add_generation_prompt": True,
                                          "enable_thinking": False}
    assert "题目" in prompt


class FakeBase:
    def __init__(self):
        self.eval_called = self.to_device = False
        self.generate_calls = 0

    def to(self, device): self.to_device = device; return self
    def eval(self): self.eval_called = True; return self


def test_base_loader_uses_causal_lm(monkeypatch, tmp_path):
    base = FakeBase(); calls = {}
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda name, **kw: calls.update(name=name, **kw) or base))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    model = subject.load_base_model(args(tmp_path))
    assert model is base and base.eval_called and base.to_device == "cpu"
    assert calls["name"] == "base" and calls["local_files_only"] is True
    assert calls["torch_dtype"] is torch.float32 and "device_map" not in calls


class FakePeft:
    returned = None
    from_kwargs = None

    @classmethod
    def from_pretrained(cls, base, path, **kwargs):
        cls.from_kwargs = kwargs
        cls.returned = cls(base)
        return cls.returned

    def __init__(self, base):
        self.base = base; self.peft_config = {"default": {}}; self.active_adapters = []
        self.eval_called = False; self.generate_calls = 0; self.disabled = False

    def set_adapter(self, name, inference_mode=False): self.active_adapters = [name]; self.inference_mode = inference_mode
    def eval(self): self.eval_called = True; return self
    def parameters(self): return iter([torch.nn.Parameter(torch.zeros(2), requires_grad=False)])
    def generate(self, **kwargs): self.generate_calls += 1; return torch.cat([kwargs["input_ids"], torch.tensor([[5, 6]])], 1)
    def __call__(self, **kwargs):
        value = 0.0 if self.disabled else 2.0
        return SimpleNamespace(logits=torch.tensor([[[value, 1.0, 0.0]]]))
    def disable_adapter(self):
        parent = self
        class Context:
            def __enter__(self): parent.disabled = True
            def __exit__(self, *unused): parent.disabled = False
        return Context()


def make_adapter(tmp_path: Path) -> Path:
    path = tmp_path / "adapter"; path.mkdir()
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"x")
    return path


def test_adapter_return_value_is_used_and_activated(monkeypatch, tmp_path):
    monkeypatch.setattr(subject, "load_base_model", lambda unused: FakeBase())
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeft))
    model, info = subject.load_model_with_optional_adapter(args(tmp_path, peft_path=make_adapter(tmp_path)))
    assert model is FakePeft.returned and model.eval_called
    assert FakePeft.from_kwargs["adapter_name"] == "default"
    assert FakePeft.from_kwargs["is_trainable"] is False
    assert model.inference_mode is True
    assert info["active_adapters"] == ["default"]


def test_adapter_files_are_required(tmp_path):
    path = tmp_path / "adapter"; path.mkdir()
    with pytest.raises(subject.EvaluationError, match="adapter_config"):
        subject.validate_adapter_path(path)


def test_adapter_logits_check_passes_and_restores():
    model = FakePeft(FakeBase()); model.set_adapter("default")
    result = subject.verify_adapter_changes_logits(model, FakeTokenizer(), [record()], device="cpu",
                                                    sample_count=1, atol=0.0, adapter_path="adapter")
    assert result["passed"] is True and result["samples"][0]["max_abs_diff"] == 2.0
    assert result["samples"][0]["mean_abs_diff"] == pytest.approx(2 / 3)
    assert result["samples"][0]["nonzero_count"] == 1
    assert model.active_adapters == ["default"] and model.disabled is False


def test_adapter_logits_check_rejects_identical():
    model = FakePeft(FakeBase()); model.set_adapter("default")
    model.__call__ = lambda **kw: SimpleNamespace(logits=torch.ones((1, 1, 3)))
    # Special method lookup is on the class; use a subclass with identical outputs.
    class Same(FakePeft):
        def __call__(self, **kwargs): return SimpleNamespace(logits=torch.ones((1, 1, 3)))
    same = Same(FakeBase()); same.set_adapter("default")
    with pytest.raises(subject.EvaluationError, match="差异"):
        subject.verify_adapter_changes_logits(same, FakeTokenizer(), [record()], device="cpu",
                                              sample_count=1, atol=0.0, adapter_path="adapter")


def test_generate_uses_final_model_and_only_decodes_new_tokens(tmp_path):
    model = FakePeft(FakeBase()); model.set_adapter("default")
    cfg = args(tmp_path, peft_path=Path("adapter"))
    rows = subject.generate_predictions(model, FakeTokenizer(), [record()], cfg,
                                        progress_factory=lambda it, **kw: it)
    assert model.generate_calls == 1 and model.base.generate_calls == 0
    assert rows[0]["raw_output"] == "<answer>A</answer>"
    assert rows[0]["generated_tokens"] == 2 and rows[0]["correct"] is True


def prediction(rid, gold, pred, *, valid=True, fmt=True, truncated=False):
    return {"id": rid, "gold_labels": list(gold), "gold_answer": "".join(gold),
            "predicted_answer": "".join(pred), "valid_prediction": valid, "format_valid": fmt,
            "parse_error": None if valid else "bad", "correct": set(gold) == set(pred),
            "is_multiple_choice": len(gold) > 1, "input_tokens": 10, "generated_tokens": 2,
            "latency_seconds": .1, "input_truncated": truncated}


def test_metrics_use_exact_set_and_groups():
    rows = [prediction("a", ["A"], ["A"]), prediction("b", ["A", "B"], ["B", "A"]),
            prediction("c", ["A", "B"], ["A"], valid=False, fmt=False, truncated=True)]
    metrics = subject.compute_metrics(rows, 3.0)
    assert metrics["exact_match_accuracy"] == pytest.approx(2 / 3)
    assert metrics["multiple_choice"]["exact_set_accuracy"] == .5
    assert metrics["by_gold_label_count"]["2"]["records"] == 2
    assert metrics["invalid_error_distribution"] == {"bad": 1}
    assert metrics["truncated_input_count"] == 1


def test_output_policy_and_overwrite_only_known_files(tmp_path):
    out = tmp_path / "out"; out.mkdir(); (out / "foreign.txt").write_text("keep")
    with pytest.raises(subject.EvaluationError, match="非空"):
        subject.prepare_output_dir(out, False, create=False)
    (out / "metrics.json").write_text("old")
    subject.prepare_output_dir(out, True, create=True)
    assert (out / "foreign.txt").read_text() == "keep" and not (out / "metrics.json").exists()


def test_dry_run_reads_data_but_loads_no_model(monkeypatch, tmp_path, capsys):
    root = write_split(tmp_path, "validation", [record(), record("q2", ["A", "B"])])
    monkeypatch.setattr(subject, "load_tokenizer", lambda unused: pytest.fail("tokenizer loaded"))
    monkeypatch.setattr(subject, "load_model_with_optional_adapter", lambda unused: pytest.fail("model loaded"))
    result = subject.run(args(tmp_path, data_root=root, dry_run=True, max_samples=1))
    assert result == 0 and not (tmp_path / "out").exists()
    output = capsys.readouterr().out
    assert '"records": 1' in output and "answer_labels" not in output


def test_help_works():
    result = subprocess.run([sys.executable, "evaluation/eval_cmexam_v2.py", "--help"],
                            cwd=Path(__file__).parents[1], text=True, capture_output=True)
    assert result.returncode == 0 and "--verify_adapter_effect" in result.stdout
