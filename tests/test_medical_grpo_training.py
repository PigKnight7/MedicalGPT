"""医疗 CMExam GRPO 训练入口测试；不加载真实模型、不训练、不访问网络。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset

import training.medical_grpo_training as module
from medicalgpt_ext.cmexam_grpo_data import CMExamGRPODataSummary
from training.medical_grpo_training import (
    CMExamGRPODataArguments,
    CMExamGRPORuntimeArguments,
    CMExamRewardArguments,
    MedicalGRPOConfig,
    MedicalGRPOError,
    ModelArguments,
    TrainableParameterSummary,
    build_grpo_trainer,
    build_reward_functions,
    configure_thinking_disabled,
    load_processing_class,
    load_trainable_peft_model,
    parse_args,
    prepare_training_dataset,
    resolve_resume_checkpoint,
    resolve_torch_dtype,
    run_training,
    validate_runtime_arguments,
    validate_trainable_adapter,
)


def data_record(index: int = 1, answer: list[str] | None = None) -> dict[str, object]:
    labels = answer or ["A"]
    return {
        "id": f"cmexam-train-{index:06d}",
        "split": "train",
        "stem": f"题目{index} GOLD-NOT-IN-PROMPT",
        "question": f"题目{index}",
        "options": [
            {"label": "A", "text": "甲"},
            {"label": "B", "text": "乙"},
            {"label": "C", "text": "丙"},
            {"label": "D", "text": "丁"},
            {"label": "E", "text": "戊"},
        ],
        "answer": "".join(labels),
        "answer_labels": labels,
        "answer_texts": ["SECRET-GOLD-TEXT"],
        "is_multiple_choice": len(labels) > 1,
        "metadata": {},
    }


def write_strict_data(tmp_path: Path, count: int = 4) -> Path:
    path = tmp_path / "cmexam/decontaminated/train/train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(data_record(i), ensure_ascii=False) + "\n" for i in range(1, count + 1)),
        encoding="utf-8",
    )
    return path


def model_args(**updates: object) -> ModelArguments:
    values: dict[str, object] = {"model_name_or_path": "Qwen/Qwen3.5-2B-Base", "peft_path": "/adapter"}
    values.update(updates)
    return ModelArguments(**values)  # type: ignore[arg-type]


def runtime(**updates: object) -> CMExamGRPORuntimeArguments:
    values = {"dry_run": True}
    values.update(updates)
    return CMExamGRPORuntimeArguments(**values)


def training(tmp_path: Path, **updates: object) -> MedicalGRPOConfig:
    values: dict[str, object] = {
        "output_dir": str(tmp_path / "output"),
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_generations": 2,
        "max_completion_length": 32,
        "remove_unused_columns": False,
        "bf16": False,
        "report_to": "none",
        "beta": 0.0,
        "gradient_checkpointing": False,
    }
    values.update(updates)
    if values.get("bf16") is True and "use_cpu" not in updates:
        values["use_cpu"] = True
    return MedicalGRPOConfig(**values)


def data_args(path: Path, **updates: object) -> CMExamGRPODataArguments:
    values: dict[str, object] = {"train_file": str(path)}
    values.update(updates)
    return CMExamGRPODataArguments(**values)  # type: ignore[arg-type]


def test_cli_help_and_defaults() -> None:
    result = subprocess.run(
        [sys.executable, "training/medical_grpo_training.py", "--help"],
        cwd=Path(__file__).parents[1], text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0
    assert "--model_name_or_path" in result.stdout and "--dry_run" in result.stdout
    assert CMExamGRPODataArguments().train_file.endswith("decontaminated/train/train.jsonl")


def test_parse_args_required_model_and_boolean_syntax(tmp_path: Path) -> None:
    with pytest.raises((SystemExit, MedicalGRPOError)):
        parse_args(["--output_dir", str(tmp_path)])
    parsed = parse_args([
        "--model_name_or_path", "base", "--output_dir", str(tmp_path / "out"),
        "--dry_run", "True", "--data_seed", "9", "--num_generations", "2",
        "--per_device_train_batch_size", "2", "--remove_unused_columns", "False",
    ])
    assert parsed[0].model_name_or_path == "base"
    assert parsed[1].data_seed == 9 and parsed[3].dry_run is True
    assert parsed[4].remove_unused_columns is False


def test_dtype_resolution_and_invalid_model() -> None:
    import torch

    assert resolve_torch_dtype("auto") == "auto"
    assert resolve_torch_dtype("bfloat16") is torch.bfloat16
    assert resolve_torch_dtype("float16") is torch.float16
    assert resolve_torch_dtype("float32") is torch.float32
    with pytest.raises(MedicalGRPOError, match="torch_dtype"):
        resolve_torch_dtype("int8")
    with pytest.raises(MedicalGRPOError, match="model_name_or_path"):
        ModelArguments()


def test_runtime_requires_peft_only_for_real_training(tmp_path: Path) -> None:
    path = write_strict_data(tmp_path)
    args = model_args(peft_path=None)
    validate_runtime_arguments(args, data_args(path), CMExamRewardArguments(), runtime(), training(tmp_path))
    with pytest.raises(MedicalGRPOError, match="peft_path"):
        validate_runtime_arguments(
            args, data_args(path), CMExamRewardArguments(), runtime(dry_run=False),
            training(tmp_path, bf16=True),
        )


def test_runtime_protocol_and_numeric_errors(tmp_path: Path) -> None:
    path = write_strict_data(tmp_path)
    base = (model_args(), data_args(path), CMExamRewardArguments(), runtime())
    cases = [
        ({"beta": -0.1}, "beta"),
        ({"beta": 0.1}, "ref adapter"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"max_completion_length": 0}, "max_completion_length"),
        ({"remove_unused_columns": True}, "remove_unused_columns"),
        ({"use_vllm": True}, "use_vllm"),
    ]
    for kwargs, match in cases:
        with pytest.raises((MedicalGRPOError, ValueError), match=match):
            cfg = training(tmp_path, **kwargs)
            validate_runtime_arguments(*base, cfg)
    mixed = training(tmp_path)
    mixed.bf16 = True
    mixed.fp16 = True
    with pytest.raises(MedicalGRPOError, match="同时"):
        validate_runtime_arguments(*base, mixed)
    # 当前 TRL 在 GRPOConfig 构造阶段直接执行 generation/batch 约束。
    with pytest.raises(ValueError, match="num_generations"):
        training(tmp_path, num_generations=3)
    with pytest.raises(ValueError, match="at least 2"):
        training(tmp_path, num_generations=1)


def test_invalid_reward_config() -> None:
    with pytest.raises(ValueError):
        build_reward_functions(CMExamRewardArguments(correct_reward=0, incorrect_reward=0))
    functions, config = build_reward_functions(CMExamRewardArguments(2, -1, 0.2, -0.3))
    assert [fn.__name__ for fn in functions] == [
        "exact_set_correctness_reward", "answer_only_format_reward", "invalid_answer_penalty"
    ]
    assert (config.correct_reward, config.invalid_penalty) == (2.0, -0.3)


@pytest.mark.parametrize("part", ["validation", "test"])
def test_validation_and_test_paths_rejected(tmp_path: Path, part: str) -> None:
    path = tmp_path / f"cmexam/decontaminated/{part}/{part}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data_record()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="validation 和 test"):
        prepare_training_dataset(data_args(path))


def test_official_train_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cmexam/official/train/train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data_record()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="decontaminated/train"):
        prepare_training_dataset(data_args(path))


def test_output_directory_resume_overwrite_and_same_adapter(tmp_path: Path) -> None:
    path = write_strict_data(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    (out / "existing").write_text("x")
    cfg = training(tmp_path)
    with pytest.raises(FileExistsError):
        validate_runtime_arguments(model_args(), data_args(path), CMExamRewardArguments(), runtime(), cfg)
    validate_runtime_arguments(
        model_args(), data_args(path), CMExamRewardArguments(),
        runtime(overwrite_output_dir=True), cfg,
    )
    checkpoint = out / "checkpoint-1"
    checkpoint.mkdir()
    cfg.resume_from_checkpoint = str(checkpoint)
    assert validate_runtime_arguments(
        model_args(), data_args(path), CMExamRewardArguments(), runtime(), cfg
    ) == str(checkpoint)
    with pytest.raises(MedicalGRPOError, match="不能与"):
        validate_runtime_arguments(
            model_args(peft_path=str(out)), data_args(path), CMExamRewardArguments(), runtime(), cfg
        )


def test_resume_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    checkpoint = out / "checkpoint-20"
    checkpoint.mkdir()
    monkeypatch.setattr(module, "get_last_checkpoint", lambda _: str(checkpoint))
    assert resolve_resume_checkpoint("latest", out) == str(checkpoint)
    assert resolve_resume_checkpoint(str(checkpoint), out) == str(checkpoint)
    with pytest.raises(MedicalGRPOError, match="不存在"):
        resolve_resume_checkpoint(str(out / "missing"), out)


def test_data_module_connection_and_sampling(tmp_path: Path) -> None:
    path = write_strict_data(tmp_path, 7)
    dataset, summary = prepare_training_dataset(data_args(path, max_samples=3, data_seed=17))
    assert len(dataset) == summary.selected_records == 3
    assert {"id", "prompt", "answer_labels"}.issubset(dataset.column_names)
    assert summary.seed == 17 and summary.max_samples == 3
    prompt = json.dumps(dataset[0]["prompt"], ensure_ascii=False)
    assert "SECRET-GOLD-TEXT" not in prompt


def test_data_config_is_strictly_connected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    summary = CMExamGRPODataSummary("x", 1, 1, 1, 0, 1, 0, {"1": 1}, ("id",), 8, 1, True, False)

    def fake_prepare(config):
        captured.update(config.__dict__)
        return ([{"id": "id", "prompt": "p", "answer": "A", "answer_labels": ["A"],
                  "is_multiple_choice": False, "metadata": {}, "question": "q", "options": []}], summary)

    monkeypatch.setattr(module, "prepare_cmexam_grpo_examples", fake_prepare)
    dataset, _ = prepare_training_dataset(
        CMExamGRPODataArguments("x", max_samples=1, data_seed=8, shuffle_before_select=False)
    )
    assert len(dataset) == 1
    assert captured["require_decontaminated_train"] is True
    assert (captured["max_samples"], captured["seed"], captured["shuffle_before_select"]) == (1, 8, False)


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token = None
    chat_template = "template"
    padding_side = "right"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, prompt, **kwargs):
        self.calls.append(kwargs)
        return "rendered"


def test_tokenizer_loading_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = FakeTokenizer()
    captured: dict[str, object] = {}

    def fake_load(source, **kwargs):
        captured["source"] = source
        captured.update(kwargs)
        return tokenizer

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", fake_load)
    result = load_processing_class(model_args(tokenizer_name_or_path="tok", local_files_only=True,
                                               trust_remote_code=True, use_fast_tokenizer=True))
    assert result is tokenizer and tokenizer.pad_token == "<eos>" and tokenizer.padding_side == "left"
    assert captured == {
        "source": "tok", "cache_dir": None, "trust_remote_code": True,
        "local_files_only": True, "use_fast": True, "padding_side": "left",
    }


def test_tokenizer_requires_eos_and_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    tokenizer.eos_token = None
    with pytest.raises(MedicalGRPOError, match="eos_token"):
        load_processing_class(model_args())
    tokenizer.eos_token = "eos"
    tokenizer.chat_template = None
    with pytest.raises(MedicalGRPOError, match="chat_template"):
        load_processing_class(model_args())


def test_thinking_current_strategy_and_conflict(tmp_path: Path) -> None:
    dataset = Dataset.from_list([{"id": "x", "prompt": [{"role": "user", "content": "q"}], "answer_labels": ["A"]}])
    args = training(tmp_path)
    returned, strategy = configure_thinking_disabled(args, dataset=dataset)
    assert returned is dataset and args.chat_template_kwargs == {"enable_thinking": False}
    assert "enable_thinking=False" in strategy
    args.chat_template_kwargs = {"enable_thinking": True}
    with pytest.raises(MedicalGRPOError, match="启用 thinking"):
        configure_thinking_disabled(args, dataset=dataset)


def test_thinking_rendered_fallback_preserves_fields() -> None:
    class NoChatKwargs:
        pass

    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list([{
        "id": "x", "prompt": [{"role": "user", "content": "q"}],
        "answer_labels": ["A"], "metadata": {"x": "y"},
    }])
    rendered, strategy = configure_thinking_disabled(
        NoChatKwargs(), tokenizer=tokenizer, dataset=dataset
    )
    assert rendered is not None and rendered[0]["prompt"] == "rendered"
    assert rendered[0]["answer_labels"] == ["A"] and rendered[0]["id"] == "x"
    assert tokenizer.calls == [{"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}]
    assert "预渲染" in strategy
    with pytest.raises(MedicalGRPOError, match="无法确认"):
        configure_thinking_disabled(NoChatKwargs())


class FakeParameter:
    def __init__(self, size: int, requires_grad: bool) -> None:
        self.size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.size


class FakePeftModel:
    def __init__(self, params=None, configs=None) -> None:
        self.peft_config = {"default": object()} if configs is None else configs
        self.params = params or [
            ("base.weight", FakeParameter(1000, False)),
            ("base.lora_A.default.weight", FakeParameter(10, True)),
        ]
        self.config = SimpleNamespace(use_cache=True)
        self.set_calls: list[str] = []
        self.input_grads = False

    def named_parameters(self):
        return iter(self.params)

    def set_adapter(self, name: str) -> None:
        self.set_calls.append(name)

    def enable_input_require_grads(self) -> None:
        self.input_grads = True


def test_validate_trainable_adapter_boundaries() -> None:
    summary = validate_trainable_adapter(FakePeftModel())
    assert summary == TrainableParameterSummary(1010, 10, 10 / 1010, ("base.lora_A.default.weight",))
    with pytest.raises(MedicalGRPOError, match="default adapter 配置"):
        validate_trainable_adapter(FakePeftModel(configs={}))
    with pytest.raises(MedicalGRPOError, match="没有任何"):
        validate_trainable_adapter(FakePeftModel(params=[("base", FakeParameter(4, False))]))
    with pytest.raises(MedicalGRPOError, match="意外解冻"):
        validate_trainable_adapter(FakePeftModel(params=[
            ("base.weight", FakeParameter(100, True)),
            ("x.default.weight", FakeParameter(2, True)),
        ]))


def create_adapter(path: Path, base: str = "Qwen/Qwen3.5-2B-Base") -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base}), encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_bytes(b"fake")


def test_adapter_file_checks(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(MedicalGRPOError, match="配置缺失"):
        load_trainable_peft_model(model_args(peft_path=str(adapter)), training(tmp_path))
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(MedicalGRPOError, match="权重缺失"):
        load_trainable_peft_model(model_args(peft_path=str(adapter)), training(tmp_path))


def test_load_existing_adapter_exact_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    create_adapter(adapter)
    base = object()
    peft = FakePeftModel()
    calls: dict[str, object] = {}

    def base_load(path, **kwargs):
        calls["base"] = (path, kwargs)
        return base

    def peft_load(base_arg, path, **kwargs):
        calls["peft"] = (base_arg, path, kwargs)
        return peft

    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", base_load)
    monkeypatch.setattr(module.PeftModel, "from_pretrained", peft_load)
    args = model_args(peft_path=str(adapter), local_files_only=True, trust_remote_code=True)
    result, stats = load_trainable_peft_model(args, training(tmp_path, gradient_checkpointing=True))
    assert result is peft and stats.trainable_parameters == 10
    assert calls["base"][0] == "Qwen/Qwen3.5-2B-Base"
    base_kwargs = calls["base"][1]
    assert base_kwargs["local_files_only"] is True and base_kwargs["trust_remote_code"] is True
    assert "device_map" not in base_kwargs
    peft_call = calls["peft"]
    assert peft_call[0] is base and peft_call[2]["adapter_name"] == "default"
    assert peft_call[2]["is_trainable"] is True
    assert peft.set_calls == ["default"] and peft.config.use_cache is False and peft.input_grads


def test_adapter_base_mismatch_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    create_adapter(adapter, "Other/Base-7B")
    with pytest.raises(MedicalGRPOError, match="不兼容"):
        load_trainable_peft_model(model_args(peft_path=str(adapter)), training(tmp_path))


def test_trainer_exact_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "GRPOTrainer", FakeTrainer)
    dataset = Dataset.from_list([{"id": "x", "prompt": "q", "answer_labels": ["A"]}])
    functions, _ = build_reward_functions(CMExamRewardArguments())
    cfg = training(tmp_path, seed=17, beta=0.0, num_generations=2)
    model = FakePeftModel()
    tokenizer = FakeTokenizer()
    trainer = build_grpo_trainer(
        model=model, training_args=cfg, train_dataset=dataset,
        reward_functions=functions, processing_class=tokenizer,
    )
    assert isinstance(trainer, FakeTrainer)
    assert captured == {
        "model": model, "args": cfg, "train_dataset": dataset,
        "reward_funcs": functions, "processing_class": tokenizer,
    }
    assert cfg.seed == 17 and cfg.beta == 0 and cfg.num_generations == 2
    assert "peft_config" not in captured and "eval_dataset" not in captured and "ref_model" not in captured


def test_dry_run_no_loading_or_trainer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    path = write_strict_data(tmp_path, 5)
    forbidden = lambda *a, **k: pytest.fail("dry run 不得加载或创建训练对象")
    monkeypatch.setattr(module, "load_processing_class", forbidden)
    monkeypatch.setattr(module, "load_trainable_peft_model", forbidden)
    monkeypatch.setattr(module, "build_grpo_trainer", forbidden)
    result = run_training(
        model_args(peft_path="cloud/adapter-placeholder"),
        data_args(path, max_samples=2, data_seed=42),
        CMExamRewardArguments(), runtime(), training(tmp_path),
    )
    output = capsys.readouterr().out
    assert result == 0 and '"selected_records": 2' in output
    assert "exact_set_correctness_reward" in output and "enable_thinking=False" in output
    assert "SECRET-GOLD-TEXT" not in output and '"answer_labels"' not in output
    assert not Path(training(tmp_path).output_dir).exists()


class FakeTrainResult:
    metrics = {"loss": 0.5}


class RecordingTrainer:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.calls: list[tuple[object, ...]] = []

    def train(self, resume_from_checkpoint=None):
        self.calls.append(("train", resume_from_checkpoint))
        return FakeTrainResult()

    def log_metrics(self, *args): self.calls.append(("log_metrics", *args))
    def save_metrics(self, *args): self.calls.append(("save_metrics", *args))

    def save_state(self):
        self.calls.append(("save_state",))
        (self.output / "trainer_state.json").write_text("{}", encoding="utf-8")

    def save_model(self, output):
        self.calls.append(("save_model", output))
        (self.output / "adapter_config.json").write_text("{}", encoding="utf-8")
        (self.output / "adapter_model.safetensors").write_bytes(b"x")


def test_training_calls_and_saves_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = write_strict_data(tmp_path, 3)
    out = tmp_path / "output"
    trainer = RecordingTrainer(out)
    tokenizer = FakeTokenizer()
    tokenizer.saved = []
    tokenizer.save_pretrained = lambda target: tokenizer.saved.append(target)
    monkeypatch.setattr(module, "load_processing_class", lambda _: tokenizer)
    monkeypatch.setattr(
        module, "load_trainable_peft_model",
        lambda *a: (FakePeftModel(), TrainableParameterSummary(100, 10, 0.1, ("x.default.w",))),
    )
    monkeypatch.setattr(module, "build_grpo_trainer", lambda **kwargs: trainer)
    monkeypatch.setattr(module, "set_seed", lambda seed: None)
    cfg = training(tmp_path, bf16=True)
    result = run_training(
        model_args(), data_args(path), CMExamRewardArguments(), runtime(dry_run=False), cfg
    )
    assert result == 0
    assert trainer.calls[0] == ("train", None)
    assert [call[0] for call in trainer.calls[1:]] == [
        "log_metrics", "save_metrics", "save_state", "save_model"
    ]
    assert tokenizer.saved == [str(out.resolve())]
    assert (out / module.DATA_SUMMARY_FILENAME).is_file()
    assert (out / module.RUN_CONFIG_FILENAME).is_file()
    run_config = json.loads((out / module.RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert run_config["thinking_disabled_strategy"].endswith("enable_thinking=False")


def test_resume_passed_and_runtime_errors_not_swallowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = write_strict_data(tmp_path)
    checkpoint = tmp_path / "output/checkpoint-1"
    checkpoint.mkdir(parents=True)

    class FailingTrainer:
        def train(self, **kwargs):
            assert kwargs["resume_from_checkpoint"] == str(checkpoint)
            raise RuntimeError("ordinary failure")

    monkeypatch.setattr(module, "load_processing_class", lambda _: FakeTokenizer())
    monkeypatch.setattr(
        module, "load_trainable_peft_model",
        lambda *a: (FakePeftModel(), TrainableParameterSummary(100, 10, 0.1, ("x.default.w",))),
    )
    monkeypatch.setattr(module, "build_grpo_trainer", lambda **kwargs: FailingTrainer())
    cfg = training(tmp_path, bf16=True, resume_from_checkpoint=str(checkpoint))
    with pytest.raises(RuntimeError, match="ordinary failure"):
        run_training(model_args(), data_args(path), CMExamRewardArguments(), runtime(dry_run=False), cfg)
    assert not (tmp_path / "output" / module.RUN_CONFIG_FILENAME).exists()


def test_cuda_oom_has_clear_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = write_strict_data(tmp_path)

    class OOMTrainer:
        def train(self, **kwargs):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(module, "load_processing_class", lambda _: FakeTokenizer())
    monkeypatch.setattr(
        module, "load_trainable_peft_model",
        lambda *a: (FakePeftModel(), TrainableParameterSummary(100, 10, 0.1, ("x.default.w",))),
    )
    monkeypatch.setattr(module, "build_grpo_trainer", lambda **kwargs: OOMTrainer())
    with pytest.raises(RuntimeError, match="显存不足"):
        run_training(
            model_args(), data_args(path), CMExamRewardArguments(), runtime(dry_run=False),
            training(tmp_path, bf16=True),
        )
