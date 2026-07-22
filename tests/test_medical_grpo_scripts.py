"""医疗 CMExam GRPO Shell 脚本的静态与 dry-run 测试。"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "scripts/run_medical_grpo_smoke.sh"
FORMAL = ROOT / "scripts/run_medical_grpo.sh"
SCRIPTS = (SMOKE, FORMAL)


def run_script(script: Path, tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "DRY_RUN": "1",
        "STORAGE_ROOT": str(tmp_path / "storage root"),
        "RUN_ID": "unit-run",
        "PT_SFT_ADAPTER_PATH": str(tmp_path / "formal sft" / "adapter"),
        **overrides,
    }
    return subprocess.run(
        ["bash", str(script)], cwd=ROOT, env=environment,
        text=True, capture_output=True, check=False,
    )


def command_from(result: subprocess.CompletedProcess[str]) -> list[str]:
    line = next(line for line in result.stdout.splitlines() if line.startswith("COMMAND: "))
    return shlex.split(line.removeprefix("COMMAND: "))


def option_map(command: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, token in enumerate(command):
        if token.startswith("--"):
            result[token] = command[index + 1]
    return result


@pytest.mark.parametrize("script", SCRIPTS)
def test_bash_syntax(script: Path) -> None:
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_dry_run_safe_and_does_not_create_dirs(script: Path, tmp_path: Path) -> None:
    result = run_script(script, tmp_path, PYTHON_BIN="python-that-must-not-run")
    assert result.returncode == 0, result.stderr
    assert "不启动Python" in result.stdout and "COMMAND:" in result.stdout
    assert not (tmp_path / "storage root").exists()
    assert "nvidia-smi" not in result.stdout + result.stderr
    command = command_from(result)
    assert command[:2] == ["python-that-must-not-run", "training/medical_grpo_training.py"]


def test_smoke_default_command_values(tmp_path: Path) -> None:
    result = run_script(SMOKE, tmp_path)
    options = option_map(command_from(result))
    assert options["--max_samples"] == "64"
    assert options["--max_steps"] == "10"
    assert options["--logging_steps"] == "1" and options["--save_steps"] == "5"
    assert "grpo_smoke" in options["--output_dir"]
    assert options["--run_name"] == "medical-cmexam-grpo-smoke-unit-run"


def test_formal_default_and_optional_max_samples(tmp_path: Path) -> None:
    default = command_from(run_script(FORMAL, tmp_path))
    options = option_map(default)
    assert "--max_samples" not in default
    assert options["--max_steps"] == "500"
    assert options["--logging_steps"] == "5" and options["--save_steps"] == "100"
    assert "/grpo/unit-run" in options["--output_dir"] and "grpo_smoke" not in options["--output_dir"]
    overridden = option_map(command_from(run_script(FORMAL, tmp_path, MAX_SAMPLES="123")))
    assert overridden["--max_samples"] == "123"


@pytest.mark.parametrize("script", SCRIPTS)
def test_common_protocol_command_values(script: Path, tmp_path: Path) -> None:
    command = command_from(run_script(script, tmp_path))
    options = option_map(command)
    expected = {
        "--torch_dtype": "bfloat16", "--local_files_only": "True",
        "--correct_reward": "1.0", "--incorrect_reward": "0.0",
        "--format_reward": "0.1", "--invalid_penalty": "-0.1",
        "--data_seed": "42", "--shuffle_before_select": "True",
        "--per_device_train_batch_size": "2", "--gradient_accumulation_steps": "4",
        "--learning_rate": "5e-6", "--warmup_ratio": "0.03", "--weight_decay": "0.01",
        "--num_generations": "2", "--max_completion_length": "32", "--beta": "0",
        "--bf16": "True", "--fp16": "False", "--gradient_checkpointing": "True",
        "--remove_unused_columns": "False", "--save_total_limit": "2",
        "--report_to": "tensorboard", "--seed": "42",
    }
    assert all(options[name] == value for name, value in expected.items())
    assert options["--peft_path"].endswith("formal sft/adapter")
    assert "/decontaminated/train/train.jsonl" in options["--train_file"]
    assert "--num_train_epochs" not in command and "--max_prompt_length" not in command
    assert int(options["--per_device_train_batch_size"]) * int(options["--gradient_accumulation_steps"]) % int(options["--num_generations"]) == 0
    assert int(options["--num_generations"]) >= 2 and int(options["--max_completion_length"]) > 0
    assert float(options["--learning_rate"]) > 0 and float(options["--beta"]) >= 0


@pytest.mark.parametrize("script", SCRIPTS)
def test_path_and_run_overrides_are_quoted(script: Path, tmp_path: Path) -> None:
    output = tmp_path / "custom output"
    log = tmp_path / "custom logs" / "my log.txt"
    result = run_script(
        script, tmp_path, RUN_ID="special-id", OUTPUT_DIR=str(output), LOG_FILE=str(log),
        MODEL_NAME_OR_PATH=str(tmp_path / "base model"),
    )
    options = option_map(command_from(result))
    assert options["--output_dir"] == str(output)
    assert options["--model_name_or_path"] == str(tmp_path / "base model")
    assert f"LOG_FILE={log}" in result.stdout and not output.exists() and not log.parent.exists()


@pytest.mark.parametrize("script", SCRIPTS)
def test_resume_and_overwrite_are_conditional(script: Path, tmp_path: Path) -> None:
    base = command_from(run_script(script, tmp_path))
    assert "--resume_from_checkpoint" not in base and "--overwrite_output_dir" not in base
    changed = option_map(command_from(run_script(
        script, tmp_path, RESUME_FROM_CHECKPOINT="/checkpoints/checkpoint-10", OVERWRITE_OUTPUT_DIR="1"
    )))
    assert changed["--resume_from_checkpoint"] == "/checkpoints/checkpoint-10"
    assert changed["--overwrite_output_dir"] == "True"


@pytest.mark.parametrize("bad_part", ["validation", "test"])
@pytest.mark.parametrize("script", SCRIPTS)
def test_forbidden_train_split_rejected(script: Path, bad_part: str, tmp_path: Path) -> None:
    path = tmp_path / f"processed/cmexam/decontaminated/{bad_part}/{bad_part}.jsonl"
    result = run_script(script, tmp_path, TRAIN_FILE=str(path))
    assert result.returncode != 0 and "validation和test禁止" in result.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_equal_or_nested_adapter_output_rejected(script: Path, tmp_path: Path) -> None:
    adapter = tmp_path / "same adapter"
    result = run_script(script, tmp_path, PT_SFT_ADAPTER_PATH=str(adapter), OUTPUT_DIR=str(adapter))
    assert result.returncode != 0 and "禁止原地覆盖" in result.stderr
    nested = run_script(
        script, tmp_path, PT_SFT_ADAPTER_PATH=str(tmp_path / "out/adapter"), OUTPUT_DIR=str(tmp_path / "out")
    )
    assert nested.returncode != 0 and "互相嵌套" in nested.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_dry_run_allows_placeholder_adapter(script: Path, tmp_path: Path) -> None:
    result = run_script(script, tmp_path, PT_SFT_ADAPTER_PATH="")
    assert result.returncode == 0
    assert option_map(command_from(result))["--peft_path"].endswith("PT_SFT_ADAPTER_PLACEHOLDER")


@pytest.mark.parametrize("script", SCRIPTS)
def test_real_run_missing_adapter_fails_before_gpu_check(script: Path, tmp_path: Path) -> None:
    result = run_script(script, tmp_path, DRY_RUN="0", PT_SFT_ADAPTER_PATH="")
    assert result.returncode != 0
    assert "PT_SFT_ADAPTER_PATH必须指向正式Base→PT→SFT" in result.stderr
    assert "nvidia-smi" not in result.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_nonempty_output_rejected_and_overwrite_allows_fake_run(script: Path, tmp_path: Path) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "fake-python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_nvidia = fake_bin / "nvidia-smi"
    fake_nvidia.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_nvidia.chmod(0o755)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    adapter = tmp_path / "formal_sft" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    train = tmp_path / "processed/cmexam/decontaminated/train/train.jsonl"
    train.parent.mkdir(parents=True)
    train.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "existing-output"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}", "DRY_RUN": "0",
        "PYTHON_BIN": str(fake_python), "MODEL_NAME_OR_PATH": str(base),
        "PT_SFT_ADAPTER_PATH": str(adapter), "TRAIN_FILE": str(train),
        "OUTPUT_DIR": str(output), "LOG_ROOT": str(tmp_path / "logs"),
        "CACHE_DIR": str(tmp_path / "cache"), "MIN_FREE_GB": "0",
    }
    rejected = subprocess.run(
        ["bash", str(script)], cwd=ROOT, env=environment,
        text=True, capture_output=True, check=False,
    )
    assert rejected.returncode != 0 and "OUTPUT_DIR非空" in rejected.stderr
    environment["OVERWRITE_OUTPUT_DIR"] = "1"
    allowed = subprocess.run(
        ["bash", str(script)], cwd=ROOT, env=environment,
        text=True, capture_output=True, check=False,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("script", SCRIPTS)
def test_unsupported_or_incompatible_overrides_fail(script: Path, tmp_path: Path) -> None:
    prompt = run_script(script, tmp_path, MAX_PROMPT_LENGTH="1024")
    assert prompt.returncode != 0 and "没有max_prompt_length" in prompt.stderr
    beta = run_script(script, tmp_path, BETA="0.04")
    assert beta.returncode != 0 and "第二个ref adapter" in beta.stderr
    columns = run_script(script, tmp_path, REMOVE_UNUSED_COLUMNS="True")
    assert columns.returncode != 0 and "必须为False" in columns.stderr
    batch = run_script(script, tmp_path, NUM_GENERATIONS="3")
    assert batch.returncode != 0 and "不能被NUM_GENERATIONS" in batch.stderr


def test_static_shell_engineering_and_forbidden_features() -> None:
    forbidden = {
        "eval", "torchrun", "bitsandbytes", "load_in_4bit", "load_in_8bit",
        "LoraConfig", "target_modules", "lora_rank", "lora_alpha", "--deepspeed", "--use_vllm",
    }
    for script in SCRIPTS:
        source = script.read_text(encoding="utf-8")
        assert source.startswith("#!/usr/bin/env bash\nset -Eeuo pipefail\nIFS=$'\\n\\t'")
        assert "CMD=(" in source and '"${CMD[@]}"' in source
        assert "PIPESTATUS[0]" in source and "CUDA_VISIBLE_DEVICES" in source
        assert "TOKENIZERS_PARALLELISM=false" in source
        assert "find" not in source.split("PT_SFT_ADAPTER_PATH=", 1)[0]
        assert not forbidden.intersection(source.split())


def test_all_command_options_exist_in_entrypoint_help(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, "training/medical_grpo_training.py", "--help"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert help_result.returncode == 0
    supported = set(re.findall(r"--[a-z][a-z0-9_]*", help_result.stdout))
    for script in SCRIPTS:
        command = command_from(run_script(script, tmp_path))
        used = {token for token in command if token.startswith("--")}
        assert used <= supported, sorted(used - supported)
