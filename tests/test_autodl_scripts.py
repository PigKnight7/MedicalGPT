"""AutoDL Shell脚本的离线静态测试。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
SETUP = ROOT / "scripts/setup_autodl.sh"
DOWNLOAD = ROOT / "scripts/download_qwen35_2b_base.sh"


def run_bash(script: Path, env: dict[str, str]):
    return subprocess.run(
        ["bash", str(script)], cwd=ROOT, env={**os.environ, **env},
        text=True, capture_output=True, check=False,
    )


def test_scripts_pass_bash_syntax() -> None:
    for script in (SETUP, DOWNLOAD):
        result = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr


def test_setup_dry_run_is_read_only(tmp_path: Path) -> None:
    storage = tmp_path / "never-created"
    result = run_bash(SETUP, {"DRY_RUN": "1", "STORAGE_ROOT": str(storage)})
    assert result.returncode == 0, result.stderr
    assert not storage.exists()
    assert "未创建目录" in result.stdout and "INSTALL_TORCH=0" in result.stdout


def test_download_dry_run_is_read_only(tmp_path: Path) -> None:
    model = tmp_path / "qwen-model"
    result = run_bash(DOWNLOAD, {"DRY_RUN": "1", "MODEL_DIR": str(model)})
    assert result.returncode == 0, result.stderr
    assert not model.exists() and not Path(str(model) + ".partial").exists()
    assert "hf download" in result.stdout and ".partial" in result.stdout


def test_scripts_have_required_safety_and_validation() -> None:
    setup = SETUP.read_text(encoding="utf-8")
    download = DOWNLOAD.read_text(encoding="utf-8")
    assert 'INSTALL_TORCH="${INSTALL_TORCH:-0}"' in setup
    assert "TORCH_INDEX_URL" in setup and "eval " not in setup
    assert "HF_TOKEN=" not in setup and "HF_TOKEN=" not in download
    assert "config.json" in download and "tokenizer" in download
    assert "*.safetensors" in download and "model.safetensors.index.json" in download
    assert "local_files_only=True" in download
    assert "PARTIAL_DIR" in download and ".backup." in download
    assert "rm -rf" not in download
