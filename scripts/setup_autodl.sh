#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

CURRENT_STAGE="初始化"
trap 'status=$?; echo "[ERROR] 阶段=${CURRENT_STAGE} 行号=${LINENO} 命令=${BASH_COMMAND}" >&2; exit "${status}"' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -d /root/autodl-tmp ]]; then DEFAULT_STORAGE_ROOT=/root/autodl-tmp; else DEFAULT_STORAGE_ROOT="${HOME}"; fi
STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
MODELS_DIR="${MODELS_DIR:-${STORAGE_ROOT}/models}"
DATA_ROOT="${DATA_ROOT:-${STORAGE_ROOT}/datasets/medicalgpt}"
ENV_DIR="${ENV_DIR:-${STORAGE_ROOT}/venvs/medicalgpt}"
CACHE_DIR="${CACHE_DIR:-${STORAGE_ROOT}/cache/huggingface}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/medicalgpt}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/medicalgpt}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${STORAGE_ROOT}/evaluations/medicalgpt}"
ENVIRONMENT_DIR="${ENVIRONMENT_DIR:-${STORAGE_ROOT}/environment/medicalgpt}"
RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-${ENVIRONMENT_DIR}/runtime_env.sh}"
LOCK_FILE="${LOCK_FILE:-${ENVIRONMENT_DIR}/requirements-lock-autodl.txt}"
REPORT_FILE="${REPORT_FILE:-${ENVIRONMENT_DIR}/autodl_environment.json}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${PROJECT_ROOT}/requirements.txt}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DRY_RUN="${DRY_RUN:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
CREATE_VENV="${CREATE_VENV:-1}"
USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES:-1}"
INSTALL_TORCH="${INSTALL_TORCH:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
RUN_TESTS="${RUN_TESTS:-1}"

quote_command() { printf '%q ' "$@"; printf '\n'; }
stage() { CURRENT_STAGE="$1"; printf '\n[%s]\n' "${CURRENT_STAGE}"; }

if [[ "${DRY_RUN}" == 1 ]]; then
    stage "DRY_RUN配置"
    printf 'PROJECT_ROOT=%q\nSTORAGE_ROOT=%q\nENV_DIR=%q\nREQUIREMENTS_FILE=%q\n' \
        "${PROJECT_ROOT}" "${STORAGE_ROOT}" "${ENV_DIR}" "${REQUIREMENTS_FILE}"
    printf '模式: CHECK_ONLY=%s CREATE_VENV=%s INSTALL_DEPS=%s INSTALL_TORCH=%s RUN_TESTS=%s\n' \
        "${CHECK_ONLY}" "${CREATE_VENV}" "${INSTALL_DEPS}" "${INSTALL_TORCH}" "${RUN_TESTS}"
    quote_command mkdir -p "${MODELS_DIR}" "${DATA_ROOT}" "${CACHE_DIR}" "${OUTPUT_ROOT}" "${LOG_ROOT}" "${EVALUATION_ROOT}" "${ENVIRONMENT_DIR}"
    if [[ "${CREATE_VENV}" == 1 ]]; then
        VENV_CMD=("${PYTHON_BIN}" -m venv)
        [[ "${USE_SYSTEM_SITE_PACKAGES}" == 1 ]] && VENV_CMD+=(--system-site-packages)
        VENV_CMD+=("${ENV_DIR}")
        quote_command "${VENV_CMD[@]}"
    fi
    printf '将检查CUDA版PyTorch和BF16；已有可用torch时保持不变。\n'
    if [[ "${INSTALL_TORCH}" == 1 ]]; then
        [[ -n "${TORCH_INDEX_URL}" ]] || printf '警告：INSTALL_TORCH=1但未提供TORCH_INDEX_URL，正式运行会拒绝。\n'
        quote_command "${ENV_DIR}/bin/python" -m pip install torch --index-url "${TORCH_INDEX_URL:-<必须显式提供>}"
    fi
    quote_command "${ENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}" pytest
    quote_command "${ENV_DIR}/bin/python" -m pip check
    quote_command "${ENV_DIR}/bin/python" -m pytest -q
    printf '将原子写入：%s、%s、%s\n' "${RUNTIME_ENV_FILE}" "${LOCK_FILE}" "${REPORT_FILE}"
    printf 'DRY_RUN完成：未创建目录、环境或文件，未安装依赖，未访问网络。\n'
    exit 0
fi

stage "基础检查"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "错误：找不到Python：${PYTHON_BIN}" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import sys; print("Python:", sys.version); print("Executable:", sys.executable)'
[[ -f "${REQUIREMENTS_FILE}" ]] || { echo "错误：依赖文件不存在：${REQUIREMENTS_FILE}" >&2; exit 1; }
echo "Requirements: ${REQUIREMENTS_FILE}"

if [[ "${CHECK_ONLY}" == 1 ]]; then
    ACTIVE_PYTHON="${PYTHON_BIN}"
else
    stage "目录准备"
    mkdir -p \
        "${MODELS_DIR}" "${DATA_ROOT}" "${CACHE_DIR}" "${OUTPUT_ROOT}/pt" "${OUTPUT_ROOT}/sft" \
        "${LOG_ROOT}/pt" "${LOG_ROOT}/sft" "${EVALUATION_ROOT}/pt" "${EVALUATION_ROOT}/sft" \
        "${EVALUATION_ROOT}/cmexam" "${ENVIRONMENT_DIR}"

    ACTIVE_PYTHON="${PYTHON_BIN}"
    if [[ "${CREATE_VENV}" == 1 ]]; then
        stage "虚拟环境"
        if [[ -e "${ENV_DIR}" ]]; then
            [[ -x "${ENV_DIR}/bin/python" ]] || { echo "错误：虚拟环境已存在但损坏：${ENV_DIR}" >&2; exit 1; }
            echo "复用虚拟环境：${ENV_DIR}"
        else
            VENV_CMD=("${PYTHON_BIN}" -m venv)
            [[ "${USE_SYSTEM_SITE_PACKAGES}" == 1 ]] && VENV_CMD+=(--system-site-packages)
            VENV_CMD+=("${ENV_DIR}")
            "${VENV_CMD[@]}"
        fi
        ACTIVE_PYTHON="${ENV_DIR}/bin/python"
    fi
fi

check_torch() {
    "${ACTIVE_PYTHON}" - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("TORCH_STATUS=missing")
    raise SystemExit(2)
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
print("BF16:", torch.cuda.is_bf16_supported())
if not torch.cuda.is_available():
    print("TORCH_STATUS=cpu_or_unavailable")
    raise SystemExit(3)
if not torch.cuda.is_bf16_supported():
    print("TORCH_STATUS=no_bf16")
    raise SystemExit(4)
print("TORCH_STATUS=ok")
PY
}

stage "PyTorch与GPU检查"
set +e
check_torch
TORCH_STATUS=$?
set -e
if [[ "${TORCH_STATUS}" -ne 0 ]]; then
    if [[ "${CHECK_ONLY}" == 1 || "${INSTALL_TORCH}" != 1 ]]; then
        echo "错误：当前环境缺少可用的CUDA/BF16 PyTorch。请选择合适的AutoDL镜像，或确认官方安装命令后设置 INSTALL_TORCH=1 和 TORCH_INDEX_URL。" >&2
        exit 1
    fi
    [[ -n "${TORCH_INDEX_URL}" ]] || { echo "错误：INSTALL_TORCH=1必须显式提供TORCH_INDEX_URL；脚本不会猜测CUDA版本。" >&2; exit 1; }
    stage "显式安装PyTorch"
    TORCH_CMD=("${ACTIVE_PYTHON}" -m pip install torch --index-url "${TORCH_INDEX_URL}")
    "${TORCH_CMD[@]}"
    check_torch
else
    echo "保留当前可用PyTorch，不执行重装。"
fi

if [[ "${CHECK_ONLY}" == 1 ]]; then
    stage "只读环境检查"
else
    if [[ "${INSTALL_DEPS}" == 1 ]]; then
        stage "安装非PyTorch依赖"
        echo "使用依赖文件：${REQUIREMENTS_FILE}"
        if command -v uv >/dev/null 2>&1; then
            INSTALL_CMD=(uv pip install --python "${ACTIVE_PYTHON}" -r "${REQUIREMENTS_FILE}" pytest)
        else
            INSTALL_CMD=("${ACTIVE_PYTHON}" -m pip install -r "${REQUIREMENTS_FILE}" pytest)
        fi
        "${INSTALL_CMD[@]}"
    fi
fi

stage "环境信息与依赖验证"
uname -a
command -v nvidia-smi >/dev/null 2>&1 || { echo "错误：找不到nvidia-smi。" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h "${STORAGE_ROOT}"
echo "项目根目录：${PROJECT_ROOT}"
echo "Git branch：$(git branch --show-current 2>/dev/null || true)"
echo "Git commit：$(git rev-parse HEAD 2>/dev/null || true)"
if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then echo "Git dirty：true"; else echo "Git dirty：false"; fi
[[ -d "${DATA_ROOT}/processed" ]] || echo "警告：处理后数据尚未准备：${DATA_ROOT}/processed" >&2
[[ -f "${MODELS_DIR}/Qwen3.5-2B-Base/config.json" ]] || echo "警告：Base模型尚未准备：${MODELS_DIR}/Qwen3.5-2B-Base" >&2
"${ACTIVE_PYTHON}" - <<'PY'
import importlib
packages = ("torch", "transformers", "datasets", "accelerate", "peft", "trl", "tensorboard", "pytest")
missing = []
for package in packages:
    try: importlib.import_module(package)
    except ImportError: missing.append(package)
if missing: raise SystemExit("缺少关键依赖：" + ", ".join(missing))
print("关键Python依赖均可导入。")
PY

if [[ "${CHECK_ONLY}" == 1 ]]; then
    echo "CHECK_ONLY完成：未创建环境、安装依赖或写入报告。"
    exit 0
fi

stage "写入运行环境文件"
RUNTIME_TMP="${RUNTIME_ENV_FILE}.tmp.$$"
{
    printf '# MedicalGPT AutoDL运行环境（不含任何密钥）\n'
    printf 'export HF_HOME=%q\n' "${CACHE_DIR}"
    printf 'export HF_HUB_CACHE=%q\n' "${CACHE_DIR}/hub"
    printf 'export HF_DATASETS_CACHE=%q\n' "${CACHE_DIR}/datasets"
    printf 'export TRANSFORMERS_CACHE=%q\n' "${CACHE_DIR}/transformers"
    printf 'export TOKENIZERS_PARALLELISM=false\n'
} > "${RUNTIME_TMP}"
chmod 0644 "${RUNTIME_TMP}"
mv -f "${RUNTIME_TMP}" "${RUNTIME_ENV_FILE}"

stage "依赖一致性与测试"
"${ACTIVE_PYTHON}" -m pip check
if [[ "${RUN_TESTS}" == 1 ]]; then "${ACTIVE_PYTHON}" -m pytest -q; fi

stage "环境锁定与报告"
LOCK_TMP="${LOCK_FILE}.tmp.$$"
"${ACTIVE_PYTHON}" -m pip freeze > "${LOCK_TMP}"
mv -f "${LOCK_TMP}" "${LOCK_FILE}"
REPORT_TMP="${REPORT_FILE}.tmp.$$"
REPORT_PATH="${REPORT_TMP}" ACTIVE_PYTHON_PATH="${ACTIVE_PYTHON}" STORAGE_ROOT_VALUE="${STORAGE_ROOT}" \
ENV_DIR_VALUE="${ENV_DIR}" REQUIREMENTS_VALUE="${REQUIREMENTS_FILE}" PROJECT_ROOT_VALUE="${PROJECT_ROOT}" \
"${ACTIVE_PYTHON}" - <<'PY'
import importlib.metadata, json, os, platform, socket, subprocess
from datetime import datetime, timezone
import torch
def git(*args):
    result = subprocess.run(["git", *args], cwd=os.environ["PROJECT_ROOT_VALUE"], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None
props = torch.cuda.get_device_properties(0)
report = {
    "timestamp": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
    "os": platform.platform(), "python_version": platform.python_version(),
    "python_path": os.path.realpath(os.environ["ACTIVE_PYTHON_PATH"]), "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0), "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
    "gpu_total_memory_bytes": props.total_memory, "bf16_supported": torch.cuda.is_bf16_supported(),
    **{f"{name}_version": importlib.metadata.version(name) for name in ("transformers", "datasets", "accelerate", "peft", "trl")},
    "git_branch": git("branch", "--show-current"), "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain")), "requirements_file": os.environ["REQUIREMENTS_VALUE"],
    "storage_root": os.environ["STORAGE_ROOT_VALUE"], "env_dir": os.environ["ENV_DIR_VALUE"],
}
with open(os.environ["REPORT_PATH"], "w", encoding="utf-8") as file:
    json.dump(report, file, ensure_ascii=False, indent=2); file.write("\n")
PY
mv -f "${REPORT_TMP}" "${REPORT_FILE}"

stage "完成"
echo "激活环境：source ${ENV_DIR}/bin/activate"
echo "加载缓存环境：source ${RUNTIME_ENV_FILE}"
echo "下一步：下载Base模型；上传并校验processed数据；运行pytest；运行Base评估smoke；最后运行PT smoke。"
