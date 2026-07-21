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
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
MODEL_REVISION="${MODEL_REVISION:-main}"
MODEL_DIR="${MODEL_DIR:-${STORAGE_ROOT}/models/Qwen3.5-2B-Base}"
PARTIAL_DIR="${MODEL_DIR}.partial"
CACHE_DIR="${CACHE_DIR:-${STORAGE_ROOT}/cache/huggingface}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VERIFY_SHA256="${VERIFY_SHA256:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_IF_COMPLETE="${SKIP_IF_COMPLETE:-1}"

stage() { CURRENT_STAGE="$1"; printf '\n[%s]\n' "${CURRENT_STAGE}"; }
quote_command() { printf '%q ' "$@"; printf '\n'; }

DOWNLOAD_CMD=(env "HF_HOME=${CACHE_DIR}" "HF_HUB_CACHE=${CACHE_DIR}/hub")
[[ -n "${HF_ENDPOINT:-}" ]] && DOWNLOAD_CMD+=("HF_ENDPOINT=${HF_ENDPOINT}")
DOWNLOAD_CMD+=(hf download "${MODEL_ID}" --revision "${MODEL_REVISION}" --local-dir "${PARTIAL_DIR}")

if [[ "${DRY_RUN}" == 1 ]]; then
    stage "DRY_RUN"
    printf 'MODEL_ID=%q\nMODEL_REVISION=%q\nMODEL_DIR=%q\nPARTIAL_DIR=%q\nCACHE_DIR=%q\n' \
        "${MODEL_ID}" "${MODEL_REVISION}" "${MODEL_DIR}" "${PARTIAL_DIR}" "${CACHE_DIR}"
    quote_command "${DOWNLOAD_CMD[@]}"
    printf 'DRY_RUN完成：未读取认证变量、未创建目录、未访问网络或下载文件。\n'
    exit 0
fi

stage "工具与路径检查"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "错误：找不到Python：${PYTHON_BIN}" >&2; exit 1; }
command -v hf >/dev/null 2>&1 || {
    "${PYTHON_BIN}" -c 'import huggingface_hub' >/dev/null 2>&1 || true
    echo "错误：找不到hf命令。请在当前Python环境安装huggingface_hub后重试：python -m pip install huggingface_hub" >&2
    exit 1
}
"${PYTHON_BIN}" - <<'PY'
import huggingface_hub
print("huggingface_hub:", huggingface_hub.__version__)
PY
MODEL_PARENT="$(dirname -- "${MODEL_DIR}")"
mkdir -p "${MODEL_PARENT}" "${CACHE_DIR}"
[[ -w "${MODEL_PARENT}" ]] || { echo "错误：模型父目录不可写：${MODEL_PARENT}" >&2; exit 1; }
df -h "${MODEL_PARENT}"
echo "HF_HOME=${CACHE_DIR}"
echo "HF_HUB_CACHE=${CACHE_DIR}/hub"
echo "下载模式：hf download，支持partial目录断点续传；不会在命令行展开认证Token。"

verify_model() {
    local candidate="$1" write_manifest="$2" resolved_revision="$3"
    CANDIDATE_DIR="${candidate}" FINAL_MODEL_DIR="${MODEL_DIR}" MODEL_ID_VALUE="${MODEL_ID}" \
    REQUESTED_REVISION="${MODEL_REVISION}" RESOLVED_REVISION="${resolved_revision}" \
    VERIFY_SHA256_VALUE="${VERIFY_SHA256}" WRITE_MANIFEST="${write_manifest}" \
    "${PYTHON_BIN}" - <<'PY'
import hashlib, importlib.metadata, json, os
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

root = Path(os.environ["CANDIDATE_DIR"])
def require_file(path, label):
    if not path.is_file() or path.stat().st_size <= 0: raise SystemExit(f"{label}不存在或为空：{path}")
    if path.read_bytes()[:120].startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise SystemExit(f"文件仍是Git LFS pointer：{path}")
    return path
config = require_file(root / "config.json", "config.json")
tokenizer_candidates = [root / name for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json")]
tokenizer_files = [path for path in tokenizer_candidates if path.is_file() and path.stat().st_size > 0]
if not tokenizer_files: raise SystemExit("缺少非空tokenizer配置文件。")
weights = sorted(root.glob("*.safetensors"))
if not weights: raise SystemExit("缺少*.safetensors模型权重。")
for path in weights: require_file(path, "权重文件")
index = root / "model.safetensors.index.json"
if index.exists():
    require_file(index, "权重索引")
    try: mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc: raise SystemExit(f"权重索引无效：{exc}")
    for name in sorted(set(mapping.values())): require_file(root / name, "索引引用的权重分片")
AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=True)
encoded = tokenizer("患者出现发热和咳嗽。")
if not encoded.get("input_ids"): raise SystemExit("tokenizer中文医疗文本编码结果为空。")
all_files = sorted(path for path in root.rglob("*") if path.is_file())
sha_files = sorted(set([config, index] + tokenizer_files + weights)) if index.exists() else sorted(set([config] + tokenizer_files + weights))
if os.environ["VERIFY_SHA256_VALUE"] == "1":
    sums = []
    for path in sha_files:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""): digest.update(chunk)
        sums.append(f"{digest.hexdigest()}  {path.relative_to(root)}")
    temporary = root / "sha256sums.txt.tmp"
    temporary.write_text("\n".join(sums) + "\n", encoding="utf-8")
    temporary.replace(root / "sha256sums.txt")
manifest = {
    "model_id": os.environ["MODEL_ID_VALUE"], "requested_revision": os.environ["REQUESTED_REVISION"],
    "resolved_revision": os.environ["RESOLVED_REVISION"] or None,
    "download_time": datetime.now(timezone.utc).isoformat(), "local_path": os.environ["FINAL_MODEL_DIR"],
    "huggingface_hub_version": importlib.metadata.version("huggingface_hub"), "file_count": len(all_files),
    "total_bytes": sum(path.stat().st_size for path in all_files), "config_file": config.name,
    "tokenizer_files": [str(path.relative_to(root)) for path in tokenizer_files],
    "weight_files": [{"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size} for path in weights],
    "sha256_performed": os.environ["VERIFY_SHA256_VALUE"] == "1", "validation_result": "passed",
    "tokenizer_validation": "passed",
}
if os.environ["WRITE_MANIFEST"] == "1":
    temporary = root / "model_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "model_manifest.json")
print(json.dumps({"total_bytes": manifest["total_bytes"], "weight_shards": len(weights), "tokenizer_validation": "passed"}))
PY
}

if [[ -d "${MODEL_DIR}" ]]; then
    set +e
    verify_model "${MODEL_DIR}" 0 ""
    FINAL_VALID=$?
    set -e
    if [[ "${FINAL_VALID}" -eq 0 && "${SKIP_IF_COMPLETE}" == 1 && "${FORCE}" != 1 ]]; then
        echo "模型目录已完整，按SKIP_IF_COMPLETE=1跳过：${MODEL_DIR}"
        exit 0
    fi
    if [[ "${FORCE}" != 1 ]]; then
        echo "错误：最终模型目录已存在；如需重新下载请设置FORCE=1，旧目录将先备份。" >&2
        exit 1
    fi
    BACKUP_DIR="${MODEL_DIR}.backup.$(date '+%Y%m%d_%H%M%S')"
    echo "FORCE保护：将旧模型移动到备份目录：${BACKUP_DIR}"
    mv "${MODEL_DIR}" "${BACKUP_DIR}"
fi

if [[ -e "${PARTIAL_DIR}" && ! -d "${PARTIAL_DIR}" ]]; then
    echo "错误：partial路径存在但不是目录：${PARTIAL_DIR}" >&2
    exit 1
fi
mkdir -p "${PARTIAL_DIR}"

stage "解析revision"
set +e
RESOLVE_INFO="$(HF_HOME="${CACHE_DIR}" HF_HUB_CACHE="${CACHE_DIR}/hub" "${PYTHON_BIN}" - "${MODEL_ID}" "${MODEL_REVISION}" <<'PY'
import sys
from huggingface_hub import HfApi
info = HfApi().model_info(sys.argv[1], revision=sys.argv[2], files_metadata=True)
sizes = [item.size for item in (info.siblings or []) if item.size is not None]
print(f"{info.sha or ''}|{sum(sizes) if sizes else ''}")
PY
)"
RESOLVE_STATUS=$?
set -e
RESOLVED_REVISION="${RESOLVE_INFO%%|*}"
PLANNED_BYTES="${RESOLVE_INFO#*|}"
if [[ "${RESOLVE_STATUS}" -ne 0 || -z "${RESOLVED_REVISION}" ]]; then
    RESOLVED_REVISION=""
    PLANNED_BYTES=""
    echo "警告：无法解析实际commit SHA，将按请求revision下载，不虚构SHA。" >&2
else
    echo "Resolved revision: ${RESOLVED_REVISION}"
fi

stage "下载前空间信息"
if [[ "${PLANNED_BYTES}" =~ ^[0-9]+$ ]]; then
    PARTIAL_BYTES="$(du -sb "${PARTIAL_DIR}" | awk '{print $1}')"
    FREE_BYTES="$(df -PB1 "${MODEL_PARENT}" | awk 'NR==2 {print $4}')"
    REMAINING_BYTES=$(( PLANNED_BYTES > PARTIAL_BYTES ? PLANNED_BYTES - PARTIAL_BYTES : 0 ))
    echo "计划总字节数：${PLANNED_BYTES}；partial已有：${PARTIAL_BYTES}；估计剩余：${REMAINING_BYTES}；可用：${FREE_BYTES}"
    if (( REMAINING_BYTES > FREE_BYTES )); then
        echo "错误：目标文件系统剩余空间不足，无法完成模型下载。" >&2
        exit 1
    fi
else
    echo "警告：Hub未提供可用的文件总量，无法可靠预估空间；下载失败时partial会保留。" >&2
fi
DRY_CHECK_CMD=(env "HF_HOME=${CACHE_DIR}" "HF_HUB_CACHE=${CACHE_DIR}/hub")
[[ -n "${HF_ENDPOINT:-}" ]] && DRY_CHECK_CMD+=("HF_ENDPOINT=${HF_ENDPOINT}")
DRY_CHECK_CMD+=(hf download "${MODEL_ID}" --revision "${MODEL_REVISION}" --dry-run)
set +e
"${DRY_CHECK_CMD[@]}"
DRY_CHECK_STATUS=$?
set -e
if [[ "${DRY_CHECK_STATUS}" -ne 0 ]]; then echo "警告：当前hf版本不支持或无法完成大小预估，将继续正式下载。" >&2; fi

stage "下载到partial目录"
"${DOWNLOAD_CMD[@]}"

stage "离线完整性验证"
verify_model "${PARTIAL_DIR}" 1 "${RESOLVED_REVISION}"

stage "发布最终模型目录"
mv "${PARTIAL_DIR}" "${MODEL_DIR}"
MANIFEST_PATH="${MODEL_DIR}/model_manifest.json"
TOTAL_BYTES="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["total_bytes"])' "${MANIFEST_PATH}")"
WEIGHT_COUNT="$("${PYTHON_BIN}" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["weight_files"]))' "${MANIFEST_PATH}")"
echo "模型目录：${MODEL_DIR}"
echo "Resolved revision：${RESOLVED_REVISION:-未解析}"
echo "总字节数：${TOTAL_BYTES}"
echo "权重分片数：${WEIGHT_COUNT}"
echo "Tokenizer验证：通过"
echo "Manifest：${MANIFEST_PATH}"
echo "训练参数 model_name_or_path 应使用：${MODEL_DIR}"
