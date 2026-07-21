#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

trap 'echo "[ERROR] 第 ${LINENO} 行执行失败：${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -d /root/autodl-tmp ]]; then DEFAULT_STORAGE_ROOT=/root/autodl-tmp; else DEFAULT_STORAGE_ROOT="${HOME}"; fi
STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_LOCAL_MODEL="${STORAGE_ROOT}/models/Qwen3.5-2B-Base"
if [[ -f "${DEFAULT_LOCAL_MODEL}/config.json" ]]; then DEFAULT_MODEL="${DEFAULT_LOCAL_MODEL}"; else DEFAULT_MODEL="Qwen/Qwen3.5-2B-Base"; fi
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${DEFAULT_MODEL}}"
PT_ADAPTER_PATH="${PT_ADAPTER_PATH:-${STORAGE_ROOT}/outputs/medicalgpt/pt/REPLACE_WITH_PT_RUN}"
TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-${STORAGE_ROOT}/datasets/medicalgpt/processed/sft/train}"
VALIDATION_FILE_DIR="${VALIDATION_FILE_DIR:-${STORAGE_ROOT}/datasets/medicalgpt/processed/sft/validation}"
CACHE_DIR="${CACHE_DIR:-${STORAGE_ROOT}/cache/huggingface}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/medicalgpt/sft_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/medicalgpt/sft_smoke}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_ID}.log}"
RUN_NAME="${RUN_NAME:-medical-sft-2b-smoke-${RUN_ID}}"

MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_STEPS="${MAX_STEPS:-20}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-200}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-50}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
SEED="${SEED:-42}"

CMD=(
    "${PYTHON_BIN}" training/supervised_finetuning.py
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --peft_path "${PT_ADAPTER_PATH}"
    --cache_dir "${CACHE_DIR}"
    --trust_remote_code True
    --torch_dtype bfloat16
    --device_map auto
    --train_file_dir "${TRAIN_FILE_DIR}"
    --validation_file_dir "${VALIDATION_FILE_DIR}"
    --max_train_samples "${MAX_TRAIN_SAMPLES}"
    --max_eval_samples "${MAX_EVAL_SAMPLES}"
    --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}"
    --model_max_length "${MODEL_MAX_LENGTH}"
    --do_train
    --do_eval
    --max_steps "${MAX_STEPS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --warmup_steps "${WARMUP_STEPS}"
    --weight_decay "${WEIGHT_DECAY}"
    --seed "${SEED}"
    --data_seed "${SEED}"
    --use_peft True
    --bf16
    --gradient_checkpointing True
    --logging_strategy steps
    --logging_steps 1
    --logging_first_step True
    --report_to tensorboard
    --logging_dir "${LOG_ROOT}/tensorboard/${RUN_ID}"
    --run_name "${RUN_NAME}"
    --eval_strategy steps
    --eval_steps 10
    --save_strategy steps
    --save_steps 10
    --save_total_limit 2
    --output_dir "${OUTPUT_DIR}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY_RUN模式：不会启动训练。\n\n'
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

check_jsonl_dir() {
    local label="$1" directory="$2"
    [[ -d "${directory}" ]] || { echo "错误：${label}目录不存在：${directory}" >&2; exit 1; }
    [[ -n "$(find "${directory}" -type f -name '*.jsonl' -print -quit)" ]] || { echo "错误：${label}目录中没有JSONL文件：${directory}" >&2; exit 1; }
}
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "错误：找不到Python命令：${PYTHON_BIN}" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo '错误：找不到nvidia-smi。' >&2; exit 1; }
[[ -n "${PT_ADAPTER_PATH}" && -d "${PT_ADAPTER_PATH}" ]] || { echo "错误：请通过PT_ADAPTER_PATH提供存在的PT adapter目录。当前值：${PT_ADAPTER_PATH}" >&2; exit 1; }
check_jsonl_dir SFT训练数据 "${TRAIN_FILE_DIR}"
check_jsonl_dir SFT验证数据 "${VALIDATION_FILE_DIR}"
if [[ "${MODEL_NAME_OR_PATH}" == /* && ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then echo "错误：本地模型目录缺少config.json：${MODEL_NAME_OR_PATH}" >&2; exit 1; fi
if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then echo "错误：输出目录已存在且非空：${OUTPUT_DIR}" >&2; exit 1; fi

mkdir -p "${CACHE_DIR}" "${OUTPUT_DIR}" "${LOG_ROOT}" "${LOG_ROOT}/tensorboard"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
"${PYTHON_BIN}" - <<'PY'
import sys
try:
    import accelerate, datasets, peft, torch, transformers
except ImportError as exc:
    print(f"缺少训练依赖：{exc}", file=sys.stderr); raise SystemExit(1)
if not torch.cuda.is_available(): print("错误：当前PyTorch不是可用的CUDA环境。", file=sys.stderr); raise SystemExit(1)
if not torch.cuda.is_bf16_supported(): print("错误：当前GPU或PyTorch不支持BF16。", file=sys.stderr); raise SystemExit(1)
PY

echo "开始SFT冒烟测试：${RUN_ID}"
printf '%q ' "${CMD[@]}"; printf '\n'
set +e
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then echo "SFT冒烟测试失败，退出码：${TRAIN_STATUS}；日志：${LOG_FILE}" >&2; exit "${TRAIN_STATUS}"; fi
echo "SFT冒烟测试完成；输出：${OUTPUT_DIR}；日志：${LOG_FILE}"
