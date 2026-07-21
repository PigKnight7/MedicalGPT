#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

trap 'echo "[ERROR] 第 ${LINENO} 行执行失败：${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -d /root/autodl-tmp ]]; then
    DEFAULT_STORAGE_ROOT=/root/autodl-tmp
else
    DEFAULT_STORAGE_ROOT="${HOME}"
fi
STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DEFAULT_LOCAL_MODEL="${STORAGE_ROOT}/models/Qwen3.5-2B-Base"
if [[ -f "${DEFAULT_LOCAL_MODEL}/config.json" ]]; then
    DEFAULT_MODEL="${DEFAULT_LOCAL_MODEL}"
else
    DEFAULT_MODEL="Qwen/Qwen3.5-2B-Base"
fi
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${DEFAULT_MODEL}}"
TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-${STORAGE_ROOT}/datasets/medicalgpt/processed/pt/train}"
VALIDATION_FILE_DIR="${VALIDATION_FILE_DIR:-${STORAGE_ROOT}/datasets/medicalgpt/processed/pt/validation}"
CACHE_DIR="${CACHE_DIR:-${STORAGE_ROOT}/cache/huggingface}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/medicalgpt/pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/medicalgpt/pt}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_ID}.log}"
RUN_NAME="${RUN_NAME:-medical-pt-2b-${RUN_ID}}"

BLOCK_SIZE="${BLOCK_SIZE:-1024}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
SEED="${SEED:-42}"

CMD=(
    "${PYTHON_BIN}" training/pretraining.py
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --cache_dir "${CACHE_DIR}"
    --trust_remote_code True
    --torch_dtype bfloat16
    --device_map auto
    --train_file_dir "${TRAIN_FILE_DIR}"
    --validation_file_dir "${VALIDATION_FILE_DIR}"
    --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}"
    --block_size "${BLOCK_SIZE}"
    --packing True
    --do_train
    --do_eval
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --warmup_ratio "${WARMUP_RATIO}"
    --weight_decay "${WEIGHT_DECAY}"
    --seed "${SEED}"
    --data_seed "${SEED}"
    --use_peft True
    --target_modules all
    --lora_rank "${LORA_RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --bf16
    --gradient_checkpointing True
    --logging_strategy steps
    --logging_steps "${LOGGING_STEPS}"
    --logging_first_step True
    --report_to tensorboard
    --logging_dir "${LOG_ROOT}/tensorboard/${RUN_ID}"
    --run_name "${RUN_NAME}"
    --eval_strategy steps
    --eval_steps "${EVAL_STEPS}"
    --save_strategy steps
    --save_steps "${SAVE_STEPS}"
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
    [[ -n "$(find "${directory}" -type f -name '*.jsonl' -print -quit)" ]] || {
        echo "错误：${label}目录中没有JSONL文件：${directory}" >&2
        exit 1
    }
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "错误：找不到Python命令：${PYTHON_BIN}" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo '错误：找不到nvidia-smi。' >&2; exit 1; }
check_jsonl_dir PT训练数据 "${TRAIN_FILE_DIR}"
check_jsonl_dir PT验证数据 "${VALIDATION_FILE_DIR}"
if [[ "${MODEL_NAME_OR_PATH}" == /* && ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
    echo "错误：本地模型目录缺少config.json：${MODEL_NAME_OR_PATH}" >&2
    exit 1
fi
if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "错误：输出目录已存在且非空：${OUTPUT_DIR}" >&2
    exit 1
fi

mkdir -p "${CACHE_DIR}" "${OUTPUT_DIR}" "${LOG_ROOT}" "${LOG_ROOT}/tensorboard"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" - <<'PY'
import sys
try:
    import accelerate, datasets, peft, torch, transformers
except ImportError as exc:
    print(f"缺少训练依赖：{exc}", file=sys.stderr)
    raise SystemExit(1)
if not torch.cuda.is_available():
    print("错误：当前PyTorch不是可用的CUDA环境。", file=sys.stderr)
    raise SystemExit(1)
if not torch.cuda.is_bf16_supported():
    print("错误：当前GPU或PyTorch不支持BF16。", file=sys.stderr)
    raise SystemExit(1)
PY

echo "开始PT正式训练：${RUN_ID}"
printf '%q ' "${CMD[@]}"
printf '\n'
set +e
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
    echo "PT训练失败，退出码：${TRAIN_STATUS}；日志：${LOG_FILE}" >&2
    exit "${TRAIN_STATUS}"
fi
echo "PT训练完成；输出：${OUTPUT_DIR}；日志：${LOG_FILE}"
