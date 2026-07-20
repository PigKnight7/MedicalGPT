#!/usr/bin/env bash

# 单张GPU上的MedicalGPT增量预训练冒烟测试。
#
# 目标：
# 1. 验证模型、数据和LoRA配置能否正常加载；
# 2. 验证1024长度packing是否会OOM；
# 3. 验证训练、评估、保存和日志流程；
# 4. 只训练20步，不用于评价最终模型效果。
#
# 本脚本默认面向RTX 4090D等支持BF16的GPU。
# 本地CPU版PyTorch环境不要直接执行正式训练。

set -Eeuo pipefail
IFS=$'\n\t'

trap 'echo "[ERROR] 第 ${LINENO} 行执行失败：${BASH_COMMAND}" >&2' ERR


# =============================================================================
# 1. 定位项目目录
# =============================================================================

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

PROJECT_ROOT="$(
    cd -- "${SCRIPT_DIR}/.."
    pwd
)"

cd "${PROJECT_ROOT}"


# =============================================================================
# 2. 基础配置
# =============================================================================

# AutoDL存在持久化数据盘时优先使用；本地则使用用户主目录。
if [[ -d "/root/autodl-tmp" ]]; then
    DEFAULT_STORAGE_ROOT="/root/autodl-tmp"
else
    DEFAULT_STORAGE_ROOT="${HOME}"
fi

STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"

# 可以通过环境变量选择GPU，例如：
# GPU_ID=1 bash scripts/run_pt_medical_2b_smoke.sh
GPU_ID="${GPU_ID:-0}"

# 默认使用当前环境中的python。
PYTHON_BIN="${PYTHON_BIN:-python}"

# 如果云端已经提前下载了模型，则优先使用本地模型目录；
# 否则使用Hugging Face模型ID，由Transformers自动下载。
DEFAULT_LOCAL_MODEL="${STORAGE_ROOT}/models/Qwen3.5-2B-Base"

if [[ -f "${DEFAULT_LOCAL_MODEL}/config.json" ]]; then
    DEFAULT_MODEL="${DEFAULT_LOCAL_MODEL}"
else
    DEFAULT_MODEL="Qwen/Qwen3.5-2B-Base"
fi

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${DEFAULT_MODEL}}"

# 处理后的PT数据目录。
TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-${STORAGE_ROOT}/datasets/medicalgpt/processed/pt/train}"
VALIDATION_FILE_DIR="${VALIDATION_FILE_DIR:-${STORAGE_ROOT}/datasets/medicalgpt/processed/pt/validation}"

# Hugging Face缓存目录。
CACHE_DIR="${CACHE_DIR:-${STORAGE_ROOT}/cache/huggingface}"

# 为每次冒烟测试生成独立目录，防止覆盖之前的实验。
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/medicalgpt/pt_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"

LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/medicalgpt/pt_smoke}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_ID}.log}"

RUN_NAME="${RUN_NAME:-medical-pt-2b-smoke-${RUN_ID}}"


# =============================================================================
# 3. 冒烟测试超参数
# =============================================================================

MAX_STEPS="${MAX_STEPS:-20}"

# 注意：pretraining.py在packing之后再应用max_train_samples，
# 因此这里限制的是固定长度的packed训练块，而不是原始JSONL行数。
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-200}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-50}"

BLOCK_SIZE="${BLOCK_SIZE:-1024}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"

SEED="${SEED:-42}"


# =============================================================================
# 4. 构造训练命令
# =============================================================================

CMD=(
    "${PYTHON_BIN}"
    "training/pretraining.py"

    # 模型与缓存
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --cache_dir "${CACHE_DIR}"
    --trust_remote_code True
    --torch_dtype bfloat16
    --device_map auto

    # 数据
    --train_file_dir "${TRAIN_FILE_DIR}"
    --validation_file_dir "${VALIDATION_FILE_DIR}"
    --max_train_samples "${MAX_TRAIN_SAMPLES}"
    --max_eval_samples "${MAX_EVAL_SAMPLES}"
    --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}"
    --block_size "${BLOCK_SIZE}"
    --packing True

    # 训练与评估
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

    # LoRA
    --use_peft True
    --target_modules all
    --lora_rank "${LORA_RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"

    # 混合精度与显存优化
    --bf16
    --gradient_checkpointing True

    # 日志
    --logging_strategy steps
    --logging_steps 1
    --logging_first_step True
    --report_to tensorboard
    --logging_dir "${LOG_ROOT}/tensorboard/${RUN_ID}"
    --run_name "${RUN_NAME}"

    # 每10步评估一次
    --eval_strategy steps
    --eval_steps 10

    # 每10步保存一次，最多保留两个checkpoint
    --save_strategy steps
    --save_steps 10
    --save_total_limit 2
    --output_dir "${OUTPUT_DIR}"

    # DataLoader
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
)


# =============================================================================
# 5. 本地只检查命令，不启动训练
# =============================================================================

# 使用方法：
# DRY_RUN=1 bash scripts/run_pt_medical_2b_smoke.sh
#
# 此模式不会检查CUDA，也不会下载模型或启动训练。
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN模式：不会启动训练。"
    echo
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi


# =============================================================================
# 6. 启动前检查
# =============================================================================

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "错误：找不到Python命令：${PYTHON_BIN}" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "错误：当前环境找不到nvidia-smi，无法确认GPU状态。" >&2
    exit 1
fi

if [[ ! -d "${TRAIN_FILE_DIR}" ]]; then
    echo "错误：PT训练数据目录不存在：" >&2
    echo "  ${TRAIN_FILE_DIR}" >&2
    exit 1
fi

if [[ ! -d "${VALIDATION_FILE_DIR}" ]]; then
    echo "错误：PT验证数据目录不存在：" >&2
    echo "  ${VALIDATION_FILE_DIR}" >&2
    exit 1
fi

if [[ -z "$(
    find "${TRAIN_FILE_DIR}" \
        -type f \
        -name '*.jsonl' \
        -print \
        -quit
)" ]]; then
    echo "错误：训练目录中没有找到JSONL文件：" >&2
    echo "  ${TRAIN_FILE_DIR}" >&2
    exit 1
fi

if [[ -z "$(
    find "${VALIDATION_FILE_DIR}" \
        -type f \
        -name '*.jsonl' \
        -print \
        -quit
)" ]]; then
    echo "错误：验证目录中没有找到JSONL文件：" >&2
    echo "  ${VALIDATION_FILE_DIR}" >&2
    exit 1
fi

# 如果传入的是本地模型目录，检查其中是否有config.json。
if [[ "${MODEL_NAME_OR_PATH}" == /* ]]; then
    if [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
        echo "错误：本地模型目录缺少config.json：" >&2
        echo "  ${MODEL_NAME_OR_PATH}" >&2
        exit 1
    fi
fi

# 防止覆盖已有实验结果。
if [[ -d "${OUTPUT_DIR}" ]] &&
   [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "错误：输出目录已存在且非空：" >&2
    echo "  ${OUTPUT_DIR}" >&2
    echo "请更换RUN_ID或OUTPUT_DIR，不要覆盖已有实验。" >&2
    exit 1
fi

mkdir -p \
    "${CACHE_DIR}" \
    "$(dirname "${OUTPUT_DIR}")" \
    "${LOG_ROOT}" \
    "${LOG_ROOT}/tensorboard"


# =============================================================================
# 7. 检查Python依赖、CUDA和BF16支持
# =============================================================================

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"

"${PYTHON_BIN}" - <<'PY'
import sys

try:
    import accelerate
    import datasets
    import peft
    import torch
    import transformers
except ImportError as exc:
    print(f"缺少训练依赖：{exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Datasets:     {datasets.__version__}")
print(f"Accelerate:   {accelerate.__version__}")
print(f"PEFT:         {peft.__version__}")

if not torch.cuda.is_available():
    print(
        "错误：当前PyTorch无法使用CUDA。"
        "请确认安装的是CUDA版PyTorch，而不是CPU版。",
        file=sys.stderr,
    )
    raise SystemExit(1)

device_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)

print(f"GPU:          {device_name}")
print(f"Compute cap.: {capability}")
print(f"BF16 support: {torch.cuda.is_bf16_supported()}")

if not torch.cuda.is_bf16_supported():
    print(
        "错误：当前GPU或PyTorch环境不支持BF16，"
        "本脚本不能按当前配置启动。",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY


# =============================================================================
# 8. 打印实验配置
# =============================================================================

echo
echo "======================================================================"
echo "MedicalGPT PT冒烟测试"
echo "======================================================================"
echo "项目目录：        ${PROJECT_ROOT}"
echo "GPU编号：         ${GPU_ID}"
echo "模型：            ${MODEL_NAME_OR_PATH}"
echo "训练数据：        ${TRAIN_FILE_DIR}"
echo "验证数据：        ${VALIDATION_FILE_DIR}"
echo "缓存目录：        ${CACHE_DIR}"
echo "输出目录：        ${OUTPUT_DIR}"
echo "运行日志：        ${LOG_FILE}"
echo "最大训练步数：    ${MAX_STEPS}"
echo "Packed训练块数：  ${MAX_TRAIN_SAMPLES}"
echo "Packed验证块数：  ${MAX_EVAL_SAMPLES}"
echo "Block size：      ${BLOCK_SIZE}"
echo "Batch size：      ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "梯度累积：        ${GRADIENT_ACCUMULATION_STEPS}"
echo "学习率：          ${LEARNING_RATE}"
echo "======================================================================"
echo

nvidia-smi


# =============================================================================
# 9. 启动训练并记录日志
# =============================================================================

echo
echo "开始执行训练命令："
printf '%q ' "${CMD[@]}"
printf '\n\n'

set +e

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
TRAIN_STATUS="${PIPESTATUS[0]}"

set -e

if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
    echo
    echo "PT冒烟测试失败，退出码：${TRAIN_STATUS}" >&2
    echo "完整日志：${LOG_FILE}" >&2
    exit "${TRAIN_STATUS}"
fi

echo
echo "======================================================================"
echo "PT冒烟测试成功完成"
echo "输出目录：${OUTPUT_DIR}"
echo "日志文件：${LOG_FILE}"
echo "======================================================================"