#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_NAME="$(basename -- "$0")"
CURRENT_STAGE="初始化"
on_error() {
    local status=$?
    printf '[ERROR] 脚本=%s 阶段=%s 行号=%s 退出码=%s 命令=%s\n' "$SCRIPT_NAME" "$CURRENT_STAGE" "$1" "$status" "$2" >&2
    exit "$status"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
trap 'echo "[ERROR] 脚本=$SCRIPT_NAME 阶段=$CURRENT_STAGE 收到SIGINT" >&2; exit 130' INT
trap 'echo "[ERROR] 脚本=$SCRIPT_NAME 阶段=$CURRENT_STAGE 收到SIGTERM" >&2; exit 143' TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -d /root/autodl-tmp ]]; then DEFAULT_STORAGE_ROOT=/root/autodl-tmp; else DEFAULT_STORAGE_ROOT="$HOME"; fi
STORAGE_ROOT="${STORAGE_ROOT:-$DEFAULT_STORAGE_ROOT}"
GPU_ID="${GPU_ID:-0}"; PYTHON_BIN="${PYTHON_BIN:-python}"; RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${STORAGE_ROOT}/models/Qwen3.5-2B-Base}"
PT_SFT_ADAPTER_PATH="${PT_SFT_ADAPTER_PATH:-}"
TRAIN_FILE="${TRAIN_FILE:-${STORAGE_ROOT}/datasets/medicalgpt/processed/cmexam/decontaminated/train/train.jsonl}"
CACHE_DIR="${CACHE_DIR:-${STORAGE_ROOT}/cache/huggingface}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/medicalgpt/grpo}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/medicalgpt/grpo}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_ID}.log}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOG_ROOT}/tensorboard/${RUN_ID}}"
RUN_NAME="${RUN_NAME:-medical-cmexam-grpo-${RUN_ID}}"

SEED="${SEED:-42}"; DATA_SEED="${DATA_SEED:-42}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"; REPORT_TO="${REPORT_TO:-tensorboard}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"; WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"; WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_STEPS="${MAX_STEPS:-500}"; NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-}"; MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-32}"; BETA="${BETA:-0}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"; SAVE_STEPS="${SAVE_STEPS:-100}"; SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"; BF16="${BF16:-True}"; FP16="${FP16:-False}"
REMOVE_UNUSED_COLUMNS="${REMOVE_UNUSED_COLUMNS:-False}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"; OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"; DRY_RUN="${DRY_RUN:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-True}"; TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
SHUFFLE_BEFORE_SELECT="${SHUFFLE_BEFORE_SELECT:-True}"
CORRECT_REWARD="${CORRECT_REWARD:-1.0}"; INCORRECT_REWARD="${INCORRECT_REWARD:-0.0}"
FORMAT_REWARD="${FORMAT_REWARD:-0.1}"; INVALID_PENALTY="${INVALID_PENALTY:--0.1}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"

ADAPTER_GUIDANCE='PT_SFT_ADAPTER_PATH必须指向正式Base→PT→SFT输出目录，而不是PT、Base→SFT或smoke输出。'
[[ "$BF16" == True && "$FP16" == False ]] || { echo "错误：固定协议要求BF16=True且FP16=False。" >&2; exit 1; }
[[ "$REMOVE_UNUSED_COLUMNS" == False ]] || { echo "错误：REMOVE_UNUSED_COLUMNS必须为False。" >&2; exit 1; }
[[ -z "$MAX_PROMPT_LENGTH" ]] || { echo "错误：当前训练入口没有max_prompt_length参数，请勿设置MAX_PROMPT_LENGTH。" >&2; exit 1; }
[[ "$BETA" == 0 || "$BETA" == 0.0 ]] || { echo "错误：固定协议要求BETA=0；非零beta会创建第二个ref adapter。" >&2; exit 1; }
[[ "$TRAIN_FILE" == */decontaminated/train/* && "$TRAIN_FILE" != */validation/* && "$TRAIN_FILE" != */test/* ]] || {
    echo "错误：TRAIN_FILE必须位于连续目录decontaminated/train；validation和test禁止用于GRPO训练：$TRAIN_FILE" >&2; exit 1;
}
for value in "$PER_DEVICE_TRAIN_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$MAX_STEPS" "$NUM_GENERATIONS" "$MAX_COMPLETION_LENGTH"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "错误：正整数超参数非法：$value" >&2; exit 1; }
done
[[ -z "$MAX_SAMPLES" || "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]] || { echo "错误：MAX_SAMPLES必须为空或正整数。" >&2; exit 1; }
(( NUM_GENERATIONS >= 2 )) || { echo "错误：NUM_GENERATIONS至少为2。" >&2; exit 1; }
GENERATION_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
(( GENERATION_BATCH_SIZE % NUM_GENERATIONS == 0 )) || {
    echo "错误：generation_batch_size=$GENERATION_BATCH_SIZE不能被NUM_GENERATIONS=$NUM_GENERATIONS整除。" >&2; exit 1;
}

if [[ "$DRY_RUN" == 1 && -z "$PT_SFT_ADAPTER_PATH" ]]; then
    PT_SFT_ADAPTER_PATH="$STORAGE_ROOT/outputs/medicalgpt/sft/PT_SFT_ADAPTER_PLACEHOLDER"
fi
if [[ -n "$PT_SFT_ADAPTER_PATH" ]]; then
    ADAPTER_ABS="$(realpath -m -- "$PT_SFT_ADAPTER_PATH")"; OUTPUT_ABS="$(realpath -m -- "$OUTPUT_DIR")"
    [[ "$ADAPTER_ABS" != "$OUTPUT_ABS" && "$ADAPTER_ABS" != "$OUTPUT_ABS/"* && "$OUTPUT_ABS" != "$ADAPTER_ABS/"* ]] || {
        echo "错误：输出目录与起始adapter相同或互相嵌套，禁止原地覆盖。$ADAPTER_GUIDANCE" >&2; exit 1;
    }
fi

CMD=(
    "$PYTHON_BIN" training/medical_grpo_training.py
    --model_name_or_path "$MODEL_NAME_OR_PATH" --peft_path "$PT_SFT_ADAPTER_PATH"
    --train_file "$TRAIN_FILE" --output_dir "$OUTPUT_DIR"
    --torch_dtype "$TORCH_DTYPE" --cache_dir "$CACHE_DIR"
    --local_files_only "$LOCAL_FILES_ONLY" --trust_remote_code "$TRUST_REMOTE_CODE"
    --correct_reward "$CORRECT_REWARD" --incorrect_reward "$INCORRECT_REWARD"
    --format_reward "$FORMAT_REWARD" --invalid_penalty "$INVALID_PENALTY"
    --data_seed "$DATA_SEED" --shuffle_before_select "$SHUFFLE_BEFORE_SELECT"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE" --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY"
    --max_steps "$MAX_STEPS" --num_generations "$NUM_GENERATIONS"
    --max_completion_length "$MAX_COMPLETION_LENGTH" --beta "$BETA"
    --bf16 "$BF16" --fp16 "$FP16" --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
    --remove_unused_columns "$REMOVE_UNUSED_COLUMNS"
    --logging_steps "$LOGGING_STEPS" --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_TOTAL_LIMIT"
    --report_to "$REPORT_TO" --logging_dir "$TENSORBOARD_DIR" --run_name "$RUN_NAME" --seed "$SEED"
)
[[ -z "$MAX_SAMPLES" ]] || CMD+=(--max_samples "$MAX_SAMPLES")
[[ -z "$RESUME_FROM_CHECKPOINT" ]] || CMD+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
[[ "$OVERWRITE_OUTPUT_DIR" != 1 ]] || CMD+=(--overwrite_output_dir True)

if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY_RUN：不检查CUDA、Base、adapter或数据；不创建目录，不启动Python。\n'
    printf 'MODEL_NAME_OR_PATH=%s\nPT_SFT_ADAPTER_PATH=%s\nTRAIN_FILE=%s\nOUTPUT_DIR=%s\nLOG_FILE=%s\n' \
        "$MODEL_NAME_OR_PATH" "$PT_SFT_ADAPTER_PATH" "$TRAIN_FILE" "$OUTPUT_DIR" "$LOG_FILE"
    printf 'MAX_STEPS=%s MAX_SAMPLES=%s BATCH=%s GRAD_ACCUM=%s NUM_GENERATIONS=%s MAX_COMPLETION_LENGTH=%s BETA=%s\n' \
        "$MAX_STEPS" "${MAX_SAMPLES:-<all>}" "$PER_DEVICE_TRAIN_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$NUM_GENERATIONS" "$MAX_COMPLETION_LENGTH" "$BETA"
    printf 'COMMAND: '; printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

CURRENT_STAGE="真实运行前检查"
[[ -n "$PT_SFT_ADAPTER_PATH" ]] || { echo "错误：未提供PT_SFT_ADAPTER_PATH。$ADAPTER_GUIDANCE" >&2; exit 1; }
ADAPTER_LOWER="${PT_SFT_ADAPTER_PATH,,}"
[[ "$ADAPTER_LOWER" != *smoke* && "$ADAPTER_LOWER" != *"/pt/"* && "$ADAPTER_LOWER" != *sft_base* && "$ADAPTER_LOWER" != *base_sft* ]] || {
    echo "错误：adapter路径疑似PT、Base→SFT或smoke输出。$ADAPTER_GUIDANCE" >&2; exit 1;
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "错误：找不到PYTHON_BIN：$PYTHON_BIN" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "错误：找不到nvidia-smi。" >&2; exit 1; }
[[ -f training/medical_grpo_training.py ]] || { echo "错误：缺少训练入口。" >&2; exit 1; }
[[ -d "$MODEL_NAME_OR_PATH" && -f "$MODEL_NAME_OR_PATH/config.json" ]] || { echo "错误：本地Base目录不存在或缺少config.json：$MODEL_NAME_OR_PATH" >&2; exit 1; }
[[ -d "$PT_SFT_ADAPTER_PATH" && -f "$PT_SFT_ADAPTER_PATH/adapter_config.json" ]] || { echo "错误：adapter目录或配置缺失。$ADAPTER_GUIDANCE" >&2; exit 1; }
[[ -f "$PT_SFT_ADAPTER_PATH/adapter_model.safetensors" || -f "$PT_SFT_ADAPTER_PATH/adapter_model.bin" ]] || { echo "错误：adapter权重缺失。$ADAPTER_GUIDANCE" >&2; exit 1; }
[[ -s "$TRAIN_FILE" ]] || { echo "错误：GRPO train.jsonl不存在或为空：$TRAIN_FILE" >&2; exit 1; }
if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" && -z "$RESUME_FROM_CHECKPOINT" && "$OVERWRITE_OUTPUT_DIR" != 1 ]]; then
    echo "错误：OUTPUT_DIR非空；请设置RESUME_FROM_CHECKPOINT或OVERWRITE_OUTPUT_DIR=1：$OUTPUT_DIR" >&2; exit 1
fi
CHECK_PATH="$OUTPUT_DIR"; while [[ ! -e "$CHECK_PATH" && "$CHECK_PATH" != / ]]; do CHECK_PATH="$(dirname -- "$CHECK_PATH")"; done
FREE_KB="$(df -Pk "$CHECK_PATH" | awk 'NR==2 {print $4}')"
(( FREE_KB >= MIN_FREE_GB * 1024 * 1024 )) || { echo "错误：输出盘剩余空间不足${MIN_FREE_GB}GB。" >&2; exit 1; }
"$PYTHON_BIN" - <<'PY'
import sys
try:
    import accelerate, datasets, peft, torch, transformers, trl
except ImportError as exc:
    print(f"错误：缺少训练依赖：{exc}", file=sys.stderr); raise SystemExit(1)
if not torch.cuda.is_available(): print("错误：CUDA不可用。", file=sys.stderr); raise SystemExit(1)
if not torch.cuda.is_bf16_supported(): print("错误：当前GPU或PyTorch不支持BF16。", file=sys.stderr); raise SystemExit(1)
PY

CURRENT_STAGE="创建运行目录"
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$LOG_ROOT" "$TENSORBOARD_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
CURRENT_STAGE="TRL GRPO正式训练"
set +e
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
    echo "GRPO正式训练失败，退出码=$TRAIN_STATUS；日志=$LOG_FILE；输出=$OUTPUT_DIR" >&2; exit "$TRAIN_STATUS"
fi
printf 'GRPO正式训练完成。\n输出目录：%s\n日志文件：%s\n起始adapter：%s\n数据文件：%s\n' \
    "$OUTPUT_DIR" "$LOG_FILE" "$PT_SFT_ADAPTER_PATH" "$TRAIN_FILE"
