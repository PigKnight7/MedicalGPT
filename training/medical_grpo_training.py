#!/usr/bin/env python3
"""使用现有 PT→SFT LoRA adapter 执行 CMExam TRL GRPO 训练。"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

# 支持 ``python training/medical_grpo_training.py`` 直接运行。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medicalgpt_ext.cmexam_grpo_data import (  # noqa: E402
    DEFAULT_CMEXAM_GRPO_TRAIN_FILE,
    CMExamGRPODataConfig,
    CMExamGRPODataSummary,
    prepare_cmexam_grpo_examples,
)
from medicalgpt_ext.cmexam_rewards import (  # noqa: E402
    CMExamRewardConfig,
    RewardFunction,
    build_cmexam_reward_functions,
)


LOGGER = logging.getLogger("medical_grpo_training")
THINKING_DISABLED_STRATEGY = "TRL chat_template_kwargs: enable_thinking=False"
DATA_SUMMARY_FILENAME = "cmexam_grpo_data_summary.json"
RUN_CONFIG_FILENAME = "medical_grpo_run_config.json"


class MedicalGRPOError(ValueError):
    """训练配置或固定实验协议不合法。"""


@dataclass(frozen=True)
class ModelArguments:
    """Base 模型、tokenizer 与既有 PT→SFT adapter 参数。"""

    model_name_or_path: str | None = field(default=None, metadata={"help": "Base 模型路径或标识。"})
    peft_path: str | None = field(default=None, metadata={"help": "正式 PT→SFT adapter 目录。"})
    tokenizer_name_or_path: str | None = None
    cache_dir: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    use_fast_tokenizer: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_name_or_path, str) or not self.model_name_or_path.strip():
            raise MedicalGRPOError("必须提供非空 model_name_or_path。")
        resolve_torch_dtype(self.torch_dtype)


@dataclass(frozen=True)
class CMExamGRPODataArguments:
    """严格去污染训练数据及确定性抽样参数。"""

    train_file: str = str(DEFAULT_CMEXAM_GRPO_TRAIN_FILE)
    max_samples: int | None = None
    data_seed: int = 42
    shuffle_before_select: bool = True


@dataclass(frozen=True)
class CMExamRewardArguments:
    """CMExam exact-set 与 answer-only 奖励参数。"""

    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    format_reward: float = 0.1
    invalid_penalty: float = -0.1

    def to_config(self) -> CMExamRewardConfig:
        return CMExamRewardConfig(
            correct_reward=self.correct_reward,
            incorrect_reward=self.incorrect_reward,
            format_reward=self.format_reward,
            invalid_penalty=self.invalid_penalty,
        )


@dataclass(frozen=True)
class CMExamGRPORuntimeArguments:
    """不与 GRPOConfig 重复的运行和保存参数。"""

    dry_run: bool = False
    overwrite_output_dir: bool = False
    save_data_summary: bool = True
    require_thinking_disabled: bool = True


@dataclass
class MedicalGRPOConfig(GRPOConfig):
    """避免与模型/数据参数重复 CLI 名称的 GRPOConfig 薄封装。"""

    # 入口分别由 ModelArguments 和 CMExamGRPODataArguments 管理这两个值。
    trust_remote_code: bool = field(default=False, init=False)
    data_seed: int | None = field(default=None, init=False)
    # dry run 在无 CUDA 的开发机也必须可解析；真实训练预检强制 BF16。
    bf16: bool | None = False


@dataclass(frozen=True)
class TrainableParameterSummary:
    """既有 default adapter 的参数冻结检查结果。"""

    total_parameters: int
    trainable_parameters: int
    trainable_ratio: float
    trainable_parameter_names: tuple[str, ...]


def resolve_torch_dtype(value: str) -> Any:
    """把受限字符串解析为 torch dtype；``auto`` 原样交给 Transformers。"""

    if value == "auto":
        return "auto"
    if value not in {"bfloat16", "float16", "float32"}:
        raise MedicalGRPOError(
            f"torch_dtype={value!r} 非法；只支持 auto、bfloat16、float16、float32。"
        )
    import torch

    return getattr(torch, value)


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[
    ModelArguments,
    CMExamGRPODataArguments,
    CMExamRewardArguments,
    CMExamGRPORuntimeArguments,
    GRPOConfig,
]:
    """使用 HfArgumentParser 解析入口参数。"""

    parser = HfArgumentParser(
        (
            ModelArguments,
            CMExamGRPODataArguments,
            CMExamRewardArguments,
            CMExamGRPORuntimeArguments,
            MedicalGRPOConfig,
        )
    )
    values = parser.parse_args_into_dataclasses(args=list(argv) if argv is not None else None)
    return values  # type: ignore[return-value]


def _is_nonempty_directory(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def resolve_resume_checkpoint(value: str | bool | None, output_dir: Path) -> str | bool | None:
    """解析具体 checkpoint 或 ``latest``，并验证其存在。"""

    if value in (None, False, "", "false", "False"):
        return None
    if value is True or value in {"latest", "true", "True"}:
        latest = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
        if latest is None:
            raise MedicalGRPOError(f"output_dir 中没有可恢复的 checkpoint：{output_dir}")
        return latest
    checkpoint = Path(str(value)).expanduser()
    if not checkpoint.is_dir():
        raise MedicalGRPOError(f"resume_from_checkpoint 目录不存在：{checkpoint}")
    return str(checkpoint)


def validate_runtime_arguments(
    model_args: ModelArguments,
    data_args: CMExamGRPODataArguments,
    reward_args: CMExamRewardArguments,
    runtime_args: CMExamGRPORuntimeArguments,
    training_args: GRPOConfig,
) -> str | bool | None:
    """在加载 tokenizer/model 前执行固定协议和关键超参数预检。"""

    reward_args.to_config()
    # 构造数据配置即可复用其中 max_samples、seed 和路径类型校验；实际路径保护在读取时执行。
    CMExamGRPODataConfig(
        train_file=Path(data_args.train_file),
        max_samples=data_args.max_samples,
        seed=data_args.data_seed,
        shuffle_before_select=data_args.shuffle_before_select,
        require_decontaminated_train=True,
    )
    if training_args.bf16 and training_args.fp16:
        raise MedicalGRPOError("bf16 和 fp16 不能同时启用。")
    if not runtime_args.dry_run and training_args.bf16 is not True:
        raise MedicalGRPOError("固定主实验的真实训练必须启用 bf16=True。")
    if training_args.num_generations is None or training_args.num_generations < 2:
        raise MedicalGRPOError("num_generations 必须是至少 2 的整数。")
    if training_args.max_completion_length is None or training_args.max_completion_length <= 0:
        raise MedicalGRPOError("max_completion_length 必须是正整数。")
    if training_args.beta < 0:
        raise MedicalGRPOError("beta 不能为负数。")
    if training_args.beta != 0:
        raise MedicalGRPOError(
            "固定协议要求 beta=0；TRL 1.8.0 对既有 PEFT adapter 使用 beta>0 会创建第二个 ref adapter。"
        )
    if training_args.learning_rate <= 0:
        raise MedicalGRPOError("learning_rate 必须为正数。")
    if training_args.per_device_train_batch_size <= 0:
        raise MedicalGRPOError("per_device_train_batch_size 必须为正整数。")
    if training_args.gradient_accumulation_steps <= 0:
        raise MedicalGRPOError("gradient_accumulation_steps 必须为正整数。")
    generation_batch_size = training_args.generation_batch_size
    if generation_batch_size is None or generation_batch_size % training_args.num_generations:
        raise MedicalGRPOError(
            "generation_batch_size 必须能被 num_generations 整除；"
            f"实际为 {generation_batch_size} 与 {training_args.num_generations}。"
        )
    if training_args.remove_unused_columns is not False:
        raise MedicalGRPOError("remove_unused_columns 必须显式为 False，以保留 answer_labels 和 id。")
    if training_args.use_vllm:
        raise MedicalGRPOError("固定协议不允许 use_vllm。")
    if training_args.deepspeed is not None:
        raise MedicalGRPOError("固定协议不允许 DeepSpeed。")
    if training_args.use_liger_kernel:
        raise MedicalGRPOError("固定协议不允许 Liger/FlashAttention 类额外内核。")
    if training_args.push_to_hub:
        raise MedicalGRPOError("本入口禁止 push_to_hub。")

    output_dir = Path(training_args.output_dir).expanduser().resolve()
    if not runtime_args.dry_run and not model_args.peft_path:
        raise MedicalGRPOError("真实训练必须提供正式 PT→SFT peft_path。")
    if model_args.peft_path:
        peft_path = Path(model_args.peft_path).expanduser().resolve()
        if output_dir == peft_path:
            raise MedicalGRPOError("output_dir 不能与起始 peft_path 相同，禁止原地训练 adapter。")
    resume = resolve_resume_checkpoint(training_args.resume_from_checkpoint, output_dir)
    if _is_nonempty_directory(output_dir) and resume is None and not runtime_args.overwrite_output_dir:
        raise FileExistsError(
            f"output_dir 已存在且非空：{output_dir}；请显式 overwrite_output_dir 或 resume。"
        )
    if training_args.bf16:
        import torch

        if not torch.cuda.is_available():
            warnings.warn("当前 CUDA 不可用；BF16 正式训练将无法在目标 RTX 4090D 上启动。", RuntimeWarning)
    return resume


def prepare_training_dataset(
    data_args: CMExamGRPODataArguments,
) -> tuple[Dataset, CMExamGRPODataSummary]:
    """调用公共数据模块构造严格去污染训练 Dataset。"""

    config = CMExamGRPODataConfig(
        train_file=Path(data_args.train_file),
        max_samples=data_args.max_samples,
        seed=data_args.data_seed,
        prompt_style="answer_only",
        shuffle_before_select=data_args.shuffle_before_select,
        require_decontaminated_train=True,
    )
    examples, summary = prepare_cmexam_grpo_examples(config)
    if not examples:
        raise MedicalGRPOError("CMExam GRPO 训练 Dataset 为空。")
    dataset = Dataset.from_list(examples)
    required = {"id", "prompt", "answer_labels", "is_multiple_choice", "metadata"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise MedicalGRPOError(f"CMExam GRPO Dataset 缺少字段：{sorted(missing)}")
    return dataset, summary


def build_reward_functions(
    reward_args: CMExamRewardArguments,
) -> tuple[list[RewardFunction], CMExamRewardConfig]:
    """调用公共奖励模块构造固定顺序的三个奖励函数。"""

    config = reward_args.to_config()
    functions = build_cmexam_reward_functions(config)
    expected = [
        "exact_set_correctness_reward",
        "answer_only_format_reward",
        "invalid_answer_penalty",
    ]
    if [function.__name__ for function in functions] != expected:
        raise MedicalGRPOError("CMExam reward functions 顺序不符合固定协议。")
    return functions, config


def load_processing_class(model_args: ModelArguments) -> Any:
    """加载纯文本 tokenizer，并满足 TRL 左填充要求。"""

    source = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        local_files_only=model_args.local_files_only,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
    )
    if tokenizer.eos_token is None:
        raise MedicalGRPOError(f"tokenizer {source!r} 缺少 eos_token。")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if not getattr(tokenizer, "chat_template", None):
        raise MedicalGRPOError(f"tokenizer {source!r} 缺少 chat_template。")
    return tokenizer


def configure_thinking_disabled(
    training_args: Any,
    *,
    tokenizer: Any | None = None,
    dataset: Dataset | None = None,
    required: bool = True,
) -> tuple[Dataset | None, str]:
    """显式关闭 thinking；当前 TRL 优先使用 chat_template_kwargs，兼容安全渲染后备。"""

    if hasattr(training_args, "chat_template_kwargs"):
        kwargs = dict(training_args.chat_template_kwargs or {})
        if kwargs.get("enable_thinking") is True:
            raise MedicalGRPOError("chat_template_kwargs 不能启用 thinking。")
        kwargs["enable_thinking"] = False
        training_args.chat_template_kwargs = kwargs
        return dataset, THINKING_DISABLED_STRATEGY
    if tokenizer is not None and dataset is not None:
        rendered = dataset.map(
            lambda row: {
                "prompt": tokenizer.apply_chat_template(
                    row["prompt"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            }
        )
        return rendered, "预渲染字符串 prompt: enable_thinking=False"
    if required:
        raise MedicalGRPOError("当前 TRL/tokenizer 无法确认 enable_thinking=False，拒绝创建 Trainer。")
    warnings.warn("未能显式确认 thinking 已关闭。", RuntimeWarning)
    return dataset, "未确认"


def _check_adapter_files(peft_path: Path) -> dict[str, Any]:
    if not peft_path.is_dir():
        raise FileNotFoundError(f"peft_path 目录不存在：{peft_path}")
    config_path = peft_path / "adapter_config.json"
    if not config_path.is_file():
        raise MedicalGRPOError(f"adapter 配置缺失：{config_path}")
    weights = [peft_path / "adapter_model.safetensors", peft_path / "adapter_model.bin"]
    if not any(path.is_file() for path in weights):
        raise MedicalGRPOError(
            f"adapter 权重缺失：需要 {weights[0].name} 或 {weights[1].name}（目录 {peft_path}）。"
        )
    try:
        content = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MedicalGRPOError(f"adapter_config.json 无法读取：{config_path}：{exc}") from exc
    if not isinstance(content, dict):
        raise MedicalGRPOError(f"adapter_config.json 必须是 JSON 对象：{config_path}")
    return content


def _validate_adapter_base(config: dict[str, Any], model_name_or_path: str) -> None:
    declared = config.get("base_model_name_or_path")
    if not isinstance(declared, str) or not declared.strip():
        warnings.warn("adapter_config.json 未声明 base_model_name_or_path。", RuntimeWarning)
        return
    declared_name = Path(declared.rstrip("/")).name.casefold()
    requested_name = Path(model_name_or_path.rstrip("/")).name.casefold()
    if declared_name and requested_name and declared_name != requested_name:
        raise MedicalGRPOError(
            "adapter 声明的 Base 与 model_name_or_path 明显不兼容："
            f"{declared!r} != {model_name_or_path!r}。"
        )


def validate_trainable_adapter(model: Any) -> TrainableParameterSummary:
    """确认只有现有 default adapter 参数可训练，基座参数保持冻结。"""

    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, dict) or "default" not in peft_config:
        raise MedicalGRPOError("PeftModel 缺少 default adapter 配置。")
    total = 0
    trainable = 0
    names: list[str] = []
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
            names.append(name)
            if ".default." not in name and not name.endswith(".default"):
                unexpected.append(name)
    if trainable <= 0:
        raise MedicalGRPOError("default adapter 没有任何 requires_grad=True 的参数。")
    if unexpected:
        raise MedicalGRPOError(f"Base 参数被意外解冻或非 default adapter 可训练：{unexpected[0]}")
    if total <= 0 or trainable >= total:
        raise MedicalGRPOError("可训练参数比例不合理，Base 权重可能未冻结。")
    summary = TrainableParameterSummary(total, trainable, trainable / total, tuple(names))
    LOGGER.info(
        "参数统计：trainable=%d total=%d ratio=%.6f",
        summary.trainable_parameters,
        summary.total_parameters,
        summary.trainable_ratio,
    )
    return summary


def load_trainable_peft_model(
    model_args: ModelArguments,
    training_args: GRPOConfig,
) -> tuple[Any, TrainableParameterSummary]:
    """只加载一次 Base，并以可训练方式挂载已有 default adapter。"""

    if not model_args.peft_path:
        raise MedicalGRPOError("真实训练必须提供 peft_path。")
    peft_path = Path(model_args.peft_path).expanduser().resolve()
    adapter_config = _check_adapter_files(peft_path)
    _validate_adapter_base(adapter_config, str(model_args.model_name_or_path))
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=resolve_torch_dtype(model_args.torch_dtype),
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        local_files_only=model_args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(peft_path),
        adapter_name="default",
        is_trainable=True,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
    )
    model.set_adapter("default")
    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, validate_trainable_adapter(model)


def build_grpo_trainer(
    *,
    model: Any,
    training_args: GRPOConfig,
    train_dataset: Dataset,
    reward_functions: list[RewardFunction],
    processing_class: Any,
) -> GRPOTrainer:
    """构造原生 TRL Trainer；不传 eval、ref 或任何新 PEFT 配置。"""

    if "answer_labels" not in train_dataset.column_names or "id" not in train_dataset.column_names:
        raise MedicalGRPOError("Trainer Dataset 必须保留 answer_labels 和 id。")
    if training_args.remove_unused_columns is not False:
        raise MedicalGRPOError("创建 Trainer 前 remove_unused_columns 必须为 False。")
    return GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=reward_functions,
        processing_class=processing_class,
    )


def _atomic_write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(content, file, ensure_ascii=False, indent=2, default=str)
            file.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_run_metadata(
    output_dir: Path,
    *,
    model_args: ModelArguments,
    data_args: CMExamGRPODataArguments,
    reward_config: CMExamRewardConfig,
    runtime_args: CMExamGRPORuntimeArguments,
    training_args: GRPOConfig,
    summary: CMExamGRPODataSummary,
    thinking_strategy: str,
    parameter_summary: TrainableParameterSummary | None,
) -> None:
    """原子保存不含凭据和环境变量的数据摘要与运行配置。"""

    if runtime_args.save_data_summary:
        _atomic_write_json(output_dir / DATA_SUMMARY_FILENAME, summary.to_dict())
    run_config = {
        "model": asdict(model_args),
        "data": asdict(data_args),
        "reward": asdict(reward_config),
        "runtime": asdict(runtime_args),
        "training": training_args.to_dict(),
        "thinking_disabled_strategy": thinking_strategy,
        "parameter_summary": asdict(parameter_summary) if parameter_summary else None,
        "versions": {
            "trl": _version("trl"),
            "transformers": _version("transformers"),
            "peft": _version("peft"),
        },
    }
    _atomic_write_json(output_dir / RUN_CONFIG_FILENAME, run_config)


def _safe_prompt_preview(dataset: Dataset) -> dict[str, Any]:
    row = dataset[0]
    return {"id": row["id"], "prompt": row["prompt"]}


def _log_configuration(
    model_args: ModelArguments,
    data_args: CMExamGRPODataArguments,
    training_args: GRPOConfig,
    summary: CMExamGRPODataSummary,
    reward_functions: list[RewardFunction],
    thinking_strategy: str,
) -> None:
    LOGGER.info("Base 模型：%s", model_args.model_name_or_path)
    LOGGER.info("PT→SFT adapter：%s", model_args.peft_path)
    LOGGER.info("训练数据：%s；selected=%d", data_args.train_file, summary.selected_records)
    LOGGER.info("selected_ids：%s", list(summary.selected_ids))
    LOGGER.info("reward functions：%s", [function.__name__ for function in reward_functions])
    LOGGER.info("thinking 关闭策略：%s", thinking_strategy)
    LOGGER.info(
        "GRPO：output_dir=%s seed=%s batch=%s grad_accum=%s generations=%s completion=%s beta=%s",
        training_args.output_dir,
        training_args.seed,
        training_args.per_device_train_batch_size,
        training_args.gradient_accumulation_steps,
        training_args.num_generations,
        training_args.max_completion_length,
        training_args.beta,
    )
    LOGGER.info(
        "版本：trl=%s transformers=%s peft=%s",
        _version("trl"), _version("transformers"), _version("peft"),
    )


def run_training(
    model_args: ModelArguments,
    data_args: CMExamGRPODataArguments,
    reward_args: CMExamRewardArguments,
    runtime_args: CMExamGRPORuntimeArguments,
    training_args: GRPOConfig,
) -> int:
    """执行安全 dry run 或真实的既有 adapter GRPO 续训。"""

    resume = validate_runtime_arguments(model_args, data_args, reward_args, runtime_args, training_args)
    dataset, summary = prepare_training_dataset(data_args)
    reward_functions, reward_config = build_reward_functions(reward_args)
    dataset, thinking_strategy = configure_thinking_disabled(
        training_args,
        dataset=dataset,
        required=runtime_args.require_thinking_disabled,
    )
    assert dataset is not None
    _log_configuration(
        model_args, data_args, training_args, summary, reward_functions, thinking_strategy
    )
    if runtime_args.dry_run:
        LOGGER.warning("dry run 按协议不访问也不检查 adapter 文件：%s", model_args.peft_path)
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        print(json.dumps(_safe_prompt_preview(dataset), ensure_ascii=False, indent=2))
        print("reward_functions:", [function.__name__ for function in reward_functions])
        print("thinking_disabled_strategy:", thinking_strategy)
        print("model_name_or_path:", model_args.model_name_or_path)
        print("peft_path:", model_args.peft_path)
        print(
            "grpo_config:",
            {
                "output_dir": training_args.output_dir,
                "seed": training_args.seed,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "num_generations": training_args.num_generations,
                "max_completion_length": training_args.max_completion_length,
                "beta": training_args.beta,
                "remove_unused_columns": training_args.remove_unused_columns,
            },
        )
        return 0

    output_dir = Path(training_args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(training_args.seed)
    tokenizer = load_processing_class(model_args)
    dataset, thinking_strategy = configure_thinking_disabled(
        training_args,
        tokenizer=tokenizer,
        dataset=dataset,
        required=runtime_args.require_thinking_disabled,
    )
    assert dataset is not None
    model, parameter_summary = load_trainable_peft_model(model_args, training_args)
    trainer = build_grpo_trainer(
        model=model,
        training_args=training_args,
        train_dataset=dataset,
        reward_functions=reward_functions,
        processing_class=tokenizer,
    )
    try:
        train_result = trainer.train(resume_from_checkpoint=resume)
    except RuntimeError as exc:
        if "out of memory" in str(exc).casefold():
            raise RuntimeError(
                "CUDA 显存不足；请显式减小 generation batch、completion length 或 batch size。"
            ) from exc
        raise
    metrics = dict(train_result.metrics)
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_run_metadata(
        output_dir,
        model_args=model_args,
        data_args=data_args,
        reward_config=reward_config,
        runtime_args=runtime_args,
        training_args=training_args,
        summary=summary,
        thinking_strategy=thinking_strategy,
        parameter_summary=parameter_summary,
    )
    required_files = [output_dir / "adapter_config.json", output_dir / "trainer_state.json"]
    if not all(path.is_file() for path in required_files) or not any(
        (output_dir / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError(f"训练保存结果不完整：{output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口，成功返回 0。"""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run_training(*parse_args(argv))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (MedicalGRPOError, FileNotFoundError, FileExistsError, OSError, TypeError) as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)
