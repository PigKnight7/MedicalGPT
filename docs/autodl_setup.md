# MedicalGPT AutoDL 环境与实验操作手册

本文面向单张支持 BF16 的 AutoDL GPU，实验基座为 `Qwen/Qwen3.5-2B-Base`，训练方式为普通 LoRA。当前阶段不安装 FlashAttention、bitsandbytes、DeepSpeed 或 vLLM，也不使用量化训练。

> AutoDL 开始计费后不要再临时研究脚本。租卡前应完成代码检查、数据处理、打包、SHA-256 校验和全部本地静态测试。模型、processed 数据、checkpoint、日志和评估结果都应放在 `/root/autodl-tmp` 持久化盘。

## 1. 租卡前的本地准备

在本地仓库运行：

```bash
bash -n scripts/setup_autodl.sh
bash -n scripts/download_qwen35_2b_base.sh
DRY_RUN=1 bash scripts/setup_autodl.sh
DRY_RUN=1 MODEL_DIR=/tmp/qwen35-test-model bash scripts/download_qwen35_2b_base.sh
python -m pytest -q
```

原始约 2GB 数据无需上传。只上传已经由仓库工具生成并验证过的：

```text
~/datasets/medicalgpt/processed/pt
~/datasets/medicalgpt/processed/sft
~/datasets/medicalgpt/processed/cmexam
```

本地打包：

```bash
tar -C "${HOME}/datasets/medicalgpt" -czf medicalgpt_processed_data.tar.gz \
  processed/pt processed/sft processed/cmexam
sha256sum medicalgpt_processed_data.tar.gz > medicalgpt_processed_data.tar.gz.sha256
```

不要把原始数据、模型权重或认证信息提交到 Git。Hugging Face 登录可使用 `hf auth login` 或临时环境变量，但绝不能把 Token 写进代码、Shell 历史示例、配置或日志。

## 2. 租卡后检查与项目克隆

选择已经带有合适 CUDA 版 PyTorch 的镜像，不要仅根据 `nvidia-smi` 显示的 CUDA Version 猜测 wheel。启动后先检查：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print(torch.cuda.is_bf16_supported())
PY
df -h /root/autodl-tmp
```

克隆项目到代码盘或持久化盘：

```bash
git clone <YOUR_REPOSITORY_URL> MedicalGPT
cd MedicalGPT
git rev-parse HEAD
git status --short
```

## 3. 环境安装、激活与检查

先预览，不产生任何写入：

```bash
DRY_RUN=1 bash scripts/setup_autodl.sh
```

只检查当前镜像，不创建虚拟环境或安装依赖：

```bash
CHECK_ONLY=1 PYTHON_BIN=python bash scripts/setup_autodl.sh
```

默认准备环境：

```bash
CREATE_VENV=1 \
USE_SYSTEM_SITE_PACKAGES=1 \
INSTALL_DEPS=1 \
INSTALL_TORCH=0 \
bash scripts/setup_autodl.sh
```

脚本默认复用镜像中可用的 CUDA/BF16 PyTorch，不会重装 torch。若 torch 缺失或只有 CPU 版本，脚本会停止。优先更换正确的 AutoDL 镜像；只有核对 PyTorch 官方说明后，才可显式设置：

```bash
INSTALL_TORCH=1 \
TORCH_INDEX_URL='<OFFICIAL_CONFIRMED_INDEX_URL>' \
bash scripts/setup_autodl.sh
```

脚本不会从驱动版本推断安装地址。不要在当前本地 CPU 环境执行上述安装。

setup 结束后，父 Shell 不会自动激活环境，必须执行：

```bash
source /root/autodl-tmp/venvs/medicalgpt/bin/activate
source /root/autodl-tmp/environment/medicalgpt/runtime_env.sh
```

`runtime_env.sh` 只包含缓存路径和 `TOKENIZERS_PARALLELISM=false`，不包含任何 Token。环境报告与实际依赖锁位于：

```text
/root/autodl-tmp/environment/medicalgpt/autodl_environment.json
/root/autodl-tmp/environment/medicalgpt/requirements-lock-autodl.txt
```

## 4. 下载并验证 Base 模型

先预览：

```bash
DRY_RUN=1 bash scripts/download_qwen35_2b_base.sh
```

正式下载公共模型：

```bash
MODEL_ID=Qwen/Qwen3.5-2B-Base \
MODEL_REVISION=main \
MODEL_DIR=/root/autodl-tmp/models/Qwen3.5-2B-Base \
VERIFY_SHA256=1 \
bash scripts/download_qwen35_2b_base.sh
```

脚本下载到 `.partial`，可重复执行以继续下载。完整性、离线 config/tokenizer 和中文编码验证通过后才改名为最终目录。完整目录默认跳过；需要重新下载时设置 `FORCE=1`，旧目录会移动到带时间戳的备份，而不是删除。查看：

```bash
cat /root/autodl-tmp/models/Qwen3.5-2B-Base/model_manifest.json
cd /root/autodl-tmp/models/Qwen3.5-2B-Base
sha256sum -c sha256sums.txt
```

## 5. 上传、解压和校验 processed 数据

将两个打包文件上传到 `/root/autodl-tmp/datasets/medicalgpt`，然后：

```bash
cd /root/autodl-tmp/datasets/medicalgpt
sha256sum -c medicalgpt_processed_data.tar.gz.sha256
tar -xzf medicalgpt_processed_data.tar.gz
du -sh processed/pt processed/sft processed/cmexam
find processed -type f -name '*.jsonl' -print -exec wc -l {} \;
```

必须人工核对：

- PT train、validation 行数；
- SFT train、validation 行数；
- CMExam official 与 decontaminated 各 split 行数；
- 压缩包 SHA-256 和解压后文件大小；
- JSONL 文件不是空文件。

## 6. 全仓库测试

```bash
cd /path/to/MedicalGPT
source /root/autodl-tmp/venvs/medicalgpt/bin/activate
source /root/autodl-tmp/environment/medicalgpt/runtime_env.sh
python -m pip check
python -m pytest -q
```

测试失败时不要启动训练。

## 7. Base 评估 smoke

### 7.1 Base CMExam validation smoke

只使用 validation；test 只能用于最终锁定配置后的评估。

```bash
python evaluation/eval_cmexam.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen3.5-2B-Base \
  --data_root /root/autodl-tmp/datasets/medicalgpt/processed/cmexam/official \
  --split validation \
  --output_dir /root/autodl-tmp/evaluations/medicalgpt/cmexam/base_smoke \
  --max_samples 20 \
  --batch_size 1 \
  --device cuda \
  --torch_dtype bfloat16 \
  --local_files_only
```

### 7.2 Base PT perplexity smoke

```bash
python evaluation/eval_pt_perplexity.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen3.5-2B-Base \
  --validation_file_dir /root/autodl-tmp/datasets/medicalgpt/processed/pt/validation \
  --output_dir /root/autodl-tmp/evaluations/medicalgpt/pt_perplexity/base_smoke \
  --block_size 1024 \
  --batch_size 1 \
  --max_blocks 20 \
  --device cuda \
  --torch_dtype bfloat16 \
  --local_files_only
```

## 8. PT smoke、正式训练与评估

先运行 20-step PT smoke：

```bash
RUN_ID=pt-smoke-$(date +%Y%m%d-%H%M%S) bash scripts/run_pt_medical_2b_smoke.sh
```

smoke 的模型加载、训练、评估和保存均通过后才能正式训练：

```bash
RUN_ID=pt-$(date +%Y%m%d-%H%M%S) bash scripts/run_pt_medical_2b.sh
```

记下实际 adapter 目录，例如：

```bash
PT_ADAPTER=/root/autodl-tmp/outputs/medicalgpt/pt/<RUN_ID>
```

用同一 PT validation 和完全相同参数评估 PT adapter：

```bash
python evaluation/eval_pt_perplexity.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen3.5-2B-Base \
  --peft_path "${PT_ADAPTER}" \
  --validation_file_dir /root/autodl-tmp/datasets/medicalgpt/processed/pt/validation \
  --output_dir /root/autodl-tmp/evaluations/medicalgpt/pt_perplexity/pt \
  --block_size 1024 --batch_size 1 \
  --device cuda --torch_dtype bfloat16 --local_files_only
```

## 9. 两条 SFT 路线

### 9.1 Base→SFT smoke

```bash
RUN_ID=sft-base-smoke-$(date +%Y%m%d-%H%M%S) bash scripts/run_sft_base_medical_2b_smoke.sh
```

### 9.2 PT→SFT smoke

```bash
PT_ADAPTER_PATH="${PT_ADAPTER}" \
RUN_ID=sft-pt-smoke-$(date +%Y%m%d-%H%M%S) \
bash scripts/run_sft_medical_2b_smoke.sh
```

两条 smoke 都通过后，分别运行正式训练：

```bash
RUN_ID=sft-base-$(date +%Y%m%d-%H%M%S) bash scripts/run_sft_base_medical_2b.sh

PT_ADAPTER_PATH="${PT_ADAPTER}" \
RUN_ID=sft-pt-$(date +%Y%m%d-%H%M%S) \
bash scripts/run_sft_medical_2b.sh
```

不要复用非空输出目录；保存每次 RUN_ID、环境报告、Git commit 和日志。

## 10. CMExam 统一评估

对 Base→SFT 和 PT→SFT adapter 分别运行完全相同的 validation 配置：

```bash
python evaluation/eval_cmexam.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen3.5-2B-Base \
  --peft_path /root/autodl-tmp/outputs/medicalgpt/sft_base/<RUN_ID> \
  --data_root /root/autodl-tmp/datasets/medicalgpt/processed/cmexam/official \
  --split validation \
  --output_dir /root/autodl-tmp/evaluations/medicalgpt/cmexam/sft_base_validation \
  --batch_size 1 --device cuda --torch_dtype bfloat16 --local_files_only
```

将 `--peft_path` 和输出目录替换为 PT→SFT 产物再运行一次。只有完成开发、固定模型 checkpoint、prompt、解码参数和指标后，才允许增加：

```text
--split test --allow_test_evaluation
```

## 11. 停机前检查

确认以下内容都在持久化盘：

```bash
du -sh /root/autodl-tmp/models /root/autodl-tmp/datasets \
  /root/autodl-tmp/outputs /root/autodl-tmp/logs \
  /root/autodl-tmp/evaluations /root/autodl-tmp/environment
```

保存环境报告、manifest、SHA-256、训练日志和评估结果。再次确认没有认证 Token 被写入仓库或日志，然后再停止实例。
