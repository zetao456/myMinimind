# LoRA、DPO 与 GRPO

[文档首页](../README.md) · [项目首页](../../README.md)

三个阶段都使用原生 PyTorch，不依赖 TRL Trainer。默认从 `out/full_sft_768.pth` 开始；已有模型和 tokenizer 可直接复用。训练脚本的默认模型、输出及断点目录按仓库根目录定位，从根目录或 `trainer/` 启动均可。用户传入的相对路径按当前工作目录解释。

## 1. LoRA SFT

```bash
uv run python trainer/train_lora.py \
  --data_path dataset/sft_t2t_mini.jsonl \
  --batch_size 8 --accumulation_steps 4 --epochs 2 \
  --learning_rate 1e-4 --cache_data \
  --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05
```

数据沿用现有 SFT 格式；仅 assistant 回复计算监督损失：

```json
{"conversations":[{"role":"user","content":"你是谁？"},{"role":"assistant","content":"我是 MiniMind。"}]}
```

默认对 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` 注入适配器，支持 GQA 中非方阵的 K/V 投影。用 `--lora_targets q_proj,v_proj` 缩小范围。基础模型冻结，只更新 LoRA 的 A、B；增量为 `alpha/rank * B(A(dropout(x)))`，B 零初始化。

`--cache_data` 复用现有 int16 memmap 缓存；缓存应与当前 tokenizer、数据和序列长度一致。在线数据处理每次可随机添加 system 提示；缓存将这种随机增强固定为缓存建立时的结果。缓存目录沿用现有规则，不自动识别数据/tokenizer 内容变化。

输出 `out/lora_sft_768.pth` 包含适配器参数及各层的 rank、alpha、dropout 配置。加载时自动安装目标层；旧版无配置的适配器也能加载，按旧约定采用 `alpha=rank`、dropout=0。

```bash
uv run python eval_llm.py --weight full_sft --lora_weight lora_sft --max_new_tokens 256
```

LoRA 必须配合训练时相同的基础权重。继续微调现有适配器可加 `--lora_path out/lora_sft_768.pth`；加载的适配器配置优先于命令行 LoRA 配置。

合并为无需适配器的普通权重：

```bash
uv run python trainer/merge_lora.py \
  --base_path out/full_sft_768.pth \
  --lora_path out/lora_sft_768.pth \
  --output_path out/lora_merged_768.pth
uv run python eval_llm.py --weight lora_merged --max_new_tokens 256
```

对比实验中可加 `--preference_sft`，将原始偏好对的 chosen 直接作为 SFT 监督，复用 DPO 的 token/标签且不进行随机提示增强；数据格式与 `--max_prompt_len` 同 DPO。

## 2. DPO

参考 [DPO 原论文](https://arxiv.org/abs/2305.18290)。固定参考策略，通过 chosen/rejected 的回复序列概率进行偏好学习：

```text
loss = -logsigmoid(beta * [(log π(chosen) - log πref(chosen))
                         - (log π(rejected) - log πref(rejected))])
```

回复概率按 token **求和**，不做长度平均；提示、assistant 前缀和 padding 不计入概率，回复结束标记参与。默认参考模型是训练开始时的策略副本，全程冻结；也可用 `--reference_weight full_sft` 指定 `base_dir` 中另一份完整权重。

支持字符串或消息列表 prompt：

```json
{"prompt":"1+1等于多少？","chosen":"1+1等于2。","rejected":"1+1等于3。"}
{"prompt":[{"role":"system","content":"你是一个助手。"},{"role":"user","content":"1+1？"}],"chosen":"2。","rejected":"3。"}
```

也支持 chosen/rejected 都为完整对话列表，要求两者具有完全相同的历史，最后一条均为 assistant 回复，见 `dataset/examples/dpo.jsonl`。这种格式不需要额外的 prompt 字段，并保留 assistant 的 reasoning_content。DPO 提示止于模板的开放思考前缀，思考内容和后续回答一起评分。长提示保留尾部，回复截断到剩余空间；若偏好对截断后完全相同则报错，避免无效训练。

```bash
uv run python trainer/train_dpo.py \
  --data_path dataset/my_preferences.jsonl \
  --batch_size 4 --accumulation_steps 4 \
  --max_seq_len 768 --max_prompt_len 384 \
  --learning_rate 1e-6 --beta 0.1
uv run python eval_llm.py --weight dpo --max_new_tokens 256
```

加 `--use_lora` 可以只训练适配器：

```bash
uv run python trainer/train_dpo.py \
  --data_path dataset/my_preferences.jsonl --use_lora \
  --save_weight lora_dpo --learning_rate 1e-5
uv run python eval_llm.py --weight full_sft --lora_weight lora_dpo
```

指标包含 loss、chosen/rejected 隐式奖励、偏好准确率。首次策略与参考相同时，loss 约为 `log(2)`，严格大于定义的偏好准确率为 0，均属正常。

## 3. GRPO

实现 [DeepSeekMath](https://arxiv.org/html/2402.03300v3#S4) 的 outcome-supervised GRPO：每个 prompt 采样 G 个回复，组内标准化奖励作为回复内各 token 的优势，使用 PPO 比率裁剪和独立的参考策略 KL 惩罚，不训练 critic。

```text
advantage_i = (reward_i - group_mean) / (group_std + 1e-4)
ratio_it = exp(log π_it - log πold_it)
KL_it = exp(log πref_it - log π_it) - (log πref_it - log π_it) - 1
loss = mean_sequences(mean_valid_tokens(-min(ratio*A, clip(ratio)*A) + beta*KL))
```

组内标准差使用 population std（`unbiased=False`）。保留第一个 EOS，屏蔽其后自动填充的 token。没有 EOS 的回复使用实际采样到的全部 token。参考模型默认固定为训练起点，不随策略更新。`--num_iterations` 可对固定 rollout 做多次更新；old log-prob 和奖励在这些更新中保持不变。

为确保采样分布和 PPO 中的旧策略概率一致，使用 temperature=1、top-p=1、top-k=0，无重复惩罚。GRPO 阶段关闭 dropout；rollout、参考打分与更新均用 FP32，`--dtype` 不改变 GRPO 精度。这是当前实现对一致性的选择，吞吐和显存仍有优化空间。

### 答案匹配奖励

```json
{"prompt":"计算1+1，将最终答案写在<answer>和</answer>之间。","answer":"2"}
{"prompt":[{"role":"user","content":"计算2+2，只输出数字。"}],"answer":["4","4.0"]}
```

默认 `exact_match` 从 `<answer>...</answer>`、`####` 后缀或简单 `\boxed{...}` 提取答案；没有这些标记则使用完整回复。进行 Unicode NFKC、大小写和空白归一化，再精确匹配。它不是符号数学验证器，也不执行生成的代码。需要接受多个等价答案时，使用 answer 列表或自定义奖励。

```bash
uv run python trainer/train_grpo.py \
  --data_path dataset/my_math_prompts.jsonl \
  --batch_size 1 --accumulation_steps 4 \
  --group_size 4 --num_iterations 1 \
  --max_seq_len 768 --max_prompt_len 384 --max_new_tokens 256 \
  --learning_rate 1e-6 --beta 0.04 --clip_epsilon 0.2
uv run python eval_llm.py --weight grpo --max_new_tokens 256
```

`max_prompt_len + max_new_tokens` 必须不超过 `max_seq_len`；group size 至少为 2。batch size 表示 prompt 数，实际生成回复数还要乘以 group size。也支持 `--use_lora` 和 `--lora_path`。加 `--open_thinking` 将提示止于开放的 `<think>` 前缀，让模型生成思考与答案；默认提示带完整空思考块，直接生成回答。

### 自定义奖励

在仓库根目录创建 `my_rewards.py`：

```python
def reward(prompts, completions, records):
    # prompts: 与每条回复对应的 chat messages
    # completions: 解码后的回复字符串
    # records: 原始 JSONL 行，保留 answer 和任务自定义字段
    # 按输入顺序返回每条回复对应的有限标量。
    return [float(text.strip() == str(row["answer"]).strip())
            for text, row in zip(completions, records)]
```

训练时加 `--reward_fn my_rewards:reward`。同一 prompt 的 G 条回复相邻。模块须可通过 Python import 找到，脚本默认已将仓库根目录加入搜索路径。自定义奖励不要求数据存在 answer 字段。

也可加 `--reward_model_path /absolute/path/to/local_reward_model` 使用现有 `LMForRewardModel` 接口；奖励模型需要实现 `get_score(tokenizer, messages)`，输出按现有接口截断在 [-3, 3]。该路径优先于 reward_fn，加载采用现有 `trust_remote_code=True` 约定。普通分类模型需要自行适配为自定义奖励。

指标含平均 reward、KL、clip fraction、回复长度和 `zero_std_groups`。**同组所有奖励相同时优势为零，不会产生偏好学习信号**；尤其是小模型配稀疏答案奖励时，应检查这一指标，再决定增加 group size、改进任务数据或奖励设计。

## 4. 公共训练能力

- 原生 AdamW、余弦学习率、梯度累积、裁剪。
- LoRA SFT / DPO 支持 CUDA BF16/FP16 autocast；CPU 使用 FP32。
- 全模型训练输出普通半精度 state dict；LoRA 输出携带配置的适配器。
- 输出和续训状态使用临时文件 + 原子替换。
- 每次成功的优化器更新向 `out/<save_weight>_<hidden_size>[_moe].metrics.jsonl` 写入指标，包含 Loss、学习率、吞吐、耗时及显存；绘图与口径见 [指标说明](../experiments/training_metrics.md)。
- 断点包含 FP32 模型、优化器、GradScaler、固定参考策略、epoch/batch/update 和各 rank 的 Python/NumPy/Torch RNG。
- `--from_resume 1` 从 `checkpoint_dir` 恢复。保留原命令参数，仅增加续训开关；允许调整 max_steps、日志/保存间隔、设备和 num_workers，其余配置必须一致。
- 默认 `num_workers=0` 可精确恢复在线 SFT 随机预处理；多 worker 的预取和 worker RNG 不保存，续训不保证逐参数一致。缓存数据及确定性偏好数据不受随机增强影响。
- 保存位于完成梯度累积窗口和完整 GRPO rollout 更新之后。末尾不足窗口也更新，并按实际样本数加权。`save_interval` 按优化器更新计数，在完整 rollout 边界保存。
- `--max_steps` 适合短跑验证，0 表示完整训练；GRPO 的非零 max_steps 必须是 num_iterations 的倍数。

多卡：

```bash
uv run torchrun --nproc_per_node=2 trainer/train_dpo.py \
  --data_path dataset/my_preferences.jsonl --batch_size 4
```

使用 DDP、DistributedSampler，日志在 rank 0 写入并对各卡指标取平均。续训要求相同 world size；数据较少且不能整除 world size 时，DistributedSampler 按标准行为补齐样本。改变 world size 应从导出的普通权重/适配器开启新训练。

## 5. 测试与边界

仓库提供三个微型示例，位于 `dataset/examples/`，仅供验证管线，不能用于评价模型训练效果。

```bash
uv run python -m unittest discover -s tests -v
```

测试覆盖适配器冻结、非方阵注入、配置加载、旧格式兼容、合并等价性、DPO 标签与梯度方向、GRPO 优势/裁剪/KL/EOS，以及真实 CLI 的训练、保存、生成和断点续训等价性。测试中的人为交替奖励只用于确保 GRPO 更新路径有非零梯度，不能用于真实训练。

自动测试覆盖 CPU 小模型训练、保存加载与断点恢复；真实数据上的训练效果对比尚未完成。大规模训练收敛及外部奖励模型权重的兼容性需针对具体任务验证。数据和训练预算由调用者提供。
