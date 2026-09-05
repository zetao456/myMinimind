# SFT（监督微调）代码完整解读

## 1. 整体架构：SFT 在训练流程中的位置

```
Pretrain（预训练）  →  Full SFT（全量微调）  →  LoRA SFT / RLHF / DPO ...
     ↓                       ↓
  学习"语言"              学习"对话格式+遵循指令"
```

**训练脚本**: `trainer/train_full_sft.py`
**数据集类**: `dataset/lm_dataset.py` → `SFTDataset`
**推理评测**: `eval_llm.py`（默认加载 `full_sft` 权重）
**工具函数**: `trainer/trainer_utils.py`（`init_model`, `lm_checkpoint` 等）

---

## 2. SFTDataset 数据管道 (`dataset/lm_dataset.py:56-112`)

### 2.1 数据格式

SFT 数据是 JSONL，每行为一个**多轮对话**：

```json
{
  "conversations": [
    {"role": "system", "content": "你是minimind..."},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的？", "reasoning_content": ""},
    {"role": "user", "content": "解释一下Python"},
    {"role": "assistant", "content": "Python是一种...", "reasoning_content": ""}
  ]
}
```

每条消息可包含 5 个字段：`role`, `content`, `reasoning_content`, `tools`, `tool_calls`。
Pretrain 数据格式完全不同，只是一个简单的 `{"text": "..."}`。

### 2.2 数据预处理流水线

`__getitem__()` 调用链：

```
samples[index] → pre_processing_chat() → create_chat_prompt() → post_processing_chat() → tokenize → generate_labels()
```

#### 步骤 1: `pre_processing_chat()` (line 8-28)

```python
def pre_processing_chat(conversations, add_system_ratio=0.2):
```

- 如果对话中有 `tools` 字段（function calling 数据），**跳过所有处理直接返回**，保持原始格式
- 否则：**20% 概率**在对话开头随机插入一条 system prompt
- 目的：有些数据自带 system prompt，有些没有——这种随机增强让模型习惯两种场景

内置了 10 个 system prompt 模板（中英文各 5 个），例如：
- "你是minimind，一个小巧但有用的语言模型。"
- "You are a helpful AI assistant."

#### 步骤 2: `create_chat_prompt()` (line 69-84)

```python
def create_chat_prompt(self, conversations):
```

把对话 messages 转换成模型训练用的文本字符串。核心调用：

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
```

关键参数 `add_generation_prompt=False`：不加生成提示，这样 labels 才会包含 assistant 的回复内容。

渲染后的文本格式（ChatML 格式）：
```
<|im_start|>system
你是minimind<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮你的？<|im_end|>
```

对于有 `tools` 的对话，`apply_chat_template` 会自动处理 tool call 的渲染。

#### 步骤 3: `post_processing_chat()` (line 30-34)

```python
def post_processing_chat(prompt_content, empty_think_ratio=0.2):
```

- 80% 概率删除 `<think>\n\n</think>\n\n` 空思考标签（含换行）
- 这些空标签来自 **DeepSeek-R1 等推理模型蒸馏数据**中的无用残留
- 只删除"完全空白"的 think 标签，有内容的 think 不受影响

#### 步骤 4: Tokenize & Padding

```python
input_ids = tokenizer(prompt).input_ids[:self.max_length]
input_ids += [tokenizer.pad_token_id] * (self.max_length - len(input_ids))
```

标准左填充到 `max_length`（默认 768）。

#### 步骤 5: `generate_labels()` —— 核心 (line 86-102)

**这是 SFT 与 Pretrain 最本质的区别。**

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)     # 全部初始化为 -100（被 loss 忽略）
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:  # 找到 "assistant\n"
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:  # 找到 "\n<|im_end|>\n"
                    break
                end += 1
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]  # 对 assistant 回复区域设置真实标签
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

**逻辑图解**：

```
[<|im_start|>system ... <|im_end|> <|im_start|>user ... <|im_end|> <|im_start|>assistant 回复内容 <|im_end|>]
  ↑ labels=-100                          ↑ labels=-100                  ↑ labels=真实token，参与loss
```

用 `self.bos_id`（即 `<|im_start|>assistant\n` 的 token IDs）作为锚点，找出每个 assistant 回复的起止位置：
- `bos_id` = `tokenizer('<|im_start|>assistant\n').input_ids` → 约 3 个 token
- `eos_id` = `tokenizer('<|im_end|>\n').input_ids` → 约 3 个 token

**为什么这样做？** 我们不希望模型学会"预测用户要说什么"，只希望模型学会"给定对话历史，如何生成好的回复"。`-100` 是 `cross_entropy` 的 `ignore_index`，loss 计算时会跳过。

对比 PretrainDataset（line 36-54）的 labels 生成：
```python
labels = input_ids.clone()
labels[input_ids == tokenizer.pad_token_id] = -100  # 只 mask padding，其余全部参与 loss
```
Pretrain 的做法是**整段文本都学**——因为预训练阶段只需要模型学会"语言"，不需要区分角色。

---

## 3. 训练脚本 `trainer/train_full_sft.py` 要点

### 3.1 与 `train_pretrain.py` 的区别

训练循环代码完全一致，只有**默认参数**不同：

| 参数 | Pretrain | Full SFT | 原因 |
|---|---|---|---|
| `learning_rate` | 5e-4 | **1e-5** | SFT 有预训练基础，高 lr 会导致灾难性遗忘 |
| `batch_size` | 32 | **16** | SFT 序列更长，减小 batch 补偿显存 |
| `accumulation_steps` | 8 | **1** | SFT 数据量小，不需要大 batch |
| `max_seq_len` | 340 | **768** | 对话数据比纯文本长 |
| `data_path` | pretrain_t2t_mini.jsonl | **sft_t2t_mini.jsonl** | 不同数据集 |
| `from_weight` | 'none' | **'pretrain'** | SFT 必须基于预训练权重 |
| `save_weight` | 'pretrain' | **'full_sft'** | 区分保存的权重名 |
| `wandb_project` | MiniMind-Pretrain | **MiniMind-Full-SFT** | wandb 面板区分 |

### 3.2 关键代码解读

**模型初始化**（line 134）：
```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
# from_weight='pretrain' → trainer_utils.py 中会加载 '../out/pretrain_{hidden_size}.pth'
```

`init_model`（`trainer_utils.py:119-131`）创建一个 `MiniMindForCausalLM`，然后 `load_state_dict(path, strict=False)`。`strict=False` 允许 MoE / 非 MoE 权重间的部分参数不匹配（比如从非 MoE 模型加载权重到 MoE 模型时，缺失的 expert 层会随机初始化）。

**前向传播与 Loss 计算**（line 34-36）：
```python
res = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss
loss = loss / args.accumulation_steps
```

- `res.loss`：语言模型交叉熵 loss。在 `model_myminimind.py:269` 中 `F.cross_entropy(logits, labels, ignore_index=-100)`，只对 labels 不为 -100 的位置计算 loss
- `res.aux_loss`：MoE router 的负载均衡辅助 loss（非 MoE 模型为 0）
- 除以 `accumulation_steps`：梯度累积时做归一化

模型 forward 中的 loss 计算（`model_myminimind.py:267-269`）：
```python
x, y = logits[..., :-1, :], labels[..., 1:]     # shift: 用 token i 预测 token i+1
loss = F.cross_entropy(x, y, ignore_index=-100)  # -100 的 label 不参与 loss
```

**学习率调度**（`trainer_utils.py:40-41`）：
```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```
Cosine 退火，最终衰减到初始 lr 的 10%。

**Checkpoint 保存**（line 60-69）：每 `save_interval` 步保存两份：
1. `full_sft_{hidden_size}.pth`：纯模型权重（推理用）
2. `full_sft_{hidden_size}_resume.pth`：模型 + 优化器 + scaler + epoch/step + wandb_id（续训用）

**梯度累积+未刷新的尾部处理**（line 75-80）：
```python
if last_step > start_step and last_step % args.accumulation_steps != 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```
处理 epoch 结束时未被 `accumulation_steps` 整除的剩余梯度，避免尾部梯度丢失。

---

## 4. 推理评测 `eval_llm.py` 关键点

默认 `--weight full_sft`，即加载 SFT 训练好的权重。

**对话格式构建**（line 73-76）—— 预训练 vs SFT 的关键区别：
```python
if 'pretrain' in args.weight:
    inputs = tokenizer.bos_token + prompt         # 预训练模型：直接拼接文本
else:
    inputs = tokenizer.apply_chat_template(conversation, ...,
        add_generation_prompt=True)  # SFT 模型：用对话模板
```

`add_generation_prompt=True` 会在末尾添加 `<|im_start|>assistant\n`，相当于对模型说"轮到你了，开始回答吧"。这是推理时调用 SFT 模型的正确方式。

**生成参数**：
- `max_new_tokens=8192`：最大生成长度
- `temperature=0.85, top_p=0.95`：采样策略
- `repetition_penalty=1.0`：无重复惩罚
- `do_sample=True`：随机采样（非贪心）

---

## 5. 训练流程速览

```
1. 分布式初始化 (init_distributed_mode)
2. 构建模型配置 (MyMiniMindConfig: hidden_size, num_layers, use_moe)
3. 从 pretrain 权重加载模型 (init_model → load_state_dict)
4. 加载 SFTDataset，构建 SkipBatchSampler + DataLoader
5. 训练循环:
   for each epoch:
     for each batch:
       - 前向: model(input_ids, labels) → loss + aux_loss
       - 反向: scaler.scale(loss).backward()
       - 梯度累积满 accumulation_steps 步后: unscale → clip_grad → scaler.step → scaler.update
       - 定期打 log (loss/lr/eta)
       - 定期保存 checkpoint (.pth + _resume.pth)
   end
6. 销毁分布式进程组 (destroy_process_group)
```
