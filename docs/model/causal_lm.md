# MiniMindForCausalLM 逐行详解

[文档首页](../README.md) · [项目首页](../../README.md)

对应源码：[model_myminimind.py](../../model/model_myminimind.py)。训练中的标签构造见 [SFT 流程](../training/full_sft.md)，训练和推理入口见 [项目 README](../../README.md)。

## 类声明

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
```

- `PreTrainedModel`：HuggingFace 基类，提供 `save_pretrained`、`from_pretrained`、权重初始化等
- `GenerationMixin`：提供 HF 标准的 `generate` 接口（这里重写了）

---

## 类变量

```python
config_class = MyMiniMindConfig
```

告诉 HuggingFace 本模型对应的配置类。

```python
_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

标记 `lm_head` 和 `embed_tokens` 共享权重。HF 保存/加载时会识别这层关系，避免重复存储。

---

## `__init__` — 初始化

```python
def __init__(self, config: MyMiniMindConfig = None):
    self.config = config or MyMiniMindConfig()
```

没传 config 就用默认配置。`self.config` 必须先于 `super().__init__` 赋值，因为父类构造器马上要用。

```python
    super().__init__(self.config)
```

调用 `PreTrainedModel.__init__`，完成 HF 模型初始化。

```python
    self.model = MiniMindModel(self.config)
```

Transformer 本体：embedding + N 层 MiniMindBlock + 最终 RMSNorm。不包含输出层。

```python
    self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
```

语言模型头：`[hidden_size] → [vocab_size]`，得到每个 token 的 logits。

```python
    if self.config.tie_word_embeddings:
        self.model.embed_tokens.weight = self.lm_head.weight
```

权重绑定：输入 embedding 和输出 lm_head 共享权重。减少参数量，训练更稳定。

```python
    self.post_init()
```

`PreTrainedModel` 的方法，初始化未覆盖的权重，做 HF 内部设备映射。

---

## `forward` — 前向传播

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None,
            use_cache=False, logits_to_keep=0, labels=None, **kwargs):
```

参数说明：
- `logits_to_keep=0`：0 表示保留全部位置；正数 N 则只保留最后 N 个（节省显存）
- `labels`：训练时传入，用于计算交叉熵 loss

```python
    hidden_states, past_key_values, aux_loss = self.model(
        input_ids, attention_mask, past_key_values, use_cache, **kwargs
    )
```

交给 `MiniMindModel` 做完整前向传播。返回：
- `hidden_states`：`[batch, seq_len, hidden_size]`
- `past_key_values`：KV 缓存（`use_cache=False` 时全为 None）
- `aux_loss`：MoE router 负载均衡 loss（非 MoE 时为 0）

```python
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
```

只对最后 `logits_to_keep` 个位置做投影。`logits_to_keep=0` → `slice(0, None)` → 全部保留。

```python
    loss = None
    if labels is not None:
        x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
        loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

训练时计算交叉熵 loss：
- `x`：预测 logits，去掉最后一个 token（它没有 label）
- `y`：labels，去掉第一个 token（teacher forcing，用第 i 个 token 预测第 i+1 个）
- `ignore_index=-100`：忽略 padding 位置的 loss

```python
    return MoeCausalLMOutputWithPast(
        loss=loss, aux_loss=aux_loss, logits=logits,
        past_key_values=past_key_values, hidden_states=hidden_states
    )
```

返回 HF 标准输出格式。

---

## `generate` — 自回归生成

### 装饰器与签名

```python
@torch.inference_mode()
def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192,
             temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2,
             streamer=None, use_cache=True, num_return_sequences=1,
             do_sample=True, repetition_penalty=1.0, **kwargs):
```

`@torch.inference_mode()`：禁用梯度计算，推理时节省显存。

关键参数：
- `max_new_tokens=8192`：最多生成 token 数（不是序列总长度）
- `temperature=0.85`：< 1 则分布更尖锐（更确定），> 1 则更平滑（更多样）
- `top_p=0.85`：nucleus sampling 阈值
- `top_k=50`：只从概率最高的 50 个 token 中采样
- `repetition_penalty=1.0`：1.0 表示不惩罚，> 1.0 惩罚重复

### 初始化

```python
    input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
```

兼容 `input_ids` 和 `inputs` 两种传参。`repeat` 支持一次生成多条序列。

```python
    attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
    past_key_values = kwargs.pop("past_key_values", None)
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
```

`finished`：标记每条序列是否已生成 EOS。多序列生成时各自独立终止。

```python
    if streamer: streamer.put(input_ids.cpu())
```

先把 prompt 的 token ids 放入 streamer（`TextStreamer` 内部跳过 prompt，只打印生成部分）。

### 自回归循环

```python
    for _ in range(max_new_tokens):
```

每轮生成一个 token。

#### 增量推理

```python
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        outputs = self.forward(
            input_ids[:, past_len:], attention_mask,
            past_key_values, use_cache=use_cache, **kwargs
        )
```

- 首次：`past_len=0`，传入完整 prompt
- 后续：只传入**新生成的 token**，KV 缓存负责保留历史信息

#### 更新 attention_mask

```python
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1
        ) if attention_mask is not None else None
```

每生成一个 token，mask 延长一位（全 1，生成的 token 都有效）。

#### Temperature 缩放

```python
        logits = outputs.logits[:, -1, :] / temperature
```

取最后一个位置的 logits，除以 temperature。

#### 重复惩罚

```python
        if repetition_penalty != 1.0:
            for i in range(input_ids.shape[0]):
                seen = torch.unique(input_ids[i])
                score = logits[i, seen]
                logits[i, seen] = torch.where(
                    score > 0, score / repetition_penalty, score * repetition_penalty
                )
```

对已出现过的 token：
- 正 logit → 除以 penalty（降低概率）
- 负 logit → 乘以 penalty（进一步压低）

#### Top-K 过滤

```python
        if top_k > 0:
            logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
```

只保留概率最高的 K 个 token，其余设为 `-inf`（softmax 后概率为 0）。

#### Top-P（Nucleus）采样

```python
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
            mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
            logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
```

步骤：
1. 按 logits 从大到小排序
2. 计算 softmax 概率，累积求和
3. 标记累积概率超过 `top_p` 的位置
4. 掩码右移一位 → **至少保留概率最高的 1 个 token**
5. 被掩码的设为 `-inf`

#### 采样

```python
        next_token = (
            torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
        )
```

- `do_sample=True`：按 softmax 概率随机采样
- `do_sample=False`：贪心取最大概率 token

`-inf` 位置的 softmax 概率为 0，不会被采样。

#### 已结束序列的处理

```python
        if eos_token_id is not None:
            next_token = torch.where(
                finished.unsqueeze(-1),
                next_token.new_full((next_token.shape[0], 1), eos_token_id),
                next_token
            )
```

已结束的序列强制填充 `eos_token_id`，避免继续产生有效 token。

#### 拼接 & 更新缓存

```python
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        past_key_values = outputs.past_key_values if use_cache else None
```

#### 实时输出

```python
        if streamer: streamer.put(next_token.cpu())
```

推送给 `TextStreamer`，实现逐 token 打印。

#### 终止判断

```python
        if eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if finished.all(): break
```

所有序列都生成了 EOS → 提前退出。

### 收尾

```python
    if streamer: streamer.end()
    if kwargs.get("return_kv"):
        return {'generated_ids': input_ids, 'past_kv': past_key_values}
    return input_ids
```

- 通知 streamer 结束
- 返回完整序列（prompt + 生成部分）
