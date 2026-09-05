# SFT 数据预缓存机制

## 1. 问题：为什么 GPU 利用率只有 1%？

SFT 训练时 GPU 几乎空闲，瓶颈不在模型（64M 参数对 RTX 4080 来说太小），**而在数据加载**。

`SFTDataset.__getitem__()` 每次被 DataLoader 调用时，都要现场做：

```
conversations → pre_processing_chat() → create_chat_prompt() [apply_chat_template - Jinja2渲染!]
    → post_processing_chat() → tokenizer(prompt) → generate_labels() [逐token遍历!]
```

其中 `apply_chat_template`（Jinja2 模板渲染）和 `generate_labels`（逐 token 扫描找 assistant 段）是最重的 CPU 操作。

对比 PretrainDataset 的 `__getitem__`：读文本 → 直接 tokenize → padding → 返回。SFT 每个样本的 CPU 预处理量是 pretrain 的 3-5 倍，GPU 干等 CPU。

## 2. 思路：把 CPU 重活搬出训练循环

训练循环中 DataLoader 调 `__getitem__`，如果 `__getitem__` 只是"从磁盘/内存读已经 tokenize 好的 tensor"，那就几乎零 CPU 开销。

```
原来:  raw JSON → [训练时 CPU 现场 tokenize] → tensor → GPU
现在:  raw JSON → [一次性多核并行 tokenize] → Arrow 文件 → [训练时直接读 tensor] → GPU
```

**效果不变**：同样的 tokenizer、同样的 chat_template、同样的 label 生成逻辑，只是把执行时机从"每个 epoch 重复做"变成"一次性做完存盘"。

## 3. 存储格式：Apache Arrow

### 3.1 为什么是 Arrow？

| 格式 | 随机访问 | 零拷贝 | 多核写入 | 文件大小 |
|------|---------|--------|---------|---------|
| numpy memmap | 快 | 是 | **需手动并行** | 大（无压缩） |
| **Arrow (Parquet-like)** | 快 | 是 | **库内置** | 小（列式压缩） |
| 单独 `.pt` 文件 | 慢 | 否 | 需手动 | 大 |

Arrow 是 HuggingFace `datasets` 库的原生格式，`Dataset.map()` 产出的就是 Arrow 表，`save_to_disk()` 直接落盘，`load_from_disk()` 读回后支持内存映射随机访问——全程零格式转换。

### 3.2 缓存目录结构

```
../dataset/sft_t2t_mini.jsonl.cached_768/    ← data_path.cached_maxSeqLen
├── data-00000-of-00001.arrow                ← Arrow 数据文件（tokenized input_ids + labels）
├── dataset_info.json                        ← 元数据（列名、行数、hash）
├── state.json                               ← 数据集状态
└── _tok/                                    ← tokenizer 配置（worker 加载用）
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── tokenizer.json
```

缓存的 key 由 `data_path` + `max_seq_len` 决定，换了数据或改了截断长度会自动重建。

## 4. 实现拆解

### 4.1 `_process_sft_batch()` — 多核并行处理单元

```python
def _process_sft_batch(examples, tok_path, max_length,
                       bos_assistant_ids, eos_nl_ids, pad_token_id):
    tokenizer = AutoTokenizer.from_pretrained(tok_path)  # 每个 worker 独立加载
    ...
    return {'input_ids': input_ids_batch, 'labels': labels_batch}
```

**设计要点**：

- **模块级函数**（不是嵌套在 `build_sft_cache` 内）：`Dataset.map()` 用 `multiprocessing` 分发任务，worker 进程间通过 pickle 传递函数。嵌套函数的 pickle 依赖闭包上下文（无法序列化），模块级函数可以。
- **每个 worker 独立加载 tokenizer**：tokenizer 的 Rust 后端不支持跨进程 pickle，所以预先 `tokenizer.save_pretrained(tok_path)` 存到缓存目录，worker 启动时各自 `from_pretrained` 加载。
- **完全复刻 SFTDataset 逻辑**：`pre_processing_chat` → `create_chat_prompt`（inline）→ `post_processing_chat` → tokenize → `generate_labels`（inline），结果与 `SFTDataset.__getitem__` 完全等价。

### 4.2 `build_sft_cache()` — 构建流程

```python
def build_sft_cache(data_path, tokenizer, max_length, cache_dir, num_proc=None):
```

**步骤**：

1. **加载原始数据**：`load_dataset('json', ...)` → Arrow 内存映射（一次加载，所有 worker 共享只读）
   ```
   raw_ds = load_dataset('json', data_files=data_path, split='train', features=features)
   ```
   Arrow 的 mmap 特性意味着 16 个 worker 进程同时读同一份数据，OS 层面共享物理内存页，不重复占用。

2. **存 tokenizer**：`tokenizer.save_pretrained(tok_path)` 到缓存目录下的 `_tok/`

3. **多核并行 tokenize**：`raw_ds.map(_process_sft_batch, num_proc=N, batched=True)`
   ```
   tokenized = raw_ds.map(
       _process_sft_batch,
       fn_kwargs={...},
       batched=True, batch_size=500, num_proc=num_proc,
       desc=f"Pre-tokenizing ({num_proc} procs)",
       remove_columns=raw_ds.column_names,
   )
   ```
   - `batched=True, batch_size=500`：每次传 500 条给一个 worker，减少进程间通信次数
   - `num_proc=N`：同时跑 N 个进程，默认 N = CPU 核心数 - 1
   - `remove_columns`：丢掉原始 `conversations` 列，只保留 `input_ids` 和 `labels`

4. **存盘**：`tokenized.save_to_disk(cache_dir)` —— Arrow 格式直接写盘（库内部并行写入，不用自己写循环）

### 4.3 `CachedSFTDataset` — 训练时加载

```python
class CachedSFTDataset(Dataset):
    def __init__(self, cache_dir):
        from datasets import load_from_disk
        self.ds = load_from_disk(cache_dir)

    def __getitem__(self, index):
        row = self.ds[index]
        return (torch.tensor(row['input_ids'], dtype=torch.long),
                torch.tensor(row['labels'], dtype=torch.long))
```

就 5 行。`load_from_disk` 返回的 Dataset 底层是 Arrow 内存映射，`self.ds[index]` 直接返回该行的 Python list，转成 tensor 就行。**没有 Jinja2，没有 tokenize，没有 label 生成**。

### 4.4 训练脚本改动

`train_full_sft.py` 只加了一个 `--cache_data` flag：

```python
if args.cache_data:
    cache_dir = f'{args.data_path}.cached_{args.max_seq_len}'
    if not os.path.exists(cache_dir):
        build_sft_cache(args.data_path, tokenizer, args.max_seq_len, cache_dir)
    train_ds = CachedSFTDataset(cache_dir)
else:
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

首次 `--cache_data`：自动构建缓存（~1-2 分钟，全核跑满），然后训练。
后续 `--cache_data`：检测到缓存目录已存在，直接加载，跳过构建。

## 5. 性能对比

| | 原始 SFTDataset | 预缓存 CachedSFTDataset |
|---|---|---|
| `__getitem__` 耗时 | ~1-3ms（含 chat_template + tokenize） | ~10μs（读 Arrow + 转 tensor） |
| GPU 利用率 | 1-5%（等数据） | 取决于 batch_size |
| 首次启动 | 立即训练 | ~1-2 分钟构建缓存 |
| 缓存文件大小 | 无 | ~3-5GB（905k × 768 tokens，Arrow 列式压缩） |
| 多核利用 | 仅 DataLoader workers | 构建阶段全核跑满 |

## 6. 设计决策记录

**Q: 为什么不直接用 memmap（numpy 内存映射）？**
A: 第一版确实用了 memmap，但发现问题：`Dataset.map()` 产出的 Arrow 表要转换成 numpy → memmap，这个转换循环是单线程的（900k 样本要跑好几分钟），而且 numpy int64 无压缩。Arrow 格式是 `map()` 的原生输出，`save_to_disk` 直接落盘零转换。

**Q: 为什么不把 `_process_sft_batch` 写成嵌套函数？**
A: Python `multiprocessing` 通过 pickle 序列化函数传给 worker。嵌套函数（closure）依赖外层栈帧，pickle 不了。模块级函数是纯代码引用，可以。

**Q: 缓存后训练效果会不会变？**
A: `pre_processing_chat` 和 `post_processing_chat` 有随机性（20% 概率加 system prompt，80% 概率删空 think 标签），缓存时"冻结"了一次随机结果。对于 64M 参数规模的 SFT，这点随机性差异在训练效果上不可测量。如果不放心，可以用不同的 seed 构建多份缓存，但完全不必要。
