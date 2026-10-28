# SFT 数据预缓存机制

[文档首页](../README.md) · [项目首页](../../README.md)

本文对应当前 [lm_dataset.py](../../dataset/lm_dataset.py) 中的 `build_sft_cache`、`_tokenize_chunk_memmap` 和 `CachedSFTDataset`。缓存使用 **NumPy int16 memmap**；旧笔记中的 Arrow 缓存方案已不对应当前实现。

## 为什么预缓存

在线 `SFTDataset.__getitem__()` 每次取样都会执行：

```text
conversations
  → pre_processing_chat（随机添加 system）
  → create_chat_prompt（Chat Template 渲染）
  → post_processing_chat（随机移除空 think 块）
  → tokenize、截断、右侧 padding
  → generate_labels（定位 assistant 回复）
```

这些操作会在每个 epoch 重复执行。预缓存把模板渲染、分词和标签生成提前到构建阶段；训练期间只读取已编码的数组并转换为 tensor。是否能提高吞吐，以及提升多少，需要在具体硬件和 batch size 下测量。

## 文件格式与目录

缓存目录名由训练脚本按 `data_path + '.cached_' + max_seq_len` 生成，例如：

```text
dataset/sft_t2t_mini.jsonl.cached_768/
├── input_ids.dat        # int16，形状 [样本数, max_seq_len]
├── labels.dat           # int16，同样形状；忽略位置为 -100
├── shape.npy            # 两个数组的形状
└── _tok/                # 构建时保存的 tokenizer，供 worker 加载
```

当前词表为 6400，token ID 和标签忽略值 `-100` 都可由 int16 表示。若更换更大词表，需要先检查 token ID 是否超出 int16 的范围。

两个数组的数据大小为：

```text
样本数 × 序列长度 × 2 字节 × 2 个数组
```

例如 905718 条、长度 768 时，约为 2.78 GB（2.59 GiB），不计 tokenizer 和形状元数据。memmap 没有在此基础上额外压缩。

## 构建过程

1. `load_dataset('json', ...)` 读取原始对话，提取到模块级 `_all_conversations` 列表。
2. 将 tokenizer 保存到 `_tok/`，准备 assistant 起始标记和回复结束标记的 token IDs。
3. 预先创建两个 int16 memmap 文件及 `shape.npy`。
4. 将样本索引划分为互不重叠的区间，用 `multiprocessing.Pool` 分发。
5. 每个 worker 独立加载 tokenizer，按区间处理对话，直接写入相应 memmap 行，完成后 flush。

worker 默认数量为 `max(1, os.cpu_count() - 1)`。当前实现依赖 Linux 的 fork 方式，让 worker 通过写时复制共享父进程的对话列表；没有提供 spawn 平台的等价数据传递流程。

标签规则与在线 SFT 相同：以 assistant 头部定位回复，将回复和结束标记写入 labels，其余位置保持 `-100`。预处理中的随机增强会在缓存建立时固定一次，在线处理则可在每次取样时重新随机。

## 训练时读取

`CachedSFTDataset` 根据 `shape.npy` 以只读方式映射两个文件。取样时复制当前行，转换为 int64，再返回 PyTorch tensor。它避免了运行时重复分词，但并非从磁盘到 tensor 全程零拷贝。

从仓库根目录使用：

```bash
# 原有 Full SFT 默认路径以 trainer/ 为工作目录
(cd trainer && uv run python train_full_sft.py --cache_data)

# LoRA 入口默认模型路径按仓库根目录定位
uv run python trainer/train_lora.py \
  --data_path dataset/sft_t2t_mini.jsonl --cache_data
```

完整训练参数见 [SFT 流程](../training/full_sft.md) 和 [LoRA 与偏好对齐](../training/alignment.md)。

## 缓存复用与重建

训练入口根据缓存目录是否存在决定构建或复用。**不会自动核对数据内容、tokenizer 或缓存完整性**。修改原始文件但沿用同一路径，也不会触发重建。

更换数据、tokenizer 或标签逻辑后，应先将旧缓存目录移走备份，再重新运行 `--cache_data`。更改序列长度会使用另一个目录名。构建中断可能留下不完整目录，重新运行前也应先处理该目录。

`train_lora.py` 的公共训练入口只让主进程建立缓存，其他 DDP 进程等待；原有 `train_full_sft.py` 的自动构建没有这层协调，使用多卡 Full SFT 前应先准备好缓存。
