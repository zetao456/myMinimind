# 训练指标与曲线

[文档首页](../README.md) · [项目首页](../../README.md)

## 指标记录

所有训练入口均会自动写入本地 JSONL，无需启用 SwanLab 或其他在线服务：

- 原有 `trainer/train_pretrain.py`、`trainer/train_full_sft.py`：记录每个 microbatch；可用 `--metrics_path` 指定文件。
- `trainer/train_sft.py`、`train_lora.py`、`train_dpo.py`、`train_grpo.py`：使用公共训练循环，记录每次成功的优化器更新。`train_sft.py` 是使用公共训练循环的全参数 SFT 入口，默认从现有 full_sft 权重继续适配。

默认文件名为 `<save_dir>/<save_weight>_<hidden_size>[_moe].metrics.jsonl`，旁边的 `*.metrics.meta.json` 记录配置和指标口径。不同运行应指定各自的 save_dir，避免覆盖。

| 字段 | 含义 |
|---|---|
| `step` | microbatch 或成功更新序号，单位见元数据 `step_unit` |
| `loss` | 当前方法的训练目标；公共入口按累积窗口的实际样本数加权 |
| `learning_rate` | 本步使用的学习率 |
| `step_seconds` | 本步活跃耗时，CUDA 测量前后同步 |
| `elapsed_seconds` | 累计活跃耗时，续训时承接已有记录 |
| `samples`、`samples_per_second` | 当前更新使用的样本数及每秒样本数；DPO 按偏好对计，GRPO 按问题计 |
| `input_tokens`、`tokens_per_second` | 当前策略输入的非 padding token 数及吞吐；DPO 包含 chosen/rejected 两条序列，GRPO 包含所有组内回复 |
| `supervised_tokens`、`supervised_tokens_per_second` | SFT 的有效 next-token 标签、DPO 的回复 token、GRPO 的有效生成 token |
| `generated_tokens`、`generated_tokens_per_second` | GRPO 本轮新采样 token 数及按总活跃耗时计算的吞吐；其他阶段为 0 |
| `cuda_peak_allocated_mib` | PyTorch CUDA allocator 分配峰值，MiB |
| `cuda_peak_reserved_mib` | PyTorch CUDA allocator 保留峰值，MiB |

DDP 中计数跨 rank 求和，耗时和显存跨 rank 取最大值。CPU 上显存字段为 `null`，绘图显示不可用，不填成 0。显存指标是训练进程的 allocator 统计，不包含其他应用，也不等于 nvidia-smi 的总显存。

## 吞吐口径

公共训练循环的计时包含取数据、GRPO rollout、参考打分、前向、反向和优化器更新；不包含模型初始化、checkpoint 保存和评测。GRPO 对同一 rollout 更新多次时，准备时间与新生成 token 数均摊到各次更新，显存保留整个窗口的峰值。

分子只数当前策略更新使用的输入位置，不把生成期间和参考模型额外计算的 token 重复计入。因此跨算法的 token/s 用于描述各自管线效率，不能当作等 FLOPs 的性能排名。SFT 的 padding 计算仍会发生，但不记入有效 token 吞吐。

原有预训练/Full SFT 入口以 microbatch 计时；若恰逢梯度累积边界，该步包含优化器更新，其他步没有。比较优化器更新粒度时使用公共入口。跨 epoch 不足一个累积窗口的尾部处理仍按原训练脚本执行，尾部额外优化器耗时未单列到 microbatch 指标。

## 绘制四项曲线

环境需安装项目的 matplotlib 开发依赖：

```bash
uv sync --extra gpu --extra dev
```

从仓库根目录运行：

```bash
uv run python experiments/plot_metrics.py \
  out/lora_sft_768.metrics.jsonl \
  --output_dir out/lora_curves
```

也可同时绘制多个运行，并按累计活跃秒显示横轴：

```bash
uv run python experiments/plot_metrics.py \
  path/to/sft.metrics.jsonl path/to/dpo.metrics.jsonl \
  --output_dir out/comparison_curves --x_axis elapsed_seconds
```

输出 `training_curves.png`、`.svg`、`.pdf`。每个运行单独一行，依次展示 Loss、Learning Rate、有效输入 token/s、CUDA 峰值分配显存。方法之间训练目标不同，Loss 不叠加在同一坐标中排名；原始记录保留全部点，不通过平滑隐藏波动。

续训按断点的 step 删除日志中超前记录，再追加新记录，避免出现重复和回退的曲线。旧训练权重未记录过吞吐、耗时和显存时，不能从 checkpoint 事后还原这些曲线。

原始指标及本地实验运行目录受 Git 忽略规则保护，不影响本地保存与绘图。公开数据对比见 [中文偏好实验](zh_preference.md)。

实现：[metrics.py](../../trainer/metrics.py)、[公共训练入口](../../trainer/train_alignment.py)、[绘图工具](../../experiments/plot_metrics.py)。
