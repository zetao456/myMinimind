# 公开中文偏好数据上的对齐实验

[文档首页](../README.md) · [项目首页](../../README.md)

## 来源

数据和奖励模型均通过用户指定的国内镜像 `https://hf-mirror.com` 下载：

| 用途 | 公开仓库 | 固定版本 |
|---|---|---|
| 中文原始偏好对 | [llamafactory/DPO-En-Zh-20k](https://hf-mirror.com/datasets/llamafactory/DPO-En-Zh-20k)，`dpo_zh.json` | `21ff425feb7cc16a45a2e91536ff237a6fd3779a` |
| 冻结奖励模型 | [Skywork/Skywork-Reward-V2-Qwen3-0.6B](https://hf-mirror.com/Skywork/Skywork-Reward-V2-Qwen3-0.6B) | `8c14a4e9e6321deaf572544339b16b8d6bbe8886` |

两个仓库的元数据均标为 Apache-2.0；原始 README、实际下载 URL、文件长度和 SHA256 一起保存在本地 `download_manifest.json`。中文偏好文件实际有 9,999 条，数据卡注明中文部分来自 `wenbopan/Chinese-dpo-pairs`。这里保留发布者的 chosen/rejected，不自行编造问题或替换偏好回复。公开偏好对可能包含模型生成/排序内容，不自动视为真人标注。

## 受控划分与训练

使用本项目 tokenizer 检查完整回复长度；过滤空回复、相同回复、过长提示或过长完整回答，并按归一化完整提示去重。过滤规则只依赖内容与长度，不使用模型预测结果。固定随机种子划分：

- 训练 1,024 对；
- 验证 128 对，用于奖励模型标签一致性检查和后续调参、checkpoint 选择；
- 测试 256 对，生成评测使用其中预先固定的前 128 个提示。

三个集合的提示哈希不重合。这是原数据只有 train 时建立的本地划分，不是官方 test；也不能排除基础模型或奖励模型早先接触过相同来源。

四种方法均从仓库已有的 `out/full_sft_768.pth` 开始：

- Full SFT 与 LoRA 学习原始 chosen，采用 `--preference_sft`，与 DPO chosen 使用完全相同的 token/标签，不使用随机 system 增强。
- DPO 使用原始 chosen/rejected。
- GRPO 在相同训练提示上采样，采用公开冻结奖励模型的标量分数。温度为 1，不做 top-k/top-p 截断，概率与 PPO 比率中的策略保持一致。

初始配置见 [zh_preference.json](../../configs/experiments/zh_preference.json)：计划三个训练随机种子，实际完成 42、43 两个种子的四方法实验，随后转向调参。每方法每种子 128 次更新、batch 4、梯度累积 2，恰好覆盖一次训练子集；序列上限 768，提示上限 256，GRPO 每提示 4 个回复、每回复最多 128 token。策略统一使用 FP32，奖励模型使用 BF16。

不同方法相同 prompt 数/更新数不等于相同算力：GRPO 包含在线生成、参考打分和额外奖励模型。学习率事先按方法固定，不根据测试集选参，最终 checkpoint 用于评测。

本轮冻结奖励模型在 128 对验证偏好上的标签一致率为 **71.1%（91/128）**；这项检查未用于选择超参数或改写数据。

## 后续调优

DPO 使用学习率 `3e-5`，在验证集比较 64、128、256、512 次更新，选中 256 次更新的 checkpoint。其余三种方法由 [tune_remaining.py](../../experiments/tune_remaining.py) 调优，固定种子 42，保持原始算法和数据划分：

| 方法 | 配置候选 | 验证节点 |
|---|---|---|
| 全参数 SFT | 学习率 `3e-6`、`3e-5` | 32、64、128、256 次更新 |
| LoRA SFT | rank 32 / alpha 64 / LR `3e-4`；rank 16 / alpha 32 / LR `3e-5` | 同上 |
| GRPO | LR `1e-5`、KL beta `0.01`、每次采样更新两次、最长生成 256 token | 同上 |

SFT 和 LoRA 仍只优化 chosen 的交叉熵；GRPO 仍使用同一个冻结奖励模型。每个方法按验证偏好准确率选择，准确率相同时选择 chosen NLL 更低者，然后在固定测试集评分。GRPO 使用 batch 2、累积 4，其余 batch 4、累积 2；两轮预算下 GRPO 每次采样更新两次，因此 256 次更新覆盖一轮提示，其余方法覆盖两轮。这里是分别调优后的比较，不是等算力比较。

运行 `uv run python experiments/tune_remaining.py`。输出位于 `experiments/runs/zh_preference_remaining/`：`search_config.json`、`validation_history.json`、每个方法的 `*_best.json`、`*_selected_test.json` 和逐对结果，以及各候选的 `curves/` 和汇总 `figures/`。每个训练器逐次更新保存 Loss、学习率、吞吐、显存，绘图输出 PNG/SVG/PDF。结果见 [训练结果](zh_preference_result.md)。

用户要求尽快结束后，本轮在 SFT/LoRA 四个候选完成、GRPO 保存至 96 次更新后停止搜索，评估该 GRPO checkpoint，再用验证集选中的三个模型完成测试与绘图。后面的 GRPO 128/256 节点没有执行完成。

另已准备全部 **6,372 对**非验证、非测试的训练数据，但本轮没有用它训练，结果表不包含扩充数据结果。原来的 128 对验证和 256 对测试逐条保留，文件 SHA256 不变。可选命令 `uv run python experiments/tune_remaining.py --expanded` 会另存到 `experiments/runs/zh_preference_expanded/`；SFT 学习率 `2e-5`，LoRA rank 32、alpha 64、学习率 `1e-4`，单轮预算，在 128、256、512、768 次更新处验证。

最终运行 `uv run python experiments/report_tuning.py` 保存四方法的准确率对比、验证曲线、选中 checkpoint 之前的训练曲线和资源汇总。搜索全程曲线仍保留在每个候选的 `curves/` 中。

## 共同评测与限制

统一记录：

- chosen 回复的 token 平均 NLL 和 perplexity；
- 原始序列概率偏好准确率（受回复长度影响）；
- 每 token 平均概率的偏好准确率；
- 固定基础模型参考下的测试集 DPO loss（beta=0.1）；
- 相同测试提示下的贪心生成回复、冻结奖励模型分数、对基础模型的代理胜率与截断率。

生成评测使用的奖励模型也是 GRPO 的训练奖励，**不能把该代理胜率称为独立人类偏好胜率**。需要和原始偏好标签、NLL、生成长度、截断率及原始回复一起解释。统计汇总为跨训练种子的均值和样本标准差，不是置信区间，不据此给方法作普遍排名。

CUDA allocated/reserved 峰值来自实际运行；GRPO 包含其额外奖励模型。吞吐分子仅计策略更新的非 padding 输入位置，分母包括采样及评分；不等同于各算法等 FLOPs 的硬件吞吐。奖励模型首次按需装载可能使 GRPO 首步较慢，报告另列去掉首步的吞吐。

## 运行

从仓库根目录执行，需可访问 CUDA 的环境：

```bash
uv run python experiments/download_preference.py
uv run python experiments/run_preference.py \
  --config configs/experiments/zh_preference.json \
  --output_dir experiments/runs/zh_preference_20261029 \
  --device cuda:0
```

下载器使用固定 revision，并支持断点下载；已存在文件复用并记录 SHA256。运行器保存来源清单、基础权重/tokenizer/源码哈希和源码快照。中断后使用同一命令加 `--resume`；已完成评测的运行跳过，存在训练断点的运行通过训练器恢复。配置、数据、源码或设备来源变化时拒绝混用结果。

本地运行目录包含每方法/种子的实际命令、日志、模型与续训状态、逐对打分、原始生成回复、JSONL 指标和四项曲线（PNG/SVG/PDF）。最后汇总为 `report.md`、`results.metrics.csv`、`summary.metrics.json`，完成后同步一份文字结果到 `docs/experiments/zh_preference_report.md`。

## Git 忽略

以下自动生成内容保存在本机，不进入 Git：

```text
*.metrics.jsonl
*.metrics.meta.json
*.metrics.csv
*.metrics.json
experiments/runs/
dataset/downloads/
dataset/prepared/
models/
*.pth
```

指标与曲线不会因 Git ignore 而停止保存；只是不会被提交。代码、配置和实验说明可版本管理。指标字段口径见 [训练指标与曲线](training_metrics.md)。
