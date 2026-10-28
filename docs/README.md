# 文档导航

[返回项目首页](../README.md)

项目概览、环境安装和快速开始见根目录 [README](../README.md)。技术说明统一放在本目录。

## 按任务查找

| 想做什么 | 阅读文档 | 对应源码 |
|---|---|---|
| 理解模型前向、损失、KV Cache 和生成 | [语言模型与生成逻辑](model/causal_lm.md) | [model_myminimind.py](../model/model_myminimind.py) |
| 理解全参数 SFT 与 assistant-only 标签 | [SFT 训练流程](training/full_sft.md) | [train_full_sft.py](../trainer/train_full_sft.py)、[lm_dataset.py](../dataset/lm_dataset.py) |
| 运行 LoRA、DPO、GRPO，配置奖励及续训 | [LoRA 与偏好对齐](training/alignment.md) | [train_alignment.py](../trainer/train_alignment.py)、[alignment_objectives.py](../trainer/alignment_objectives.py) |
| 查看 Loss、学习率、吞吐和显存曲线 | [训练指标与曲线](experiments/training_metrics.md) | [metrics.py](../trainer/metrics.py)、[plot_metrics.py](../experiments/plot_metrics.py) |
| 查看改进后的测试准确率及模型用法 | [中文偏好训练结果](experiments/zh_preference_result.md) | [tune_preference.py](../experiments/tune_preference.py) |
| 运行真实中文偏好数据的四方法对比 | [公开偏好实验](experiments/zh_preference.md) | [run_preference.py](../experiments/run_preference.py) |
| 理解或重建 SFT 分词缓存 | [SFT 数据预缓存](data/sft_cache.md) | [lm_dataset.py](../dataset/lm_dataset.py) |

建议阅读顺序：项目 README → 模型与生成 → SFT 流程 → 数据缓存 → LoRA 与偏好对齐。

## 目录约定

```text
docs/
├── README.md              # 统一导航
├── model/                 # 模型结构与推理机制
│   └── causal_lm.md
├── training/              # 训练流程与使用指南
│   ├── full_sft.md
│   └── alignment.md
├── data/                  # 数据处理与缓存机制
│   └── sft_cache.md
└── experiments/           # 训练指标与曲线
    ├── training_metrics.md
    ├── zh_preference.md
    └── zh_preference_result.md
```

新增技术文档放入对应主题目录，并在本页登记。根目录 README 保留项目介绍和常用入口；`CLAUDE.md` 是工具读取的开发约定，保留在根目录。文档内的本地链接使用相对路径，命令则明确执行目录。
