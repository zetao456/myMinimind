# 中文偏好训练结果

[文档首页](../README.md) · [数据与配置](zh_preference.md)

**改进 DPO 将测试偏好准确率从 55.86% 提高到 61.72%，256 对中多判对 15 对。**

| 模型 | 测试偏好准确率 | 完成种子数 |
|---|---:|---:|
| 原始模型 | 55.86% | — |
| 全参数 SFT | 52.34% | 2 |
| LoRA SFT | 52.73% | 2 |
| 初始 DPO | 56.64% | 2 |
| 初始 GRPO | 55.86% | 2 |
| 改进 DPO | **61.72%** | 1 |

## 其余方法的本轮调优结果

本轮已结束，四种方法均有指标记录和曲线。固定种子 42，SFT、LoRA 各比较两组配置和四个训练节点；GRPO 比较 32、64、96 次更新。按验证集选择 checkpoint，再测同一组 256 对测试数据：

| 方法 | 初始测试准确率 | 调优后测试准确率 | 正确对数 | 选中更新数 |
|---|---:|---:|---:|---:|
| 全参数 SFT | 52.34% | 51.17% | 131/256 | 64 |
| LoRA SFT | 52.73% | 51.56% | 132/256 | 64 |
| DPO | 56.64% | **61.72%** | **158/256** | 256 |
| GRPO | 55.86% | 55.86% | 143/256 | 32 |

本轮 SFT 和 LoRA 没有提高测试偏好准确率，GRPO 持平；DPO 保持最好结果。没有用测试集成绩改选 checkpoint，也没有替换仓库原始模型。

最终配置：SFT 学习率 `3e-5`；LoRA rank 32、alpha 64、学习率 `3e-4`；GRPO 学习率 `1e-5`、KL beta `0.01`、生成长度 256、每批采样更新两次。GRPO 后续节点验证准确率下降，因此选中 32 次更新。用户要求尽快结束后停止继续搜索，扩充数据已准备但没有训练。

本轮文件位于 `experiments/runs/zh_preference_remaining/`：

- `report.md`、`comparison.metrics.csv/json`：准确率与资源汇总。
- `comparison.png/svg/pdf`：四方法准确率对比和验证曲线。
- `selected_curves/training_curves.png/svg/pdf`：四方法选中 checkpoint 的 Loss、Learning Rate、吞吐、显存曲线。
- `figures/training_curves.png/svg/pdf` 及每个候选的 `curves/`：搜索过程完整曲线。
- `validation_history.json`、`*_best.json`、`*_selected_test.json`、`*_selected_test_pairs.jsonl`：选模和逐对评测记录。
- `sft_selected_768.pth`、`grpo_selected_768.pth`：所选 FP32 模型；`lora_best_768.pth`：所选 adapter；`*_best_resume.pth`：完整续训状态。

原始 JSONL 指标、图、日志、数据和权重均在 Git ignore 范围内。

准确率定义：同一提示下，chosen 回复的每 token 平均对数概率高于 rejected 即判对。测试集固定为 256 对真实下载的中文偏好数据。

## 有效改动

- 学习率：`5e-6` → `3e-5`。
- 训练量：一轮、128 次更新 → 两轮、256 次更新。
- 验证准确率：42.19% → 53.12%。按验证集选择两轮的 checkpoint；四轮降至 52.34%。
- 最终测试：143/256 → **158/256**。

## 文件与使用

- 可直接推理的 FP32 模型：`experiments/runs/zh_preference_tuning/dpo_improved_768.pth`。
- 选中的完整训练状态：`experiments/runs/zh_preference_tuning/best_resume.pth`。
- 调参记录：`experiments/runs/zh_preference_tuning/validation_history.json`。
- 测试结果与逐对打分：`selected_test.json`、`selected_test_pairs.jsonl`。
- Loss、学习率、吞吐与显存曲线：`experiments/runs/zh_preference_tuning/dpo_stronger/curves/`。
- 初始四方法的指标与曲线：`experiments/runs/zh_preference_20261029/`。

模型、指标及本地运行结果均由 Git ignore 排除。

```bash
uv run python eval_llm.py \
  --save_dir experiments/runs/zh_preference_tuning \
  --weight dpo_improved --max_new_tokens 256
```
