"""Aggregate real public-data alignment runs without comparing unlike training losses."""
import csv
import json
from pathlib import Path
import statistics

from experiments.plot_metrics import read_metrics, plot_metrics, COLORS

METHODS = {'sft': 'Full SFT', 'lora': 'LoRA SFT', 'dpo': 'DPO', 'grpo': 'GRPO'}


def summarize(directory):
    evaluation = json.loads((directory / 'evaluation.json').read_text())
    metric_path = next(directory.glob('*.metrics.jsonl'))
    rows = read_metrics(metric_path)
    metadata = json.loads(metric_path.with_suffix('.meta.json').read_text())
    seconds = sum(r['step_seconds'] for r in rows)
    steady = rows[1:] or rows
    return {**evaluation, 'method': directory.name,
        'seed': int(directory.parent.name.removeprefix('seed_')),
        'updates': rows[-1]['update'], 'active_seconds': seconds,
        'tokens_per_second': sum(r['input_tokens'] for r in rows) / seconds,
        'steady_tokens_per_second': sum(r['input_tokens'] for r in steady) / sum(r['step_seconds'] for r in steady),
        'cuda_peak_allocated_mib': max(r['cuda_peak_allocated_mib'] for r in rows),
        'cuda_peak_reserved_mib': max(r['cuda_peak_reserved_mib'] for r in rows),
        'trainable_parameters': metadata['trainable_parameters'],
        'zero_std_groups': statistics.mean([r['zero_std_groups'] for r in rows]) if 'zero_std_groups' in rows[0] else None}


def build_report(output, publish=False):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    config = manifest['config']
    baseline = json.loads((output / 'baseline/evaluation.json').read_text())
    rows = [summarize(p.parent) for p in sorted(output.glob('seed_*/*/evaluation.json'))]
    if not rows:
        return
    with (output / 'results.metrics.csv').open('w') as result:
        writer = csv.DictWriter(result, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    fields = ['chosen_token_nll', 'preference_accuracy_sum', 'preference_accuracy_normalized',
              'dpo_loss_relative_to_base', 'reward_mean', 'reward_win_rate_vs_base',
              'reward_delta_to_base', 'truncated_fraction', 'active_seconds',
              'training_wall_seconds', 'tokens_per_second', 'steady_tokens_per_second',
              'cuda_peak_allocated_mib', 'cuda_peak_reserved_mib']
    for method in METHODS:
        members = [row for row in rows if row['method'] == method]
        if not members:
            continue
        item = {'n': len(members), 'trainable_parameters': members[0]['trainable_parameters']}
        for field in fields:
            values = [r[field] for r in members]
            item[field] = {'mean': statistics.mean(values), 'std': statistics.stdev(values) if len(values) > 1 else 0.}
        if method == 'grpo':
            item['zero_std_groups'] = statistics.mean(r['zero_std_groups'] for r in members)
        grouped[method] = item
    (output / 'summary.metrics.json').write_text(json.dumps({'baseline': baseline, 'methods': grouped}, indent=2))
    def show(item, key, digits=3):
        value = item[key]
        return f'{value["mean"]:.{digits}f} ± {value["std"]:.{digits}f}'
    lines = ['# 真实中文偏好数据上的四方法对比', '',
        '本报告使用国内镜像下载的公开中文偏好数据，原始偏好回复未改写。', '',
        '## 数据与协议', '',
        f'- 数据：[llamafactory/DPO-En-Zh-20k](https://hf-mirror.com/datasets/{config["dataset"]}) 的 `dpo_zh.json`。版本 `{config["dataset_revision"]}`。',
        f'- 奖励模型：[Skywork-Reward-V2-Qwen3-0.6B](https://hf-mirror.com/{config["reward_model"]})。版本 `{config["reward_revision"]}`。两者仓库元数据均标为 Apache-2.0，原数据卡与 SHA256 保存在下载清单。',
        f'- 原始 {manifest["preparation"]["raw_rows"]} 条；去重与长度过滤后 {manifest["preparation"]["eligible_rows"]} 条可用。固定划分训练 {config["train_size"]}、验证 {config["validation_size"]}、测试 {config["test_size"]} 条，按归一化完整提示分组，三个集合提示哈希不重合。',
        '- 这是原数据只有 train 时建立的本地划分，不是数据集官方 test。基础模型与奖励模型是否曾接触相关语料无法排除。',
        f'- 测试集平均长度（本模型 tokenizer）：chosen={manifest["preparation"]["mean_token_lengths"]["test"]["chosen_tokens"]:.1f}，rejected={manifest["preparation"]["mean_token_lengths"]["test"]["rejected_tokens"]:.1f} token。',
        '- Full SFT 与 LoRA 只监督原始 chosen，使用与 DPO chosen 完全相同的 token 和标签掩码；DPO 使用原始 chosen/rejected；GRPO 使用相同训练问题，按冻结奖励模型评分，不使用测试回复训练。',
        f'- 四方法共同起点：`{config["base_path"]}`，SHA256 `{manifest["base_sha256"]}`。模型 hidden={config["hidden_size"]}、layers={config["num_hidden_layers"]}。',
        f'- 训练种子 {config["seeds"]}，每个运行 {config["max_steps"]} 次更新，batch={config["batch_size"]}、累积={config["accumulation_steps"]}，完整序列上限 {config["max_seq_len"]}，提示上限 {config["max_prompt_len"]}。',
        f'- 学习率事先固定：{config["learning_rates"]}；没有按测试集选择超参数或 checkpoint。LoRA rank={config["lora_rank"]}、alpha={config["lora_alpha"]}，其余三种方法更新全参数。',
        f'- GRPO 每个问题采样 {config["group_size"]} 条，最多 {config["max_new_tokens"]} token，温度 1，无 top-k/top-p 截断；使用组内标准化优势、PPO 裁剪和固定参考 KL。',
        '- 所有策略训练与评测用 FP32；奖励模型用 BF16、SDPA、最多 1024 token。GRPO 显存包括额外的 0.6B 奖励模型。',
        f'- 设备：{manifest["gpu"]}。相同更新数和问题数不是相同 FLOPs 或墙钟预算，GRPO 有在线采样和额外评分成本。', '',
        '## 共同评测', '',
        f'教师强制指标使用全部 {config["test_size"]} 个测试偏好对；生成与奖励分数使用事先固定的前 {config["generation_eval_size"]} 个测试提示。贪心生成上限 {config["max_new_tokens"]} token，FP32 续训权重评测，避免导出精度差异。', '',
        '下列数值为训练种子间均值 ± 样本标准差，不是置信区间。', '',
        '| 方法 | 完成种子 | Chosen token NLL ↓ | 原始序列偏好准确率 ↑ | 长度归一化偏好准确率 ↑ | 相对基线 DPO loss ↓ |',
        '|---|---:|---:|---:|---:|---:|',
        f'| 未继续训练的 SFT | — | {baseline["chosen_token_nll"]:.3f} | {baseline["preference_accuracy_sum"]:.3f} | {baseline["preference_accuracy_normalized"]:.3f} | {baseline["dpo_loss_relative_to_base"]:.3f} |']
    for method, item in grouped.items():
        lines.append(f'| {METHODS[method]} | {item["n"]} | {show(item,"chosen_token_nll")} | {show(item,"preference_accuracy_sum")} | {show(item,"preference_accuracy_normalized")} | {show(item,"dpo_loss_relative_to_base")} |')
    lines += ['', '原始序列概率按回复 token 求和，会偏好短回复；长度归一化按每 token 平均，亦不能等同于人类判分。相对基线 DPO loss 使用固定基线的 chosen/rejected log-prob 差与 beta=0.1，基线为 log(2)。NLL 包含 assistant 回复及结束标记，不包含提示。', '',
        '| 方法 | 生成奖励模型分数 ↑ | 对基线代理胜率（平局算半）↑ | 生成截断率 ↓ |',
        '|---|---:|---:|---:|',
        f'| 未继续训练的 SFT | {baseline["reward_mean"]:.3f} | 0.500 | {baseline["truncated_fraction"]:.3f} |']
    for method, item in grouped.items():
        lines.append(f'| {METHODS[method]} | {show(item,"reward_mean")} | {show(item,"reward_win_rate_vs_base")} | {show(item,"truncated_fraction")} |')
    lines += ['', '**生成分数与代理胜率使用的也是 GRPO 训练奖励模型，属于同裁判代理指标，不是独立人评，也不是通用能力证明。** 截断率高时，分数主要反映受限长度的回答片段，需连同原始回复检查。', '',
        '## 资源与曲线', '',
        '| 方法 | 可训练参数 | 总活跃秒 ↓ | 训练进程总秒 ↓ | 输入 token/s ↑ | 首步后 token/s ↑ | CUDA allocated 峰值 MiB ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method, item in grouped.items():
        lines.append(f'| {METHODS[method]} | {item["trainable_parameters"]:,} | {show(item,"active_seconds",1)} | {show(item,"training_wall_seconds",1)} | {show(item,"tokens_per_second",1)} | {show(item,"steady_tokens_per_second",1)} | {show(item,"cuda_peak_allocated_mib",1)} |')
    lines += ['', '有效 token/s 仅计策略更新输入的非 padding token，耗时包含数据、采样、参考/奖励打分和更新；不重复计算额外前向的 token。GRPO 奖励模型在首轮按需加载，首步后吞吐另列。训练进程总耗时包含初始化和 checkpoint 写入，评测不计入。显存是每卡 allocator 峰值，不是全系统 nvidia-smi 占用。', '',
        '每个运行均保存 Loss、Learning Rate、有效输入 token/s 和显存四联图（PNG/SVG/PDF）。不同方法 Loss 的目标不同，不比较其绝对大小。', '',
        '## 完整性与限制', '',
        f'- 已完成 {len(rows)}/{len(config["seeds"])*4} 个运行。只汇总已经完成的运行。',
        '- 本轮使用公开真实指令偏好对，没有自行编造问题、chosen 或 rejected；公开偏好数据本身可能包含模型生成、模型排序内容，不能自动视为人类标注。',
        '- 这是 64M 模型在固定子集和固定预算上的对齐比较，不能宣称完整数据集结果或算法普遍排名；验证集仅作奖励接口检查，未用于调参。',
        '- 全部生成 metric、日志、模型、原始数据和本地结果目录均已加入 Git 忽略；代码、配置和本说明可版本管理。', '',
        '## 本地结果与复现', '',
        f'运行目录：`{output.relative_to(Path(__file__).resolve().parents[1])}`。包含 manifest.json、逐个运行的 command.json、evaluation.json、pairs.jsonl、predictions.jsonl、指标 JSONL 和曲线。', '',
        '```bash', 'uv run python experiments/download_preference.py',
        'uv run python experiments/run_preference.py --config configs/experiments/zh_preference.json --output_dir experiments/runs/zh_preference_20261029 --device cuda:0', '```', '']
    (output / 'report.md').write_text('\n'.join(lines))
    first_seed = config['seeds'][0]
    paths = [next((output / f'seed_{first_seed}' / method).glob('*.metrics.jsonl')) for method in METHODS
             if (output / f'seed_{first_seed}' / method / 'evaluation.json').exists()]
    plot_metrics(paths, output / 'figures')
    import matplotlib.pyplot as plt
    methods = list(grouped)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for axis, field, title in zip(axes, ['chosen_token_nll', 'preference_accuracy_normalized', 'reward_win_rate_vs_base'],
        ['Chosen token NLL (lower better)', 'Length-normalized preference accuracy', 'Same-RM proxy win rate vs base']):
        axis.bar(range(len(methods)), [grouped[m][field]['mean'] for m in methods],
                 yerr=[grouped[m][field]['std'] for m in methods], capsize=3, color=COLORS[:len(methods)],
                 edgecolor='black', linewidth=.5)
        base = .5 if field == 'reward_win_rate_vs_base' else baseline[field]
        axis.axhline(base, color='black', linestyle='--', label='Unchanged SFT')
        axis.set_xticks(range(len(methods)), [METHODS[m] for m in methods], rotation=15)
        axis.set_title(title, fontsize=10)
        axis.set_ylim(bottom=0)
        if 'accuracy' in field or 'win_rate' in field:
            axis.set_ylim(0, 1.05)
        axis.legend(fontsize=8)
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('Public Chinese preference data; error bars: across-seed sample SD', fontsize=11)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'comparison.{extension}', dpi=160)
    plt.close(fig)
    if publish:
        root = Path(__file__).resolve().parents[1]
        (root / 'docs/experiments/zh_preference_report.md').write_text('\n'.join(lines))
