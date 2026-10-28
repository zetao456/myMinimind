"""Render selected-checkpoint comparisons for the four public-data experiments."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.plot_metrics import plot_metrics, read_metrics
import matplotlib.pyplot as plt


def load(path):
    return json.loads(path.read_text())


def main():
    output = ROOT / 'experiments/runs/zh_preference_remaining'
    original = ROOT / 'experiments/runs/zh_preference_20261029'
    dpo = ROOT / 'experiments/runs/zh_preference_tuning'
    expanded = ROOT / 'experiments/runs/zh_preference_expanded'
    baseline = load(original / 'baseline/evaluation.json')
    rows, paths = [], []
    for stage in ['sft', 'lora', 'dpo', 'grpo']:
        directory = dpo if stage == 'dpo' else output
        if stage in ('sft', 'lora') and (expanded / f'{stage}_best.json').exists():
            candidates = [directory, expanded]
            def selection_key(path):
                scores = load(path / f'{stage}_best.json')['validation']
                return scores['preference_accuracy_normalized'], -scores['chosen_token_nll']
            directory = max(candidates, key=selection_key)
        selection = load(directory / ('best.json' if stage == 'dpo' else f'{stage}_best.json'))
        scores = load(directory / ('selected_test.json' if stage == 'dpo' else f'{stage}_selected_test.json'))
        trial = selection['trial']
        metrics = read_metrics(directory / trial / f'{trial}_768.metrics.jsonl')
        selected = [r for r in metrics if r['update'] <= selection['updates']]
        selected_path = output / 'selected_metrics' / stage / f'{stage}.metrics.jsonl'
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(''.join(json.dumps(r) + '\n' for r in selected))
        paths.append(selected_path)
        initial = load(original / 'seed_42' / stage / 'evaluation.json')
        seconds = sum(r['step_seconds'] for r in selected)
        rows.append({'method': stage, 'trial': trial, 'run_directory': str(directory.relative_to(ROOT)),
            'training_pairs': 6372 if directory == expanded else 1024, 'updates': selection['updates'],
            'initial_test_accuracy': initial['preference_accuracy_normalized'],
            'validation_accuracy': selection['validation']['preference_accuracy_normalized'],
            'test_accuracy': scores['preference_accuracy_normalized'],
            'correct_test_pairs': round(scores['preference_accuracy_normalized'] * 256),
            'chosen_token_nll': scores['chosen_token_nll'],
            'active_training_seconds': seconds,
            'input_tokens_per_second': sum(r['input_tokens'] for r in selected) / seconds,
            'cuda_peak_allocated_mib': max(r['cuda_peak_allocated_mib'] for r in selected)})
    with (output / 'comparison.metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / 'comparison.metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    plot_metrics(paths, output / 'selected_curves')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = list(range(len(rows)))
    for offset, field, label, color in [(-.19, 'initial_test_accuracy', 'Initial (seed 42)', '#999999'),
                                        (.19, 'test_accuracy', 'Validation-selected (seed 42)', '#0072B2')]:
        bars = axes[0].bar([i + offset for i in x], [r[field] * 100 for r in rows], width=.38, label=label, color=color)
        axes[0].bar_label(bars, fmt='%.2f', fontsize=8)
    axes[0].axhline(baseline['preference_accuracy_normalized'] * 100, linestyle='--', color='black', label='Base 55.86%')
    axes[0].set_xticks(x, ['Full SFT', 'LoRA SFT', 'DPO', 'GRPO'])
    axes[0].set(ylabel='Test preference accuracy (%)', ylim=(0, 76))
    axes[0].legend(fontsize=7)
    history = load(output / 'validation_history.json') + load(dpo / 'validation_history.json')
    if (expanded / 'validation_history.json').exists():
        history += load(expanded / 'validation_history.json')
    for trial in dict.fromkeys(r['trial'] for r in history):
        points = [r for r in history if r['trial'] == trial]
        axes[1].plot([r['updates'] for r in points], [100 * r['validation']['preference_accuracy_normalized'] for r in points], marker='o', label=trial)
    axes[1].axhline(42.1875, linestyle='--', color='black')
    axes[1].set(xlabel='Optimizer updates', ylabel='Validation preference accuracy (%)')
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.grid(axis='y', alpha=.2)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(output / f'comparison.{ext}', dpi=160)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
