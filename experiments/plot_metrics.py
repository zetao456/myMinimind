"""Render training curves without overlaying unlike optimization objectives."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/myminimind-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']


def read_metrics(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    # A resumed run owns the latest observation for a step.
    return [r for _, r in sorted({r.get('step', r.get('update')): r for r in rows}.items())]


def plot_metrics(paths, output_dir, x_axis='step'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    series = [(Path(path).parent.name + '/' + Path(path).name.removesuffix('.metrics.jsonl'),
               read_metrics(path)) for path in paths]
    series = [(name, rows) for name, rows in series if rows]
    if not series:
        raise ValueError('No metrics to plot')
    fig, axes = plt.subplots(len(series), 4, figsize=(15, 3 * len(series)), squeeze=False,
                             constrained_layout=True)
    panels = [('loss', 'Training objective (method-specific)'), ('learning_rate', 'Learning rate'),
              ('tokens_per_second', 'Policy input tokens / s'),
              ('cuda_peak_allocated_mib', 'Peak allocated CUDA memory (MiB)')]
    for i, (name, rows) in enumerate(series):
        for j, (key, title) in enumerate(panels):
            axis = axes[i, j]
            points = [(r.get(x_axis), r.get(key)) for r in rows
                      if r.get(x_axis) is not None and r.get(key) is not None]
            if points:
                x, y = zip(*points)
                axis.plot(x, y, color=COLORS[i % len(COLORS)], marker='.', linewidth=1.4)
                if j > 0:
                    axis.set_ylim(bottom=0)
            else:
                axis.text(.5, .5, 'CUDA unavailable' if j == 3 else 'Not recorded',
                          ha='center', va='center', transform=axis.transAxes)
            axis.set_title(title if i == 0 else '')
            axis.set_xlabel('Optimizer update / microbatch (see metadata)' if x_axis == 'step' else 'Active training time (s)')
            axis.grid(alpha=.2)
        axes[i, 0].set_ylabel(name)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output_dir / f'training_curves.{extension}', dpi=160)
    plt.close(fig)
    return output_dir / 'training_curves.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics', nargs='+', help='One or more *.metrics.jsonl files')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--x_axis', choices=['step', 'elapsed_seconds'], default='step')
    args = parser.parse_args()
    print(plot_metrics(args.metrics, args.output_dir, args.x_axis))


if __name__ == '__main__':
    main()
