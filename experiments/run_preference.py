"""Run source-pinned public Chinese preference alignment experiments on local CUDA."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from dataset.alignment_dataset import JsonlDataset
from experiments.download_preference import sha256
from experiments.preference_data import prepare
from experiments.preference_evaluate import load_policy, evaluate_likelihood, evaluate_generation, save_evaluation
from experiments.preference_reward import RewardScorer
from experiments.preference_report import build_report
from experiments.plot_metrics import plot_metrics


def run(config, output, device, resume=False):
    output = Path(output).resolve()
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('This configured experiment requires accessible CUDA')
    if (output / 'manifest.json').exists() and not resume:
        raise ValueError('Use a new output directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    downloads = json.loads((ROOT / 'dataset/downloads/dpo_en_zh_20k/download_manifest.json').read_text())
    if (downloads['dataset_revision'] != config['dataset_revision'] or
            downloads['reward_revision'] != config['reward_revision']):
        raise ValueError('Downloaded revisions differ from the experiment config')
    prepared = ROOT / 'dataset/prepared/zh_preference'
    metadata = prepare(prepared, config['train_size'], config['validation_size'], config['test_size'],
                       config['split_seed'], config['max_seq_len'], config['max_prompt_len'])
    base_path = ROOT / config['base_path']
    source = {str(p.relative_to(ROOT)): sha256(p) for folder in ('model', 'dataset', 'trainer', 'experiments')
              for p in sorted((ROOT / folder).glob('*.py'))}
    manifest = {'config': config, 'downloads': downloads, 'preparation': metadata,
        'base_sha256': sha256(base_path), 'source_sha256': source,
        'tokenizer_sha256': {name: sha256(ROOT / 'model' / name) for name in ('tokenizer.json', 'tokenizer_config.json')},
        'gpu': torch.cuda.get_device_name(device), 'device': device, 'torch': torch.__version__}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ('config', 'preparation', 'base_sha256', 'source_sha256', 'tokenizer_sha256', 'device', 'gpu'):
            if previous[key] != manifest[key]:
                raise ValueError(f'Changed experiment provenance: {key}')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for name in source:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    scorer = RewardScorer(device=device)
    calibration_path = output / 'reward_calibration.json'
    if not calibration_path.exists():
        validation = list(JsonlDataset(prepared / 'validation.jsonl'))
        chosen = scorer.score([r['prompt'] for r in validation], [r['chosen'] for r in validation])
        rejected = scorer.score([r['prompt'] for r in validation], [r['rejected'] for r in validation])
        calibration = {'pairs': len(validation), 'agreement_with_dataset':
                       sum(c > r for c, r in zip(chosen, rejected)) / len(validation),
                       'chosen_mean': sum(chosen) / len(chosen), 'rejected_mean': sum(rejected) / len(rejected)}
        calibration_path.write_text(json.dumps(calibration, indent=2))
        print('Reward-model validation calibration:', calibration, flush=True)
    baseline_dir = output / 'baseline'
    if not (baseline_dir / 'evaluation.json').exists():
        print('Evaluating unchanged SFT baseline on held-out public preference pairs', flush=True)
        model = load_policy(config, base_path, device)
        metrics, pairs = evaluate_likelihood(model, tokenizer, prepared / 'test.jsonl', config, device)
        generated, predictions = evaluate_generation(model, tokenizer, prepared / 'test.jsonl', config, device, scorer)
        save_evaluation(baseline_dir, {**metrics, **generated}, pairs, predictions)
        scorer.model.to('cpu')
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        scorer.model.to('cpu')
        torch.cuda.empty_cache()
    baseline_pairs = [json.loads(line) for line in (baseline_dir / 'pairs.jsonl').read_text().splitlines()]
    baseline_predictions = [json.loads(line) for line in (baseline_dir / 'predictions.jsonl').read_text().splitlines()]
    for seed in config['seeds']:
        for method in ('sft', 'lora', 'dpo', 'grpo'):
            directory = output / f'seed_{seed}' / method
            if (directory / 'evaluation.json').exists():
                continue
            directory.mkdir(parents=True, exist_ok=True)
            name = f'{method}_{config["hidden_size"]}'
            checkpoint_path = directory / f'{name}_resume.pth'
            command = [sys.executable, str(ROOT / f'trainer/train_{method}.py'),
                '--data_path', str(prepared / ('grpo.jsonl' if method == 'grpo' else 'train.jsonl')),
                '--base_dir', str(base_path.parent), '--from_weight', 'full_sft',
                '--save_dir', str(directory), '--checkpoint_dir', str(directory), '--save_weight', method,
                '--device', device, '--dtype', config['dtype'], '--seed', str(seed),
                '--learning_rate', str(config['learning_rates'][method]), '--log_interval', '8', '--save_interval', '64']
            for key in ('hidden_size', 'num_hidden_layers', 'epochs', 'batch_size', 'accumulation_steps',
                        'max_steps', 'max_seq_len', 'max_prompt_len'):
                command += ['--' + key, str(config[key])]
            if method in ('sft', 'lora'):
                command += ['--preference_sft']
            if method == 'lora':
                command += ['--lora_rank', str(config['lora_rank']), '--lora_alpha', str(config['lora_alpha'])]
            if method == 'grpo':
                command += ['--max_new_tokens', str(config['max_new_tokens']),
                            '--group_size', str(config['group_size']), '--reward_fn', config['reward_fn']]
            if resume and checkpoint_path.exists():
                command += ['--from_resume', '1']
            (directory / 'command.json').write_text(json.dumps(command, indent=2))
            (output / 'progress.json').write_text(json.dumps({'seed': seed, 'method': method, 'phase': 'training'}))
            print(f'Training {method}, seed {seed}, {config["max_steps"]} updates on public Chinese preferences', flush=True)
            started = time.perf_counter()
            with (directory / 'train.log').open('a' if resume else 'w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                    env={**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'REWARD_DEVICE': device})
            training_wall_seconds = time.perf_counter() - started
            (output / 'progress.json').write_text(json.dumps({'seed': seed, 'method': method, 'phase': 'evaluation'}))
            model = load_policy(config, base_path, device, checkpoint_path,
                                directory / f'{name}.pth' if method == 'lora' else None)
            metrics, pairs = evaluate_likelihood(model, tokenizer, prepared / 'test.jsonl', config, device, baseline_pairs)
            scorer.model.to(device)
            generated, predictions = evaluate_generation(model, tokenizer, prepared / 'test.jsonl', config, device,
                                                         scorer, baseline_predictions)
            save_evaluation(directory, {**metrics, **generated, 'training_wall_seconds': training_wall_seconds}, pairs, predictions)
            plot_metrics([directory / f'{name}.metrics.jsonl'], directory / 'curves')
            print(f'Completed {method}, seed {seed}: chosen NLL={metrics["chosen_token_nll"]:.4f}, normalized preference accuracy={metrics["preference_accuracy_normalized"]:.3f}, RM proxy win rate={generated["reward_win_rate_vs_base"]:.3f}', flush=True)
            scorer.model.to('cpu')
            del model
            gc.collect()
            torch.cuda.empty_cache()
            build_report(output)
    build_report(output, publish=True)
    (output / 'progress.json').write_text(json.dumps({'phase': 'complete'}))
    assert sha256(base_path) == manifest['base_sha256']
    print('All configured public-data runs complete. Report:', output / 'report.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(json.loads(Path(args.config).read_text()), args.output_dir, args.device, args.resume)


if __name__ == '__main__':
    main()
