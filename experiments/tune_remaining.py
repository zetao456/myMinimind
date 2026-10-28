"""Tune native SFT, LoRA and GRPO on validation pairs; export held-out results."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from experiments.preference_evaluate import load_policy, evaluate_likelihood
from experiments.plot_metrics import plot_metrics

OUTPUT = ROOT / 'experiments/runs/zh_preference_remaining'
DATA = ROOT / 'dataset/prepared/zh_preference'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expanded', action='store_true', help='Use all 6,372 non-held-out pairs for SFT/LoRA')
    args = parser.parse_args()
    global OUTPUT, DATA
    if args.expanded:
        OUTPUT = ROOT / 'experiments/runs/zh_preference_expanded'
        DATA = ROOT / 'dataset/prepared/zh_preference_expanded'
        if not (DATA / 'preparation.json').exists():
            from experiments.preference_data import prepare
            prepare(DATA, train_size=6372, fixed_holdout=ROOT / 'dataset/prepared/zh_preference')
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'configs/experiments/zh_preference.json').read_text())
    manifest = {}
    for relative in ['experiments/tune_remaining.py', 'experiments/preference_data.py',
                     'experiments/preference_evaluate.py', 'experiments/preference_reward.py',
                     'trainer/train_alignment.py', 'trainer/alignment_objectives.py',
                     'trainer/rollout.py', 'trainer/metrics.py', 'dataset/alignment_dataset.py',
                     'model/model_myminimind.py', 'model/model_lora.py',
                     'configs/experiments/zh_preference.json']:
        source = ROOT / relative
        target = OUTPUT / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    for path in [ROOT / config['base_path'], *[DATA / f'{name}.jsonl' for name in ['train', 'validation', 'test', 'grpo']]]:
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(OUTPUT / 'source_manifest.json', manifest)
    write(OUTPUT / 'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                      'device': torch.cuda.get_device_name(0), 'python': sys.version})
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    model = load_policy(config, ROOT / config['base_path'], 'cuda:0')
    baseline, val_pairs = evaluate_likelihood(model, tokenizer, DATA / 'validation.jsonl', config, 'cuda:0')
    del model
    gc.collect(); torch.cuda.empty_cache()
    write(OUTPUT / 'baseline_validation.json', baseline)
    trials = [
        ('sft', 'sft_conservative', 3e-6, []),
        ('sft', 'sft_stronger', 3e-5, []),
        ('lora', 'lora_rank32', 3e-4, ['--lora_rank', '32', '--lora_alpha', '64']),
        ('lora', 'lora_conservative', 3e-5, ['--lora_rank', '16', '--lora_alpha', '32']),
        ('grpo', 'grpo_stronger', 1e-5, ['--beta', '.01', '--group_size', '4',
            '--max_new_tokens', '256', '--num_iterations', '2',
            '--reward_fn', 'experiments.preference_reward:reward']),
    ]
    epochs, checkpoints = 2, [32, 64, 128, 256]
    if args.expanded:
        trials = [('sft', 'sft_expanded', 2e-5, []),
                  ('lora', 'lora_expanded', 1e-4, ['--lora_rank', '32', '--lora_alpha', '64'])]
        epochs, checkpoints = 1, [128, 256, 512, 768]
    write(OUTPUT / 'search_config.json', {'base_config': config, 'trials': trials,
        'selection': 'maximum validation preference_accuracy_normalized; ties use chosen_token_nll',
        'seed': 42, 'epochs': epochs, 'validation_checkpoints': checkpoints,
        'data_directory': str(DATA.relative_to(ROOT))})
    history_path = OUTPUT / 'validation_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    best = {stage: json.loads((OUTPUT / f'{stage}_best.json').read_text())
            for stage in ('sft', 'lora', 'grpo') if (OUTPUT / f'{stage}_best.json').exists()}
    metric_paths = []
    for stage, name, lr, extra in trials:
        directory = OUTPUT / name
        directory.mkdir(exist_ok=True)
        checkpoint = directory / f'{name}_768_resume.pth'
        adapter = directory / f'{name}_768.pth' if stage == 'lora' else None
        completed = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)['update'] if checkpoint.exists() else 0
        for steps in checkpoints:
            if (directory / f'validation_{steps}.json').exists():
                continue
            command = [sys.executable, str(ROOT / f'trainer/train_{stage}.py'),
                '--data_path', str(DATA / ('grpo.jsonl' if stage == 'grpo' else 'train.jsonl')),
                '--device', 'cuda:0', '--dtype', 'float32', '--epochs', str(epochs),
                '--batch_size', '2' if stage == 'grpo' else '4',
                '--accumulation_steps', '4' if stage == 'grpo' else '2',
                '--learning_rate', str(lr), '--max_seq_len', '768', '--max_prompt_len', '256',
                '--max_steps', str(steps), '--save_dir', str(directory), '--checkpoint_dir', str(directory),
                '--save_weight', name, '--seed', '42', '--log_interval', '8', '--save_interval', '32'] + extra
            if stage != 'grpo':
                command += ['--preference_sft']
            score_path = directory / f'validation_{steps}.json'
            if completed < steps:
                if completed:
                    command += ['--from_resume', '1']
                write(directory / f'command_{steps}.json', command)
                print(f'TRAIN {name} -> {steps}', flush=True)
                with (directory / 'train.log').open('a') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                        env={**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
                completed = steps
            if score_path.exists():
                scores = json.loads(score_path.read_text())
            else:
                model = load_policy(config, ROOT / config['base_path'], 'cuda:0', checkpoint, adapter)
                scores, _ = evaluate_likelihood(model, tokenizer, DATA / 'validation.jsonl', config, 'cuda:0', val_pairs)
                del model
                gc.collect(); torch.cuda.empty_cache()
                write(score_path, scores)
            row = {'stage': stage, 'trial': name, 'updates': steps, 'learning_rate': lr, 'validation': scores}
            history.append(row)
            accuracy = scores['preference_accuracy_normalized']
            print(f'VALIDATION {name} {steps}: {accuracy:.2%}, NLL {scores["chosen_token_nll"]:.4f}', flush=True)
            key = (accuracy, -scores['chosen_token_nll'])
            previous = best.get(stage)
            if previous is None or key > (previous['validation']['preference_accuracy_normalized'], -previous['validation']['chosen_token_nll']):
                best[stage] = row
                shutil.copyfile(checkpoint, OUTPUT / f'{stage}_best_resume.pth')
                if adapter:
                    shutil.copyfile(adapter, OUTPUT / 'lora_best_768.pth')
                write(OUTPUT / f'{stage}_best.json', row)
            write(OUTPUT / 'validation_history.json', history)
        metrics = directory / f'{name}_768.metrics.jsonl'
        metric_paths.append(metrics)
        plot_metrics([metrics], directory / 'curves')
    test_base = [json.loads(x) for x in (ROOT / 'experiments/runs/zh_preference_20261029/baseline/pairs.jsonl').read_text().splitlines()]
    results = {}
    for stage in best:
        model = load_policy(config, ROOT / config['base_path'], 'cuda:0', OUTPUT / f'{stage}_best_resume.pth',
                            OUTPUT / 'lora_best_768.pth' if stage == 'lora' else None)
        scores, pairs = evaluate_likelihood(model, tokenizer, DATA / 'test.jsonl', config, 'cuda:0', test_base)
        write(OUTPUT / f'{stage}_selected_test.json', scores)
        (OUTPUT / f'{stage}_selected_test_pairs.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in pairs))
        if stage != 'lora':
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, OUTPUT / f'{stage}_selected_768.pth')
        del model
        gc.collect(); torch.cuda.empty_cache()
        results[stage] = {'selection': best[stage], 'test': scores}
        print(f'FINAL TEST {stage}: {scores["preference_accuracy_normalized"]:.2%}', flush=True)
    write(OUTPUT / 'results.metrics.json', results)
    plot_metrics(metric_paths, OUTPUT / 'figures')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
