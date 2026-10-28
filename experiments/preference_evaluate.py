"""Evaluate all four methods against unchanged public preference labels and prompts."""
import json
import math
from pathlib import Path

import torch
from dataset.alignment_dataset import DPODataset, JsonlDataset, encode_prompt
from model.model_lora import load_lora
from model.model_myminimind import MiniMindForCausalLM, MyMiniMindConfig
from trainer.alignment_objectives import token_logps
from trainer.rollout import sample_completions


def load_policy(config, base_path, device, checkpoint_path=None, adapter_path=None):
    model = MiniMindForCausalLM(MyMiniMindConfig(hidden_size=config['hidden_size'],
        num_hidden_layers=config['num_hidden_layers']))
    if checkpoint_path is None:
        model.load_state_dict(torch.load(base_path, map_location='cpu', weights_only=True), strict=True)
    if adapter_path:
        load_lora(model, adapter_path)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def evaluate_likelihood(model, tokenizer, path, config, device, baseline_pairs=None):
    dataset = DPODataset(path, tokenizer, config['max_seq_len'], config['max_prompt_len'])
    records = list(JsonlDataset(path))
    results = []
    total_nll, total_tokens = 0., 0
    for start in range(0, len(dataset), config['batch_size']):
        rows = [dataset[i] for i in range(start, min(start + config['batch_size'], len(dataset)))]
        scores = {}
        for name in ('chosen', 'rejected'):
            ids = torch.stack([r[name + '_input_ids'] for r in rows]).to(device)
            labels = torch.stack([r[name + '_labels'] for r in rows]).to(device)
            attention = torch.stack([r[name + '_attention_mask'] for r in rows]).to(device)
            mask = labels[:, 1:] != -100
            logps = token_logps(model, ids, attention)
            sums, lengths = (logps * mask).sum(-1).cpu().tolist(), mask.sum(-1).cpu().tolist()
            scores[name] = list(zip(sums, lengths))
            if name == 'chosen':
                total_nll -= sum(sums)
                total_tokens += sum(lengths)
        for offset, (chosen, rejected) in enumerate(zip(scores['chosen'], scores['rejected'])):
            index = start + offset
            margin = chosen[0] - rejected[0]
            base_margin = (baseline_pairs[index]['chosen_logp'] - baseline_pairs[index]['rejected_logp']) if baseline_pairs is not None else margin
            gain = margin - base_margin
            results.append({'source_index': records[index]['source_index'],
                'prompt_sha256': records[index]['prompt_sha256'],
                'chosen_logp': chosen[0], 'chosen_tokens': chosen[1],
                'rejected_logp': rejected[0], 'rejected_tokens': rejected[1],
                'sum_correct': chosen[0] > rejected[0],
                'normalized_correct': chosen[0] / chosen[1] > rejected[0] / rejected[1],
                'margin_gain': gain, 'dpo_loss': float(torch.nn.functional.softplus(torch.tensor(-.1 * gain)))})
    metrics = {'test_pairs': len(results), 'chosen_tokens': total_tokens,
        'chosen_token_nll': total_nll / total_tokens,
        'chosen_perplexity': math.exp(total_nll / total_tokens),
        'preference_accuracy_sum': sum(r['sum_correct'] for r in results) / len(results),
        'preference_accuracy_normalized': sum(r['normalized_correct'] for r in results) / len(results),
        'dpo_loss_relative_to_base': sum(r['dpo_loss'] for r in results) / len(results),
        'margin_gain_mean': sum(r['margin_gain'] for r in results) / len(results)}
    return metrics, results


@torch.inference_mode()
def evaluate_generation(model, tokenizer, path, config, device, scorer, baseline_predictions=None):
    rows = list(JsonlDataset(path))[:config['generation_eval_size']]
    predictions = []
    for start in range(0, len(rows), config['batch_size']):
        batch = rows[start:start + config['batch_size']]
        prompts = [encode_prompt(tokenizer, row['prompt'], config['max_prompt_len']) for row in batch]
        completions = sample_completions(model, tokenizer, prompts, device,
                                         config['max_new_tokens'], do_sample=False)
        texts = [tokenizer.decode(c, skip_special_tokens=True) for c in completions]
        rewards = scorer.score([row['prompt'] for row in batch], texts)
        for row, completion, text, reward in zip(batch, completions, texts, rewards):
            predictions.append({'source_index': row['source_index'], 'prompt_sha256': row['prompt_sha256'],
                'prompt': row['prompt'], 'response': text, 'reward': reward,
                'generated_tokens': len(completion),
                'terminated_by_eos': bool((completion == tokenizer.eos_token_id).any())})
    if baseline_predictions is None:
        baseline_predictions = predictions
    for row, baseline in zip(predictions, baseline_predictions):
        assert row['prompt_sha256'] == baseline['prompt_sha256']
        row['reward_delta_to_base'] = row['reward'] - baseline['reward']
    count = len(predictions)
    metrics = {'generated_examples': count,
        'reward_mean': sum(r['reward'] for r in predictions) / count,
        'reward_delta_to_base': sum(r['reward_delta_to_base'] for r in predictions) / count,
        'reward_win_rate_vs_base': sum((r['reward_delta_to_base'] > 0) + .5 * (r['reward_delta_to_base'] == 0) for r in predictions) / count,
        'reward_tie_fraction': sum(r['reward_delta_to_base'] == 0 for r in predictions) / count,
        'mean_generated_tokens': sum(r['generated_tokens'] for r in predictions) / count,
        'truncated_fraction': sum(not r['terminated_by_eos'] for r in predictions) / count}
    return metrics, predictions


def save_evaluation(directory, metrics, pairs, predictions):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'evaluation.json').write_text(json.dumps(metrics, indent=2))
    for name, rows in [('pairs', pairs), ('predictions', predictions)]:
        (directory / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
