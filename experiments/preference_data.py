"""Convert genuine Chinese preference pairs into disjoint, same-prompt training views."""
import hashlib
import json
from pathlib import Path
import random
import statistics
import unicodedata

from transformers import AutoTokenizer
from experiments.download_preference import ROOT, sha256


def prompt_key(messages):
    normalized = [{'role': m['role'], 'content': ' '.join(
        unicodedata.normalize('NFKC', m['content']).split()).casefold()} for m in messages]
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def prepare(directory, train_size=1024, validation_size=128, test_size=256, seed=20261029,
            max_length=768, max_prompt_length=256, fixed_holdout=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / 'dataset/downloads/dpo_en_zh_20k/dpo_zh.json'
    source = json.loads(raw_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    roles = {'human': 'user', 'gpt': 'assistant', 'system': 'system', 'user': 'user', 'assistant': 'assistant'}
    seen, eligible = set(), []
    exclusions = {'duplicate_prompt': 0, 'long_prompt_or_reply': 0, 'empty_or_identical': 0}
    for index, row in enumerate(source):
        messages = [{'role': roles[m['from']], 'content': m['value']} for m in row['conversations']]
        chosen, rejected = row['chosen']['value'], row['rejected']['value']
        key = prompt_key(messages)
        if key in seen:
            exclusions['duplicate_prompt'] += 1
            continue
        seen.add(key)
        if not chosen.strip() or not rejected.strip() or chosen.strip() == rejected.strip():
            exclusions['empty_or_identical'] += 1
            continue
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=True)
        prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        generation_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
        generation_len = len(tokenizer(generation_prompt, add_special_tokens=False).input_ids)
        lengths = []
        for text in (chosen, rejected):
            rendered = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': text}], tokenize=False)
            if not rendered.startswith(prompt):
                raise ValueError(f'Unexpected chat-template prefix at source row {index}')
            lengths.append(prompt_len + len(tokenizer(rendered[len(prompt):], add_special_tokens=False).input_ids))
        if max(prompt_len, generation_len) > max_prompt_length or max(lengths) > max_length:
            exclusions['long_prompt_or_reply'] += 1
            continue
        eligible.append({'prompt': messages, 'chosen': chosen, 'rejected': rejected,
                         'source_index': index, 'prompt_sha256': key, 'prompt_tokens': prompt_len,
                         'chosen_tokens': lengths[0] - prompt_len, 'rejected_tokens': lengths[1] - prompt_len})
    random.Random(seed).shuffle(eligible)
    required = train_size + validation_size + test_size
    if len(eligible) < required:
        raise ValueError(f'Only {len(eligible)} eligible pairs, need {required}')
    if fixed_holdout is None:
        splits = {'train': eligible[:train_size],
                  'validation': eligible[train_size:train_size + validation_size],
                  'test': eligible[train_size + validation_size:required]}
    else:
        # Expand training data while retaining the original validation/test questions.
        splits = {name: [json.loads(line) for line in (Path(fixed_holdout) / f'{name}.jsonl').read_text().splitlines()]
                  for name in ('validation', 'test')}
        held_out = {row['prompt_sha256'] for rows in splits.values() for row in rows}
        available = [row for row in eligible if row['prompt_sha256'] not in held_out]
        if len(available) < train_size:
            raise ValueError(f'Only {len(available)} training pairs after retaining the holdout')
        splits['train'] = available[:train_size]
    def save(name, rows):
        rows = list(rows)
        path = directory / name
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        return {'sha256': sha256(path), 'rows': len(rows)}
    files = {}
    for split, rows in splits.items():
        files[split + '.jsonl'] = save(split + '.jsonl', rows)
    train = splits['train']
    files['sft.jsonl'] = save('sft.jsonl', [{'conversations': r['prompt'] + [{'role': 'assistant', 'content': r['chosen']}]} for r in train])
    files['dpo.jsonl'] = save('dpo.jsonl', train)
    files['grpo.jsonl'] = save('grpo.jsonl', [{'prompt': r['prompt'], 'source_index': r['source_index'], 'prompt_sha256': r['prompt_sha256']} for r in train])
    keys = [set(r['prompt_sha256'] for r in splits[name]) for name in splits]
    assert all(not keys[i] & keys[j] for i in range(3) for j in range(i))
    metadata = {'source_file': str(raw_path.relative_to(ROOT)), 'source_sha256': sha256(raw_path),
                'raw_rows': len(source), 'eligible_rows': len(eligible), 'excluded': exclusions,
                'split_seed': seed, 'max_length': max_length, 'max_prompt_length': max_prompt_length,
                'files': files,
                'mean_token_lengths': {split: {name: statistics.mean(r[name] for r in rows)
                    for name in ('prompt_tokens', 'chosen_tokens', 'rejected_tokens')} for split, rows in splits.items()},
                'preference_labels': 'unchanged chosen/rejected from the downloaded dataset'}
    (directory / 'preparation.json').write_text(json.dumps(metadata, indent=2))
    return metadata
