"""Frozen public Skywork sequence-classification reward, using only local files."""
import os
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


class RewardScorer:
    def __init__(self, path=None, device='cuda:0', batch_size=8, max_length=1024):
        path = path or ROOT / 'models/skywork_reward_0.6b'
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path,
            torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
            local_files_only=True, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def score(self, prompts, completions):
        scores = []
        for start in range(0, len(prompts), self.batch_size):
            texts = [self.tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': response}],
                     tokenize=False, add_generation_prompt=False)
                     for messages, response in zip(prompts[start:start + self.batch_size], completions[start:start + self.batch_size])]
            tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                                    max_length=self.max_length).to(self.device)
            values = self.model(**tokens).logits.float().flatten()
            scores.extend(values.cpu().tolist())
        return scores


_scorer = None


def reward(prompts, completions, records):
    global _scorer
    if _scorer is None:
        _scorer = RewardScorer(device=os.environ.get('REWARD_DEVICE', 'cuda:0'))
    return _scorer.score(prompts, completions)
