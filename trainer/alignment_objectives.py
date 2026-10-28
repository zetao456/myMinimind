"""DPO (Rafailov et al.) and outcome-supervised GRPO (DeepSeekMath)."""
import importlib
import re
import unicodedata
import torch
from torch.nn import functional as F


def token_logps(model, input_ids, attention_mask=None):
    logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    targets = input_ids[:, 1:]
    return logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)


def sequence_logps(model, input_ids, labels, attention_mask=None):
    mask = labels[:, 1:] != -100
    if not mask.any(dim=1).all():
        raise ValueError('Every sequence must have at least one supervised completion token')
    return (token_logps(model, input_ids, attention_mask) * mask).sum(-1)


def dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen_reward = beta * (chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss, {'chosen_reward': chosen_reward.detach().mean(),
                  'rejected_reward': rejected_reward.detach().mean(),
                  'preference_accuracy': (chosen_reward > rejected_reward).float().mean().detach()}


def group_advantages(rewards, group_size):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError('Each prompt requires at least two rewards in contiguous groups')
    grouped = rewards.reshape(-1, group_size)
    return ((grouped - grouped.mean(-1, keepdim=True)) /
            (grouped.std(-1, keepdim=True, unbiased=False) + 1e-4)).reshape(-1)


def completion_mask(completions, eos_token_id):
    """Include the first EOS and exclude all following tokens (also when pad=EOS)."""
    eos = completions.eq(eos_token_id)
    return (eos.cumsum(-1) - eos.long()).eq(0)


def grpo_loss(logps, old_logps, ref_logps, mask, advantages, beta=0.04, clip_epsilon=0.2):
    ratio = (logps - old_logps.detach()).exp()
    advantages = advantages.detach().unsqueeze(-1)
    surrogate = torch.minimum(ratio * advantages,
                              ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages)
    delta = ref_logps.detach() - logps
    kl = delta.exp() - delta - 1
    lengths = mask.sum(-1).clamp_min(1)
    loss = (((beta * kl - surrogate) * mask).sum(-1) / lengths).mean()
    return loss, {'kl': ((kl.detach() * mask).sum(-1) / lengths).mean(),
                  'clip_fraction': (((ratio.detach() - 1).abs() > clip_epsilon) * mask).sum() / mask.sum().clamp_min(1)}


def extract_answer(text):
    text = str(text).strip()
    answers = re.findall(r'<answer>(.*?)</answer>', text, flags=re.S)
    if answers:
        text = answers[-1]
    elif '####' in text:
        text = text.rsplit('####', 1)[-1]
    else:
        boxed = re.findall(r'\\boxed\{([^{}]*)\}', text)
        if boxed:
            text = boxed[-1]
    return ' '.join(unicodedata.normalize('NFKC', text).strip().casefold().split())


def exact_match_reward(prompts, completions, records):
    """Exact text matching only; no symbolic math equivalence or code execution."""
    scores = []
    for response, row in zip(completions, records):
        answer = row.get('answer')
        if answer is None:
            raise ValueError('exact_match reward requires an answer field in every row')
        answers = answer if isinstance(answer, list) else [answer]
        scores.append(float(extract_answer(response) in [extract_answer(a) for a in answers]))
    return scores


def load_reward_function(name):
    if name == 'exact_match':
        return exact_match_reward
    module, function = name.split(':', 1)
    return getattr(importlib.import_module(module), function)
