"""Batched left-padded generation with contiguous groups and explicit EOS lengths."""
import torch
from trainer.alignment_objectives import completion_mask


@torch.inference_mode()
def sample_completions(model, tokenizer, prompts, device, max_new_tokens, group_size=1, do_sample=True):
    expanded = [prompt for prompt in prompts for _ in range(group_size)]
    width = max(len(prompt) for prompt in expanded)
    ids = torch.full((len(expanded), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for i, prompt in enumerate(expanded):
        ids[i, -len(prompt):] = torch.tensor(prompt, device=device)
        attention[i, -len(prompt):] = 1
    outputs = model.generate(ids, attention_mask=attention, max_new_tokens=max_new_tokens,
        temperature=1., top_p=1., top_k=0, do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id, use_cache=True)
    completions = outputs[:, width:]
    masks = completion_mask(completions, tokenizer.eos_token_id)
    return [completion[mask] for completion, mask in zip(completions, masks)]
