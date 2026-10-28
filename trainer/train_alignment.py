"""Shared native PyTorch runner for LoRA SFT, DPO and GRPO.

Use train_lora.py / train_dpo.py / train_grpo.py as entry points.
"""
import argparse
import copy
from contextlib import nullcontext
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from dataset.lm_dataset import SFTDataset, CachedSFTDataset, build_sft_cache
from dataset.alignment_dataset import DPODataset, PreferenceSFTDataset, GRPODataset, collate_prompts, encode_prompt
from model.model_myminimind import MyMiniMindConfig, MiniMindForCausalLM
from model.model_lora import (apply_lora, load_lora, save_lora, mark_only_lora_trainable,
                              unwrap_model)
from trainer.rollout import sample_completions
from trainer.metrics import MetricWriter, PerformanceMonitor
from trainer.alignment_objectives import (sequence_logps, token_logps, dpo_loss,
    completion_mask, group_advantages, grpo_loss, load_reward_function)
from trainer.trainer_utils import setup_seed, get_lr, SkipBatchSampler, Logger, is_main_process

ROOT = Path(__file__).resolve().parents[1]


def parser_for(stage):
    parser = argparse.ArgumentParser(description=f'MyMiniMind {stage.upper()} training')
    parser.add_argument('--data_path', required=True, help='Training JSONL; see docs/training/alignment.md')
    parser.add_argument('--tokenizer_path', default=str(ROOT / 'model'))
    parser.add_argument('--base_dir', default=str(ROOT / 'out'))
    parser.add_argument('--from_weight', default='full_sft', help='Base weight prefix, or none for smoke tests')
    parser.add_argument('--save_dir', default=str(ROOT / 'out'))
    parser.add_argument('--checkpoint_dir', default=str(ROOT / 'checkpoints'))
    parser.add_argument('--save_weight', default={'lora': 'lora_sft', 'sft': 'full_sft'}.get(stage, stage))
    parser.add_argument('--from_resume', type=int, choices=[0, 1], default=0)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--use_moe', type=int, choices=[0, 1], default=0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1 if stage == 'grpo' else 4)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default={'lora': 1e-4, 'sft': 1e-5}.get(stage, 1e-6))
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--max_seq_len', type=int, default=768)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=10, help='Optimizer updates')
    parser.add_argument('--save_interval', type=int, default=100, help='Optimizer updates, at rollout boundaries')
    parser.add_argument('--max_steps', type=int, default=0, help='Stop after this many optimizer updates; 0=all')
    parser.add_argument('--use_lora', action='store_true', help='Use adapters for DPO/GRPO (always on for LoRA SFT)')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--lora_targets', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    parser.add_argument('--lora_path', help='Optional starting adapter (its saved config takes precedence)')
    parser.add_argument('--max_prompt_len', type=int, default=384)
    if stage in ('lora', 'sft'):
        parser.add_argument('--cache_data', action='store_true')
        parser.add_argument('--preference_sft', action='store_true', help='Use chosen replies from preference pairs, with identical DPO token masks')
    else:
        parser.add_argument('--reference_weight', help='Optional reference weight prefix in base_dir')
        parser.add_argument('--beta', type=float, default=0.04 if stage == 'grpo' else 0.1)
    if stage == 'grpo':
        parser.add_argument('--group_size', type=int, default=4)
        parser.add_argument('--max_new_tokens', type=int, default=256)
        parser.add_argument('--temperature', type=float, default=1.0)
        parser.add_argument('--open_thinking', action='store_true', help='Leave the think block open in rollout prompts')
        parser.add_argument('--clip_epsilon', type=float, default=0.2)
        parser.add_argument('--num_iterations', type=int, default=1, help='Optimizer passes per fixed rollout window')
        parser.add_argument('--reward_fn', default='exact_match', help='exact_match or importable module:function')
        parser.add_argument('--reward_model_path', help='Optional local model implementing get_score(tokenizer, messages)')
    return parser


def validate_args(args, stage):
    for name in ('epochs', 'batch_size', 'accumulation_steps', 'max_seq_len', 'log_interval', 'save_interval'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.learning_rate <= 0 or args.grad_clip <= 0 or args.max_steps < 0:
        raise ValueError('learning_rate/grad_clip must be positive and max_steps must be nonnegative')
    if stage not in ('lora', 'sft'):
        if not 1 <= args.max_prompt_len <= args.max_seq_len - 2 or args.beta < 0:
            raise ValueError('Require 1 <= max_prompt_len <= max_seq_len - 2 and beta >= 0')
    if stage == 'grpo':
        if args.group_size < 2 or args.num_iterations < 1 or args.max_new_tokens < 1:
            raise ValueError('GRPO requires group_size >= 2 and positive iterations/generation length')
        if args.temperature != 1.0:
            raise ValueError('GRPO samples the raw policy: temperature must be 1.0 to match PPO log probabilities')
        if not 0 < args.clip_epsilon < 1:
            raise ValueError('clip_epsilon must be in (0, 1)')
        if args.max_prompt_len + args.max_new_tokens > args.max_seq_len:
            raise ValueError('max_prompt_len + max_new_tokens must fit max_seq_len')
        if args.max_steps and args.max_steps % args.num_iterations:
            raise ValueError('max_steps must be a multiple of num_iterations (save complete rollouts)')
    if args.max_seq_len > 32768:
        raise ValueError('max_seq_len exceeds the model RoPE buffer')


def atomic_save(value, path):
    torch.save(value, str(path) + '.tmp')
    os.replace(str(path) + '.tmp', path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in unwrap_model(model).state_dict().items()}


def prepare_grpo(model, reference, rows, tokenizer, args, reward_fn):
    """Generate contiguous G-sized groups without left-padding/position ambiguity."""
    policy = unwrap_model(model)
    sequences, prompt_lengths, records, texts = [], [], [], []
    with torch.no_grad():
        prompts = [encode_prompt(tokenizer, row['messages'], args.max_prompt_len, args.open_thinking) for row in rows]
        completions = sample_completions(policy, tokenizer, prompts, args.device,
                                         args.max_new_tokens, args.group_size)
        for i, completion in enumerate(completions):
            row = rows[i // args.group_size]
            prompt = prompts[i // args.group_size]
            sequences.append(torch.cat([torch.tensor(prompt, device=args.device), completion]))
            prompt_lengths.append(len(prompt))
            records.append(row)
            texts.append(tokenizer.decode(completion, skip_special_tokens=True))
        length = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), length), tokenizer.pad_token_id, dtype=torch.long, device=args.device)
        attention = torch.zeros_like(ids)
        mask = torch.zeros((len(sequences), length - 1), dtype=torch.bool, device=args.device)
        for i, (sequence, prompt_len) in enumerate(zip(sequences, prompt_lengths)):
            ids[i, :len(sequence)] = sequence
            attention[i, :len(sequence)] = 1
            mask[i, prompt_len - 1:len(sequence) - 1] = True
        rewards = torch.as_tensor(reward_fn([r['messages'] for r in records], texts, records),
                                  dtype=torch.float32, device=args.device)
        if rewards.shape != (len(sequences),) or not torch.isfinite(rewards).all():
            raise ValueError('Reward must return one finite scalar per completion')
        old = token_logps(policy, ids, attention).detach()
        ref = token_logps(reference, ids, attention).detach()
    return {'input_ids': ids, 'attention_mask': attention, 'mask': mask,
            'old_logps': old, 'ref_logps': ref, 'rewards': rewards,
            'advantages': group_advantages(rewards, args.group_size)}


def compute_loss(stage, model, reference, batch, args):
    if stage in ('lora', 'sft'):
        ids, labels = (t.to(args.device) for t in batch)
        if not (labels[:, 1:] != -100).any(dim=1).all():
            raise ValueError('SFT sample has no assistant tokens after truncation; increase max_seq_len')
        out = model(ids, labels=labels)
        return out.loss + out.aux_loss, {'sft_loss': out.loss.detach()}
    if stage == 'dpo':
        batch = {k: v.to(args.device) for k, v in batch.items()}
        ids = torch.cat([batch['chosen_input_ids'], batch['rejected_input_ids']])
        labels = torch.cat([batch['chosen_labels'], batch['rejected_labels']])
        attention = torch.cat([batch['chosen_attention_mask'], batch['rejected_attention_mask']])
        chosen, rejected = sequence_logps(model, ids, labels, attention).chunk(2)
        with torch.no_grad():
            ref_chosen, ref_rejected = sequence_logps(reference, ids, labels, attention).chunk(2)
        return dpo_loss(chosen, rejected, ref_chosen, ref_rejected, args.beta)
    logps = token_logps(model, batch['input_ids'], batch['attention_mask'])
    loss, metrics = grpo_loss(logps, batch['old_logps'], batch['ref_logps'], batch['mask'],
                              batch['advantages'], args.beta, args.clip_epsilon)
    metrics['reward'] = batch['rewards'].mean()
    metrics['completion_length'] = batch['mask'].sum(-1).float().mean()
    groups = batch['rewards'].reshape(-1, args.group_size)
    metrics['zero_std_groups'] = (groups.std(-1, unbiased=False) == 0).float().mean()
    return loss, metrics


def main(stage):
    args = parser_for(stage).parse_args()
    validate_args(args, stage)
    args.use_lora = args.use_lora or stage == 'lora'
    if args.lora_path and not args.use_lora:
        raise ValueError('--lora_path requires --use_lora')
    distributed = int(os.environ.get('RANK', -1)) >= 0
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        if args.device.startswith('cuda'):
            torch.cuda.set_device(local_rank)
            args.device = f'cuda:{local_rank}'
        dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo')
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    setup_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = MyMiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                              use_moe=bool(args.use_moe), vocab_size=len(tokenizer))
    model = MiniMindForCausalLM(config)
    suffix = '_moe' if args.use_moe else ''
    name = f'{args.save_weight}_{args.hidden_size}{suffix}'
    for directory in (args.save_dir, args.checkpoint_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)
    resume_path = Path(args.checkpoint_dir) / f'{name}_resume.pth'
    # Local training state contains Python/NumPy RNG objects; load only your own checkpoints.
    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False) if args.from_resume else None
    if checkpoint:
        if checkpoint['stage'] != stage or checkpoint['world_size'] != world:
            raise ValueError('Resume requires the same stage and world size')
        mutable = {'from_resume', 'max_steps', 'log_interval', 'save_interval', 'device', 'num_workers'}
        changed = [k for k, v in vars(args).items() if k not in mutable and checkpoint['args'].get(k) != v]
        if changed:
            raise ValueError(f'Resume configuration differs: {changed}')
    if args.from_weight != 'none':
        base_path = Path(args.base_dir) / f'{args.from_weight}_{args.hidden_size}{suffix}.pth'
        model.load_state_dict(torch.load(base_path, map_location='cpu', weights_only=True), strict=True)
    if args.use_lora:
        if args.lora_path:
            load_lora(model, args.lora_path)
        else:
            apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout, args.lora_targets)
        mark_only_lora_trainable(model)
    reference = None
    if stage not in ('lora', 'sft'):
        reference = copy.deepcopy(model)
        if args.reference_weight:
            if args.use_lora:
                # An explicit reference is a plain full checkpoint, without the policy adapter.
                reference = MiniMindForCausalLM(config)
            path = Path(args.base_dir) / f'{args.reference_weight}_{args.hidden_size}{suffix}.pth'
            reference.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        if checkpoint:
            reference.load_state_dict(checkpoint['reference'], strict=True)
        reference.requires_grad_(False).eval().to(args.device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    model.to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    Logger(f'{stage}: {sum(p.numel() for p in params):,} trainable / {sum(p.numel() for p in model.parameters()):,} total parameters')
    if stage in ('lora', 'sft'):
        if args.preference_sft:
            dataset = PreferenceSFTDataset(args.data_path, tokenizer, args.max_seq_len, args.max_prompt_len)
        elif args.cache_data:
            cache_dir = f'{args.data_path}.cached_{args.max_seq_len}'
            if is_main_process() and not os.path.exists(cache_dir):
                build_sft_cache(args.data_path, tokenizer, args.max_seq_len, cache_dir)
            if distributed:
                dist.barrier()
            dataset = CachedSFTDataset(cache_dir)
        else:
            dataset = SFTDataset(args.data_path, tokenizer, args.max_seq_len)
    elif stage == 'dpo':
        dataset = DPODataset(args.data_path, tokenizer, args.max_seq_len, args.max_prompt_len)
    else:
        dataset = GRPODataset(args.data_path)
    reward_fn = None
    if stage == 'grpo':
        if args.reward_model_path:
            from trainer.trainer_utils import LMForRewardModel
            reward_model = LMForRewardModel(args.reward_model_path, args.device,
                              torch.float16 if args.device.startswith('cuda') else torch.float32)
            reward_fn = lambda prompts, completions, records: [reward_model.get_score(p, c) for p, c in zip(prompts, completions)]
        else:
            reward_fn = load_reward_function(args.reward_fn)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed)
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    amp = args.device.startswith('cuda') and args.dtype != 'float32'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and args.dtype == 'float16' and stage != 'grpo')
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
    if distributed:
        model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])] if args.device.startswith('cuda') else None,
                    find_unused_parameters=bool(args.use_moe))
    setup_seed(args.seed + rank)
    # Disabling dropout keeps old-policy, rollout and update probabilities consistent.
    model.train(stage in ('lora', 'sft'))
    iterations = args.num_iterations if stage == 'grpo' else 1
    batches_per_epoch = math.ceil(len(sampler) / args.batch_size)
    total_updates = math.ceil(batches_per_epoch / args.accumulation_steps) * args.epochs * iterations
    update = checkpoint['update'] if checkpoint else 0
    start_epoch = checkpoint['epoch'] if checkpoint else 0
    start_batch = checkpoint['batch'] if checkpoint else 0
    log_path = Path(args.save_dir) / f'{name}.metrics.jsonl'
    writer = MetricWriter(log_path, {'stage': stage, 'args': vars(args),
        'step_unit': 'optimizer_update', 'loss_contract': {'lora': 'assistant_token_cross_entropy',
        'sft': 'assistant_token_cross_entropy', 'dpo': 'dpo_logistic', 'grpo': 'grpo_clipped_surrogate'}[stage],
        'trainable_parameters': sum(p.numel() for p in params),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'torch_version': torch.__version__, 'cuda_device': torch.cuda.get_device_name(args.device)
        if args.device.startswith('cuda') else None}, resume_step=update if checkpoint else None) if is_main_process() else None
    monitor = PerformanceMonitor(args.device, checkpoint.get('elapsed_seconds', 0.) if checkpoint else 0.)

    def save(epoch, batch):
        states = [None] * world
        if distributed:
            dist.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if is_main_process():
            if args.use_lora:
                save_lora(model, Path(args.save_dir) / f'{name}.pth')
            else:
                atomic_save({k: v.half() for k, v in cpu_state(model).items()}, Path(args.save_dir) / f'{name}.pth')
            atomic_save({'stage': stage, 'args': vars(args), 'model': cpu_state(model),
                         'reference': cpu_state(reference) if reference is not None else None,
                         'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                         'epoch': epoch, 'batch': batch, 'update': update,
                         'world_size': world, 'rng': states, 'elapsed_seconds': monitor.elapsed}, resume_path)
        if distributed:
            dist.barrier()

    stop = bool(args.max_steps and update >= args.max_steps)
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        sampler.set_epoch(epoch)
        consumed = start_batch if epoch == start_epoch else 0
        loader = DataLoader(dataset, batch_sampler=SkipBatchSampler(sampler, args.batch_size, consumed),
                            collate_fn=collate_prompts if stage == 'grpo' else None,
                            num_workers=args.num_workers, pin_memory=args.device.startswith('cuda'),
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        iterator = iter(loader)
        if checkpoint:
            restore_rng(checkpoint['rng'][rank])
            checkpoint = None
        while True:
            monitor.begin()
            preparation_started = time.perf_counter()
            window = list(islice(iterator, args.accumulation_steps))
            if not window:
                break
            if stage == 'grpo':
                # Rollout, old and reference scores all use FP32 (no sampling warps).
                window = [prepare_grpo(model, reference, rows, tokenizer, args, reward_fn) for rows in window]
            monitor.synchronize()
            preparation_seconds = time.perf_counter() - preparation_started
            sizes = [len(b['rewards']) if stage == 'grpo' else
                     (len(b['chosen_input_ids']) if stage == 'dpo' else len(b[0])) for b in window]
            for _ in range(iterations):
                monitor.begin(reset_peak=False)
                optimizer.zero_grad(set_to_none=True)
                metrics_sum = {}
                lr = get_lr(update, total_updates, args.learning_rate)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                for i, (batch, size) in enumerate(zip(window, sizes)):
                    sync = model.no_sync() if distributed and i < len(window) - 1 else nullcontext()
                    with sync:
                        # GRPO keeps probability calculations identical to the FP32 rollout.
                        with torch.autocast('cuda', dtype=dtype, enabled=amp and stage != 'grpo'):
                            loss, metrics = compute_loss(stage, model, reference, batch, args)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'Non-finite {stage} loss')
                        weight = size / sum(sizes)
                        scaler.scale(loss * weight).backward()
                    for key, value in {'loss': loss.detach(), **metrics}.items():
                        metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value) * weight
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip,
                                               error_if_nonfinite=not scaler.is_enabled())
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    Logger('FP16 gradient overflow: skipped optimizer update and reduced loss scale')
                    continue
                update += 1
                keys = sorted(metrics_sum)
                values = torch.tensor([metrics_sum[k] for k in keys], device=args.device)
                if distributed:
                    dist.all_reduce(values)
                    values /= world
                if stage in ('lora', 'sft'):
                    tokens = sum(int((b[0] != tokenizer.pad_token_id).sum()) for b in window)
                    supervised = sum(int((b[1][:, 1:] != -100).sum()) for b in window)
                    generated = 0
                elif stage == 'dpo':
                    tokens = sum(int(b[key + '_attention_mask'].sum()) for b in window for key in ('chosen', 'rejected'))
                    supervised = sum(int((b[key + '_labels'][:, 1:] != -100).sum()) for b in window for key in ('chosen', 'rejected'))
                    generated = 0
                else:
                    tokens = sum(int(b['attention_mask'].sum()) for b in window)
                    supervised = sum(int(b['mask'].sum()) for b in window)
                    generated = supervised / iterations
                performance = monitor.finish(sum(sizes) / (args.group_size if stage == 'grpo' else 1),
                    tokens, supervised, generated, extra_seconds=preparation_seconds / iterations)
                report = {'step': update, 'update': update, 'epoch': epoch + 1, 'learning_rate': lr,
                          **performance,
                          **dict(zip(keys, values.cpu().tolist()))}
                if is_main_process():
                    writer.write(report)
                    if update % args.log_interval == 0:
                        Logger(json.dumps(report))
            consumed += len(window)
            stop = bool(args.max_steps and update >= args.max_steps)
            if update % args.save_interval < iterations or consumed == batches_per_epoch or stop:
                save(epoch + 1 if consumed == batches_per_epoch else epoch,
                     0 if consumed == batches_per_epoch else consumed)
            if stop:
                break
        if stop:
            break
    Logger(f'Finished {stage}: {update} optimizer updates; output {Path(args.save_dir) / (name + ".pth")}')
    if distributed:
        dist.destroy_process_group()
