"""LoRA adapters with portable configuration and unchanged base weight names."""
import os
import torch
from torch import nn
from torch.nn import functional as F


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=None, dropout=0.0):
        super().__init__()
        if rank < 1 or not 0 <= dropout < 1:
            raise ValueError('rank must be positive and dropout must be in [0, 1)')
        self.rank = rank
        self.alpha = rank if alpha is None else alpha
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(self.dropout(x))) * self.scaling


class LoRALinear(nn.Linear):
    def __init__(self, base, rank, alpha=None, dropout=0.0):
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.lora = LoRA(base.in_features, base.out_features, rank, alpha, dropout)
        self.lora.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.lora(x)


def unwrap_model(model):
    if hasattr(model, 'module'):
        model = model.module
    return getattr(model, '_orig_mod', model)


def apply_lora(model, rank=8, alpha=None, dropout=0.0, target_modules=None):
    """Default targets preserve the old square-linear (q/o) adapter behavior."""
    model = unwrap_model(model)
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError('LoRA is already installed')
    targets = target_modules.split(',') if isinstance(target_modules, str) else target_modules
    selected = [(n, m) for n, m in model.named_modules()
                if isinstance(m, nn.Linear) and
                (n.rsplit('.', 1)[-1] in targets if targets is not None
                 else m.in_features == m.out_features)]
    if not selected:
        raise ValueError('No linear layers matched target_modules')
    for name, module in selected:
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(module, rank, alpha, dropout))
    return model


def mark_only_lora_trainable(model):
    for name, param in model.named_parameters():
        param.requires_grad_('.lora.' in name)


def lora_checkpoint(model):
    model = unwrap_model(model)
    config = {n: {'rank': m.lora.rank, 'alpha': m.lora.alpha,
                  'dropout': m.lora.dropout.p}
              for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    if not config:
        raise ValueError('Model has no LoRA adapters')
    return {'format': 'myminimind_lora_v1', 'config': config,
            'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                           if '.lora.' in k}}


def load_lora(model, path):
    """Auto-install saved targets/rank; also accept legacy bare adapter dicts."""
    model = unwrap_model(model)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved['state_dict'] if saved.get('format') == 'myminimind_lora_v1' else saved
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    config = saved.get('config') if saved.get('format') == 'myminimind_lora_v1' else None
    if config is None:
        config = {k.removesuffix('.lora.A.weight'): {'rank': v.shape[0]}
                  for k, v in state.items() if k.endswith('.lora.A.weight')}
    if not config:
        raise ValueError('No LoRA adapters in checkpoint')
    existing = {n for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    if existing and existing != set(config):
        raise ValueError('Installed LoRA targets differ from checkpoint')
    for name, options in config.items():
        module = model.get_submodule(name)
        if not isinstance(module, LoRALinear):
            parent_name, _, child = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            module = LoRALinear(module, **options)
            setattr(parent, child, module)
        if module.lora.rank != options['rank']:
            raise ValueError(f'LoRA rank mismatch for {name}')
        module.lora.alpha = options.get('alpha', options['rank'])
        module.lora.scaling = module.lora.alpha / module.lora.rank
        module.lora.dropout.p = options.get('dropout', 0.0)
    expected = {k for k in model.state_dict() if '.lora.' in k}
    if set(state) != expected:
        raise ValueError('Adapter checkpoint has missing or unexpected parameters')
    model.load_state_dict(state, strict=False)
    return model


def save_lora(model, path):
    torch.save(lora_checkpoint(model), str(path) + '.tmp')
    os.replace(str(path) + '.tmp', path)


def merge_lora(model, lora_path, save_path):
    """Export a plain base-model state dict; the input model stays unmerged."""
    load_lora(model, lora_path)
    model = unwrap_model(model)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
             if '.lora.' not in k}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            delta = module.lora.B.weight.float() @ module.lora.A.weight.float()
            state[f'{name}.weight'] += (delta * module.lora.scaling).cpu().to(module.weight.dtype)
    torch.save(state, str(save_path) + '.tmp')
    os.replace(str(save_path) + '.tmp', save_path)
