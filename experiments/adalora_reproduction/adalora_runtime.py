"""AdaLoRA-specific state and export helpers for the shared training loop."""
import copy
import torch


def configure(spec, total_steps, config_class):
    cfg = dict(spec)
    targets = cfg.pop('target_modules')
    if sorted(targets) != ['q_proj', 'v_proj']:
        raise ValueError('AdaLoRA shared experiment targets language-model q/v only')
    cfg.setdefault('tinit', max(1, int(total_steps * 0.1)))
    cfg.setdefault('tfinal', max(1, int(total_steps * 0.1)))
    cfg.setdefault('deltaT', 1 if total_steps <= 20 else 10)
    if not 0 <= cfg['tinit'] < total_steps - cfg['tfinal'] or cfg['tfinal'] < 1 or cfg['deltaT'] < 1:
        raise ValueError('AdaLoRA schedule needs an initial, allocation and final phase')
    return config_class(target_modules=r'model\.layers\.\d+\.self_attn\.(q_proj|v_proj)',
                        total_step=total_steps, **cfg)


def snapshot_loss(model):
    """Capture task loss before PEFT adds regularization via direct forward()."""
    base = model.get_base_model()
    original = base.forward
    captured = {}
    def forward(*args, **kwargs):
        out = original(*args, **kwargs)
        if out.loss is not None:
            captured['task'] = float(out.loss.detach())
        return out
    base.forward = forward
    return captured, lambda: setattr(base, 'forward', original)


def ranks(model):
    cfg = model.peft_config['default']
    pattern = cfg.rank_pattern
    modules = [m for m in model.modules() if hasattr(m, 'lora_E') and 'default' in m.lora_E]
    values = ([sum(bool(v) for v in mask) for mask in pattern.values()] if pattern
              else [m.lora_E['default'].shape[0] for m in modules])
    if len(values) != 64:
        raise RuntimeError('Expected 64 adapter rank entries')
    cost = modules[0].lora_A['default'].shape[1] + modules[0].lora_B['default'].shape[0] + 1
    return dict(retained_rank_sum=sum(values), rank_min=min(values), rank_max=max(values),
                effective_adapter_parameters=sum(values) * cost)


def allocator_state(model):
    allocator = model.base_model.rankallocator
    return dict(rank_pattern=copy.deepcopy(model.peft_config['default'].rank_pattern),
                **{key: {n: t.detach().cpu().clone() for n, t in getattr(allocator, key).items()}
                   for key in ('ipt', 'exp_avg_ipt', 'exp_avg_unc')})


def restore_allocator(model, state):
    params = dict(model.base_model.model.named_parameters())
    model.peft_config['default'].rank_pattern = state['rank_pattern']
    for key in ('ipt', 'exp_avg_ipt', 'exp_avg_unc'):
        setattr(model.base_model.rankallocator, key,
                {n: value.to(params[n].device) for n, value in state[key].items()})


def raw_adapter_state(model):
    # No PEFT compaction or mutation of live rank_pattern at a resume checkpoint.
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if 'lora_' in n or '.ranknum.' in n}


def export_full_shape(model, path):
    """Keep zero-masked slots and ranknum scaling; preserve mask in sidecar."""
    from pathlib import Path
    import json
    cfg = model.peft_config['default']
    pattern = copy.deepcopy(cfg.rank_pattern)
    if not pattern:
        raise RuntimeError('Refusing final AdaLoRA export without a rank pattern')
    # PEFT omits ranknum from its standard adapter state. Keeping init_r shapes
    # reconstructs the same initial scaling when loading, as checked in pilot.
    cfg.rank_pattern = None
    try:
        model.save_pretrained(path)
    finally:
        cfg.rank_pattern = pattern
    Path(path, 'rank_mask.json').write_text(json.dumps(pattern, indent=2) + '\n')
