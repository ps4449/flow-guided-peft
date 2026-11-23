"""Proposed q/v-only, final-rank-512 comparison from the team discussion.

Validate actual matrix dimensions: equal rank sums only imply equal A/B budgets
when every targeted matrix has the same input+output width.
"""
import re

TOTAL_RANK = 512
MODULE = re.compile(r'layers\.(\d+)\.self_attn\.(q_proj|v_proj)')


def validate_budget(spec, model):
    targets = {name.removeprefix('model.'): module
               for name, module in model.named_modules()
               if name.startswith('model.') and MODULE.fullmatch(name.removeprefix('model.'))}
    if len(targets) != 64 or {int(MODULE.fullmatch(n)[1]) for n in targets} != set(range(32)):
        raise ValueError('Expected 32 language-model blocks with q_proj/v_proj only')
    costs = {m.in_features + m.out_features for m in targets.values()}
    if len(costs) != 1:
        raise ValueError('Unequal target dimensions: use a weighted parameter budget instead')
    cost = costs.pop()
    if 'adalora' in spec:
        if set(spec) != {'adalora'}:
            raise ValueError('Do not mix static ranks and AdaLoRA configuration')
        cfg = spec['adalora']
        if sorted(cfg['target_modules']) != ['q_proj', 'v_proj']:
            raise ValueError('Shared protocol targets q_proj/v_proj only')
        init, target = cfg['init_r'], cfg['target_r']
        if type(init) is not int or type(target) is not int or not 0 < target <= init:
            raise ValueError('AdaLoRA requires integer 0 < target_r <= init_r')
        total = len(targets) * target
        initial = len(targets) * init
        report = dict(method='adalora', initial_rank_sum=initial,
                      initial_adapter_parameters=initial * (cost + 1),
                      target_effective_parameters_nominal=total * (cost + 1),
                      note='Final effective budget only; initial and allocated budgets differ from static LoRA')
    else:
        if not spec or any(n not in targets or type(r) is not int or r < 0 for n, r in spec.items()):
            raise ValueError('Static ranks must be nonnegative integers on language-model q/v only')
        total = sum(spec.values())
        report = dict(method='static_lora', allocated_adapter_parameters=total * cost)
    if total != TOTAL_RANK:
        raise ValueError(f'Shared final rank budget is {TOTAL_RANK}, got {total}')
    return dict(final_rank_sum=total, matrix_count=64, per_rank_AB_cost=cost,
                uniform_reference_parameters=TOTAL_RANK * cost, **report)
