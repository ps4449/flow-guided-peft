"""Shared LoRA parameter budget, so every method is compared at the same size.

Budget: LoRA on q_proj and v_proj of all 32 LLaVA-1.5-7B language layers,
average rank 8 -> 64 modules, ranks summing to 512, ~4.2M trainable parameters.
AdaLoRA uses the same budget: init rank 12, pruned to an average (target) rank of 8.

Rank file format (configs/ranks/<method>.json):
  {"layers.<i>.self_attn.q_proj": <rank>, "layers.<i>.self_attn.v_proj": <rank>, ...}

Check a rank file:
  python peft_lora/budget.py configs/ranks/<method>.json
"""
import json
import re
import sys

NUM_LAYERS = 32
HIDDEN = 4096                          # q_proj and v_proj are both 4096 -> 4096
TARGET_MODULES = ["q_proj", "v_proj"]
AVG_RANK = 8
ADALORA_INIT_RANK = 12

MODULES = [f"layers.{i}.self_attn.{m}" for i in range(NUM_LAYERS) for m in TARGET_MODULES]
RANK_SUM = AVG_RANK * len(MODULES)     # 512
PARAMS_PER_RANK = 2 * HIDDEN           # LoRA A (r x 4096) + B (4096 x r)
PARAM_BUDGET = RANK_SUM * PARAMS_PER_RANK

MODULE_RE = re.compile(rf"layers\.(\d+)\.self_attn\.({'|'.join(TARGET_MODULES)})")


def uniform_ranks():
    return {m: AVG_RANK for m in MODULES}


def adalora_spec():
    return {"adalora": {"init_r": ADALORA_INIT_RANK, "target_r": AVG_RANK, "target_modules": TARGET_MODULES}}


def num_params(ranks):
    return sum(ranks.values()) * PARAMS_PER_RANK


def check(ranks):
    """Returns a list of problems; empty means the rank file fits the budget."""
    if "adalora" in ranks:
        spec = ranks["adalora"]
        problems = []
        if spec.get("target_r") != AVG_RANK:
            problems.append(f"target_r is {spec.get('target_r')}, budget is {AVG_RANK}")
        if sorted(spec.get("target_modules", [])) != sorted(TARGET_MODULES):
            problems.append(f"target_modules must be {TARGET_MODULES}")
        return problems

    problems = []
    for key, r in ranks.items():
        match = MODULE_RE.fullmatch(key)
        if not match or int(match.group(1)) >= NUM_LAYERS:
            problems.append(f"{key}: not a budget module (allowed: q_proj/v_proj of layers 0-{NUM_LAYERS - 1})")
        if not isinstance(r, int) or r < 0:
            problems.append(f"{key}: rank must be a whole number >= 0, got {r}")
    total = sum(r for r in ranks.values() if isinstance(r, int))
    if total != RANK_SUM:
        problems.append(f"ranks sum to {total}, budget is {RANK_SUM} ({num_params(ranks):,} vs {PARAM_BUDGET:,} params)")
    return problems


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        ranks = json.load(f)
    problems = check(ranks)
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"OK: fits the budget ({PARAM_BUDGET:,} params)")
