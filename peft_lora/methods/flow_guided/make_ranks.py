"""Flow-guided rank allocation: more LoRA rank in the layers where cross-modal information flows.

For each flow (Image->Question, Question->Last, Image->Last), the knockout experiments give the
% change in answer probability when attention is blocked around each layer (9-layer window).
For each question type, a layer's importance is the average drop summed over the flows
(increases count as 0). The importance is then averaged over the question types (equal weight),
since GQA training mixes all types. Ranks are split in proportion to importance, between
--min-rank and --max-rank, and sum to the shared budget (peft_lora/budget.py).
q_proj and v_proj of a layer get the same rank.

Usage (from the repo root):
  python peft_lora/methods/flow_guided/make_ranks.py                 # all question types -> flow_guided.json
  python peft_lora/methods/flow_guided/make_ranks.py --types ChooseAttr --out configs/ranks/flow_guided_ChooseAttr.json
  python peft_lora/methods/flow_guided/make_ranks.py --types All --split train_wrong --out configs/ranks/flow_wrong.json
  python peft_lora/methods/flow_guided/make_ranks.py --inverse --out configs/ranks/inverse_flow.json   # control
Prints the rank of every layer.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import budget

FLOWS = ["Image___Question", "Question___Last", "Image___Last"]


def flow_curve(path):
    """Mean % change in answer probability per layer, over all questions in the CSV."""
    changes = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            changes[int(row["layer"])].append(float(row["relative diff first"]))
    return [sum(changes[layer]) / len(changes[layer]) for layer in range(budget.NUM_LAYERS)]


def layer_importance(curves):
    """Sum of the probability drops over the flows; blocking that helps the answer counts as 0."""
    return [sum(max(0.0, -curve[layer]) for curve in curves) for layer in range(budget.NUM_LAYERS)]


def allocate(weights, total, min_rank, max_rank):
    """Whole-number ranks proportional to weights, each in [min_rank, max_rank], summing to total."""
    n = len(weights)
    if not min_rank * n <= total <= max_rank * n:
        sys.exit(f"Cannot split {total} over {n} layers within [{min_rank}, {max_rank}]")

    # give everyone min_rank, then share the rest by weight; layers that hit max_rank are fixed
    # and the remainder is shared again among the others
    ranks = [float(min_rank)] * n
    remaining = total - min_rank * n
    free = set(range(n))
    while remaining > 1e-9 and free:
        weight_sum = sum(weights[i] for i in free)
        share = {i: remaining * (weights[i] / weight_sum if weight_sum > 0 else 1 / len(free)) for i in free}
        capped = [i for i in free if ranks[i] + share[i] >= max_rank]
        if not capped:
            for i in free:
                ranks[i] += share[i]
            break
        for i in capped:
            remaining -= max_rank - ranks[i]
            ranks[i] = float(max_rank)
            free.remove(i)

    # round down, then hand the leftover units to the largest fractional parts
    whole = [int(r) for r in ranks]
    by_fraction = sorted(range(n), key=lambda i: ranks[i] - whole[i], reverse=True)
    for i in by_fraction[: total - sum(whole)]:
        whole[i] += 1
    return whole


def find_csv(flow_dir, qtype, split, flow, window):
    """The CSV of one question type, split and flow (output/information_flow/<model>/<type>/<split>/<flow>/)."""
    paths = glob.glob(os.path.join(flow_dir, qtype, split, flow, f"*_window{window}_{flow}.csv"))
    if len(paths) != 1:
        sys.exit(f"Expected 1 CSV for {qtype} {split} {flow}, found {len(paths)}")
    return paths[0]


def average(rows):
    return [sum(values) / len(values) for values in zip(*rows)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-dir", default="output/information_flow/llava_v1_5_7b")
    parser.add_argument("--types", nargs="+", default=None,
                        help="question types (default: every folder in --flow-dir that has --split)")
    parser.add_argument("--split", default="val_correct", help="val_correct (paper CSVs), train_correct or train_wrong")
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--flows", nargs="+", default=FLOWS)
    parser.add_argument("--min-rank", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=32)
    parser.add_argument("--out", default="configs/ranks/flow_guided.json")
    parser.add_argument("--inverse", action="store_true",
                        help="same ranks, reversed: the least important layer gets the largest rank (control)")
    args = parser.parse_args()

    types = args.types or sorted(d for d in os.listdir(args.flow_dir) if os.path.isdir(os.path.join(args.flow_dir, d, args.split)))
    type_curves = {t: [flow_curve(find_csv(args.flow_dir, t, args.split, flow, args.window)) for flow in args.flows]
                   for t in types}
    importance = average([layer_importance(curves) for curves in type_curves.values()])
    curves = [average([type_curves[t][i] for t in types]) for i in range(len(args.flows))]  # for the printout
    print(f"Question types: {', '.join(types)}")
    per_layer_total = budget.RANK_SUM // len(budget.TARGET_MODULES)  # same rank for q_proj and v_proj
    layer_ranks = allocate(importance, per_layer_total, args.min_rank, args.max_rank)
    if args.inverse:  # hand the largest ranks to the least important layers; same ranks, same budget
        least_first = sorted(range(budget.NUM_LAYERS), key=lambda layer: (importance[layer], layer))
        largest_first = sorted(layer_ranks, reverse=True)
        layer_ranks = [0] * budget.NUM_LAYERS
        for layer, rank in zip(least_first, largest_first):
            layer_ranks[layer] = rank

    ranks = {f"layers.{layer}.self_attn.{m}": layer_ranks[layer]
             for layer in range(budget.NUM_LAYERS) for m in budget.TARGET_MODULES}
    problems = budget.check(ranks)
    if problems:
        sys.exit("\n".join(problems))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(ranks, f, indent=2)

    print("layer  " + "  ".join(f"{flow:>16}" for flow in args.flows) + "  importance  rank")
    for layer in range(budget.NUM_LAYERS):
        drops = "  ".join(f"{curve[layer]:16.1f}" for curve in curves)
        print(f"{layer:5d}  {drops}  {importance[layer]:10.1f}  {layer_ranks[layer]:4d}")
    print(f"Wrote {args.out}: ranks sum to {sum(ranks.values())}, {budget.num_params(ranks):,} params")


if __name__ == "__main__":
    main()
