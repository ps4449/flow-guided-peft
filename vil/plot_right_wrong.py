"""Overlay the information-flow curves of questions LLaVA answers correctly vs wrongly.

Reads the knockout CSVs of output/information_flow/<model>/<dataset>/{train_correct,train_wrong}/<flow>/
(made by InformationFlow.py on datasets/GQA_train_*_All.csv or datasets/SQA_train_*_ScienceQA.csv) and draws,
per flow, both mean curves with 95% bootstrap bands. Layers where the right-wrong gap is significant
(95% bootstrap interval of the gap excludes 0, and the gap is at least 0.5 points) are shaded. Also prints the per-layer table.

Saves output/information_flow/<model>/<dataset>/right_vs_wrong_<flow>.pdf (and .png), plus
right_vs_wrong.pdf with all flows side by side.

Usage (needs matplotlib):
  python3 vil/plot_right_wrong.py                      # GQA (All)
  python3 vil/plot_right_wrong.py --dataset ScienceQA
"""
import argparse
import csv
import glob
import os
import random
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLOWS = {"Image___Question": "Image → Question", "Question___Last": "Question → Last", "Image___Last": "Image → Last"}
COLORS = {"correct": "#1f77b4", "wrong": "#d62728"}
MIN_GAP = 0.5  # percentage points; smaller gaps are not shaded even when the bootstrap interval excludes 0


def load(path):
    """{question_id: {layer: % change in answer probability}}"""
    by_question = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_question[row["question_id"]][int(row["layer"])] = float(row["relative diff first"])
    return by_question


def mean(values):
    return sum(values) / len(values)


def curve_stats(right, wrong, layers, n_boot, seed):
    """Per layer: (mean, low, high) for right and wrong, and (gap, low, high) for wrong - right."""
    rng = random.Random(seed)
    right_ids, wrong_ids = list(right), list(wrong)
    stats = []
    for layer in layers:
        r = [right[q][layer] for q in right_ids]
        w = [wrong[q][layer] for q in wrong_ids]
        boot_r, boot_w, boot_gap = [], [], []
        for _ in range(n_boot):
            br, bw = mean(rng.choices(r, k=len(r))), mean(rng.choices(w, k=len(w)))
            boot_r.append(br), boot_w.append(bw), boot_gap.append(bw - br)
        lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
        boot_r.sort(), boot_w.sort(), boot_gap.sort()
        stats.append({"right": (mean(r), boot_r[lo], boot_r[hi]), "wrong": (mean(w), boot_w[lo], boot_w[hi]),
                      "gap": (mean(w) - mean(r), boot_gap[lo], boot_gap[hi])})
    return stats


def draw(ax, flow, stats, layers, n_right, n_wrong):
    significant = [layer for layer, s in zip(layers, stats)
                   if (s["gap"][1] > 0 or s["gap"][2] < 0) and abs(s["gap"][0]) >= MIN_GAP]
    for layer in significant:
        ax.axvspan(layer - 0.5, layer + 0.5, color="#f2c94c", alpha=0.25, lw=0)
    for key, label, n in [("right", "answered correctly", n_right), ("wrong", "answered wrongly", n_wrong)]:
        color = COLORS["correct" if key == "right" else "wrong"]
        ax.plot(layers, [s[key][0] for s in stats], color=color, lw=2, label=f"{label} (n={n})")
        ax.fill_between(layers, [s[key][1] for s in stats], [s[key][2] for s in stats], color=color, alpha=0.18, lw=0)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(FLOWS.get(flow, flow))
    ax.set_xlabel("Layer (centre of the 9-layer knockout window)")
    ax.set_ylabel("Change in answer probability (%)")
    ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if significant:
        handles.append(plt.Rectangle((0, 0), 1, 1, color="#f2c94c", alpha=0.4))
        labels.append("significant gap (95% bootstrap, ≥0.5 pts)")
    ax.legend(handles, labels, loc="lower right", fontsize=8)
    return significant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-dir", default="output/information_flow/llava_v1_5_7b")
    parser.add_argument("--dataset", default="All", help="All (GQA) or ScienceQA")
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    base = os.path.join(args.flow_dir, args.dataset)
    title = "GQA" if args.dataset == "All" else args.dataset

    found = []
    for flow in FLOWS:
        paths = [glob.glob(os.path.join(base, split, flow, f"*_window{args.window}_{flow}.csv"))
                 for split in ("train_correct", "train_wrong")]
        if all(len(p) == 1 for p in paths):
            found.append((flow, load(paths[0][0]), load(paths[1][0])))
    if not found:
        raise SystemExit(f"No flow with both train_correct and train_wrong CSVs in {base}")

    fig_all, axes = plt.subplots(1, len(found), figsize=(6.5 * len(found), 4.5), squeeze=False)
    for ax_all, (flow, right, wrong) in zip(axes[0], found):
        layers = sorted(next(iter(right.values())))
        stats = curve_stats(right, wrong, layers, args.n_boot, args.seed)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        significant = draw(ax, flow, stats, layers, len(right), len(wrong))
        ax.set_title(f"{title}: {FLOWS.get(flow, flow)}, correct vs wrong answers")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(base, f"right_vs_wrong_{flow}.{ext}"), dpi=200)
        plt.close(fig)
        draw(ax_all, flow, stats, layers, len(right), len(wrong))

        print(f"\n{title} {FLOWS.get(flow, flow)}: correct n={len(right)}, wrong n={len(wrong)}")
        print("layer  correct   wrong  gap(w-c)   95% CI of gap")
        for layer, s in zip(layers, stats):
            mark = " *" if layer in significant else ""
            print(f"{layer:5d} {s['right'][0]:8.1f} {s['wrong'][0]:7.1f} {s['gap'][0]:8.1f}   "
                  f"[{s['gap'][1]:5.1f}, {s['gap'][2]:5.1f}]{mark}")

    fig_all.suptitle(f"{title}: information flow on correct vs wrong answers (LLaVA-1.5-7B)")
    fig_all.tight_layout()
    for ext in ("pdf", "png"):
        fig_all.savefig(os.path.join(base, f"right_vs_wrong.{ext}"), dpi=200)
    print(f"\nSaved plots to {base}/right_vs_wrong*.pdf/.png")


if __name__ == "__main__":
    main()
