#!/usr/bin/env python3
"""Analyze the frozen matched window-3/window-9 ScienceQA measurements.

Paired question bootstraps retain dependence across layers, flows, and windows.
Only a meaningfully different, stable top-16 gets a new rank-8 configuration.
No evaluation on the test set and no training/job submissions occur here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output/information_flow/llava_v1_5_7b/ScienceQA/window_sensitivity/train_correct_n200_seed0/resume_20261028"
FLOWS = ["Image___Question", "Question___Last"]


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def top(scores, k):
    return sorted(np.lexsort((np.arange(32), -scores))[:k].tolist())


def overlap(a, b):
    shared = sorted(set(a) & set(b))
    return {"overlap_count": len(shared), "overlap_fraction": len(shared) / len(a),
            "jaccard": len(shared) / len(set(a) | set(b)), "shared": shared,
            "removed_from_first": sorted(set(a) - set(b)), "added_in_second": sorted(set(b) - set(a))}


def scores_from(cube):
    # cube: [flow, question, layer]. Average each flow FIRST, then clip drops.
    return np.maximum(0, -cube.mean(axis=1)).sum(axis=0)


def load_cube(out, manifest, window):
    ids = manifest["question_ids"]
    questions = {r["question_id"]: r for r in read_csv(out / "calibration_questions.csv")}
    cube = np.zeros((2, len(ids), 32))
    for f, flow in enumerate(FLOWS):
        rows = read_csv(out / f"window{window}" / f"{flow}.csv")
        by_key = {(r["question_id"], int(r["layer"])): r for r in rows}
        expected = {(q, layer) for q in ids for layer in range(32)}
        if len(rows) != len(expected) or set(by_key) != expected:
            raise ValueError(f"window{window}/{flow} is incomplete or duplicated: {len(rows)} rows; expected {len(expected)}")
        for i, qid in enumerate(ids):
            for layer in range(32):
                row = by_key[qid, layer]
                question = questions[qid]
                if not (row["goden answer"] == row["predicted answer"] == question["answer"]
                        and row["question"] == question["question"]
                        and row["image"] == question["img_id"]
                        and row["block_desc"] == flow.replace("___", "->")):
                    raise ValueError(f"Mismatched example metadata: window{window}/{flow}/{qid}/{layer}")
                base, new = float(row["base_score_first"]), float(row["new_score_first"])
                if not (0 < base <= 1 and 0 <= new <= 1):
                    raise ValueError(f"Invalid probability: window{window}/{flow}/{qid}/{layer}")
                cube[f, i, layer] = float(row["relative diff first"])
                if not math.isclose(cube[f, i, layer], (new - base) * 100 / base,
                                    rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"Relative-change arithmetic mismatch: {qid}/{layer}")
        if not np.isfinite(cube[f]).all():
            raise ValueError("Non-finite measurement")
    return cube


def validate_parity(out, manifest):
    """Require complete, numerically valid evidence, including resumed jobs."""
    report = {}
    for flow in FLOWS:
        run = json.loads((out / "window3" / f"{flow}_run.json").read_text())
        if run["state"] != "complete" or run["completed_questions"] != manifest["n"]:
            raise ValueError(f"Measurement run is incomplete: {flow}")
        checks = json.loads((out / "window3" / f"{flow}_parity.json").read_text())
        if (len(checks) != manifest["n"] or
                {c["question_id"] for c in checks} != set(manifest["question_ids"])):
            raise ValueError(f"Missing or duplicate parity records: {flow}")
        reference = {(r["question_id"], int(r["layer"])): r
                     for r in read_csv(out / "window9" / f"{flow}.csv")}
        measured = read_csv(out / "window3" / f"{flow}.csv")
        current_bases = {r["question_id"]: float(r["base_score_first"]) for r in measured}
        if any(float(r["base_score_first"]) != current_bases[r["question_id"]] for r in measured):
            raise ValueError(f"Inconsistent baseline across centers: {flow}")
        repeated = []
        for c in checks:
            qid = c["question_id"]
            saved = float(reference[qid, 0]["base_score_first"])
            if not (c["base_saved"] == saved and c["base_current"] == current_bases[qid]
                    and math.isclose(c["base_current"], saved, rel_tol=.001, abs_tol=.00001)):
                raise ValueError(f"Baseline parity failed: {flow}/{qid}")
            for check in c["window9_checks"]:
                saved = float(reference[qid, check["center"]]["new_score_first"])
                if not (check["saved"] == saved and
                        math.isclose(check["current"], saved, rel_tol=.001, abs_tol=.00001)):
                    raise ValueError(f"Window9 parity failed: {flow}/{qid}/{check['center']}")
                repeated.append(check)
        if len(repeated) < 6:
            raise ValueError(f"Insufficient window9 parity checks: {flow}")
        report[flow] = {"complete_questions": len(checks), "window9_repeated_checks": len(repeated),
                        "maximum_base_absolute_difference": max(c["base_abs_diff"] for c in checks),
                        "maximum_window9_absolute_difference": max(c["abs_diff"] for c in repeated),
                        "final_job_id": run["slurm_job_id"], "actual_model_dtype": run["actual_model_dtype"],
                        "actual_attention_implementation": run["actual_attention_implementation"]}
    return report


def bootstrap(cubes, manifest):
    criteria = manifest["criteria"]
    rng = np.random.default_rng(criteria["bootstrap_seed"])
    score = {w: scores_from(c) for w, c in cubes.items()}
    reference = {w: {k: top(v, k) for k in (8, 16)} for w, v in score.items()}
    samples = {w: {k: [] for k in (8, 16)} for w in cubes}
    inclusion = {w: {k: np.zeros(32) for k in (8, 16)} for w in cubes}
    rhos = []
    cross_overlaps = {k: [] for k in (8, 16)}
    n = manifest["n"]
    for _ in range(criteria["bootstrap_replicates"]):
        indices = rng.integers(0, n, size=n)
        draw = {w: scores_from(c[:, indices, :]) for w, c in cubes.items()}
        for w, importance in draw.items():
            for k in (8, 16):
                chosen = top(importance, k)
                samples[w][k].append(len(set(chosen) & set(reference[w][k])))
                inclusion[w][k][chosen] += 1
        if len(draw) == 2:
            for k in (8, 16):
                cross_overlaps[k].append(len(set(top(draw[9], k)) & set(top(draw[3], k))))
            rho = float(spearmanr(draw[9], draw[3])[0])
            if np.isfinite(rho):
                rhos.append(rho)
    result = {}
    for w in cubes:
        result[w] = {}
        for k in (8, 16):
            values = np.array(samples[w][k])
            halves = [top(scores_from(cubes[w][:, part, :]), k) for part in (slice(0, n // 2), slice(n // 2, n))]
            result[w][k] = {"mean_overlap": float(values.mean()), "median_overlap": float(np.median(values)),
                           "fifth_percentile_overlap": float(np.percentile(values, 5)),
                           "ninety_fifth_percentile_overlap": float(np.percentile(values, 95)),
                           "overlap_histogram": dict(sorted(Counter(int(x) for x in values).items())),
                           "selection_frequency_by_layer": (inclusion[w][k] / criteria["bootstrap_replicates"]).tolist(),
                           "half_selections": halves,
                           "half_overlaps_with_full": [len(set(s) & set(reference[w][k])) for s in halves],
                           "half_to_half": overlap(*halves)}
    cross = {k: {"median_overlap": float(np.median(v)),
                 "ci95_percentile_overlap": np.percentile(v, [2.5, 97.5]).tolist(),
                 "overlap_histogram": dict(sorted(Counter(v).items()))}
             for k, v in cross_overlaps.items() if v}
    return result, ({"median": float(np.median(rhos)), "ci95_percentile": np.percentile(rhos, [2.5, 97.5]).tolist()}
                    if rhos else None), cross


def plots(out, cubes, importance, stability):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {9: "#2563a6", 3: "#db7926"}
    x = np.arange(32)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for w in (9, 3):
        values = importance[w]
        axes[0].plot(x, values / values.sum(), marker="o", ms=3, color=colors[w], label=f"window {w}")
        axes[1].plot(x, rankdata(-values, method="average"), marker="o", ms=3, color=colors[w], label=f"window {w}")
    axes[0].set_ylabel("Within-window share of total importance")
    axes[0].set_title("Matched 200 ScienceQA training-calibration questions")
    axes[1].set_ylabel("Importance rank (1 = highest)")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
        ax.set_xticks(x)
        ax.set_xlim(-.4, 31.4)
    axes[1].set_xlabel("Knockout center layer")
    fig.text(.5, .012, "Different intervention widths: normalized profiles/ranks compare allocation; raw knockout magnitudes are not equivalent.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"importance_overlay.{suffix}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, k in zip(axes, (8, 16)):
        for w in (9, 3):
            ax.plot(x, stability[w][k]["selection_frequency_by_layer"], marker="o", color=colors[w], label=f"window {w}")
        ax.axvspan(-.4, 15.4, alpha=.07, color="black", label="Historical selected layers 0–15")
        ax.set_xticks(x)
        ax.set_ylim(-.03, 1.03)
        ax.set_ylabel(f"Top-{k} inclusion frequency")
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_title("Paired question-bootstrap selection stability (2,000 resamples)")
    axes[1].set_xlabel("Knockout center layer")
    fig.tight_layout()
    fig.savefig(out / "selection_stability.png", dpi=200)
    fig.savefig(out / "selection_stability.pdf")
    plt.close(fig)

    # Raw effects are shown in separate width columns, with independent y axes.
    # This makes the actual measurements available without implying equal-sized
    # causal interventions or interpreting their magnitude ratio as importance.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, w in enumerate((9, 3)):
        for f, flow in enumerate(FLOWS):
            ax = axes[f, col]
            ax.plot(x, cubes[w][f].mean(axis=0), color=colors[w])
            ax.axhline(0, color="gray", lw=.7)
            ax.set_title(f"{flow.replace('___', ' → ')}: window {w}")
            ax.set_ylabel("Mean raw relative probability change (%)")
            ax.set_xticks(range(0, 32, 2))
            ax.grid(alpha=.2)
    fig.text(.5, .01, "Independent y scales. Raw effects describe different intervention widths and are not directly equivalent.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out / "signed_flow_measurements.png", dpi=200)
    fig.savefig(out / "signed_flow_measurements.pdf")
    plt.close(fig)


def write_report(out, summary):
    scores, stability = summary["windows"], summary["stability"]
    comparison, decision = summary["comparison"], summary["decision"]
    indices = lambda x: ", ".join(map(str, x))
    lines = ["# ScienceQA information-flow window sensitivity", "",
             "Completed matched analysis of 200 frozen, correctly answered training-calibration questions.",
             "Window 9 reuses existing measurements; window 3 measures both flows on the same inputs.",
             "Each width has 6,400 rows per flow. No training was submitted.", "",
             "## Boundary finding", "",
             "Window-9 center 4 blocks layers 0–8; center 5 blocks 1–9. Both contain nine layers.",
             "The transition removes layer 0 and adds layer 9. Window-3 centers 1 and 2 similarly",
             "change from 0–2 to 1–3. Boundary windows are clipped, not shifted to constant width.",
             "The Image→Question cliff moves to centers 1→2 under window 3. This supports the",
             "boundary explanation, but overlapping nonlinear interventions cannot isolate layer 0's causal effect.", "",
             "## Selections and ranking", "",
             "Importance is max(0, −mean Image→Question relative change) +",
             "max(0, −mean Question→Last relative change), averaging questions before clipping.",
             "Higher scores rank first; ties select the smaller layer index.", "",
             "| Width | Top-8 indices | Top-16 indices | Top-16 overlap with 0–15 |",
             "| --- | --- | --- | --- |"]
    for w in (9, 3):
        s = scores[w]
        lines.append(f"| {w} | {indices(s['top8'])} | {indices(s['top16'])} | {s['top16_vs_historical_0_15']['overlap_count']}/16 |")
    lines += ["", f"Spearman correlation over all 32 scores: **{comparison['spearman']:.6f}**.",
              f"Paired-bootstrap percentile interval: {comparison['spearman_bootstrap']['ci95_percentile']}.", "",
              "| Selection | Cross-window overlap | Jaccard |",
              "| --- | --- | --- |"]
    for k in (8, 16):
        c = comparison["selection_overlap"][k]
        lines.append(f"| Top-{k} | {c['overlap_count']}/{k} | {c['jaccard']:.6f} |")
    lines += ["", "## Stability", "",
              "2,000 paired question-bootstrap resamples (seed 0); every resample recomputes means,",
              "clipping, rankings, and selections. The two fixed 100-question halves are disjoint.", "",
              "| Width | Selection | Bootstrap median overlap | Fifth percentile | Half overlaps with full |",
              "| --- | --- | --- | --- | --- |"]
    for w in (9, 3):
        for k in (8, 16):
            s = stability[w][k]
            lines.append(f"| {w} | Top-{k} | {s['median_overlap']:g}/{k} | {s['fifth_percentile_overlap']:g}/{k} | {indices(s['half_overlaps_with_full'])} |")
    lines += ["", "## Decision", "",
              "The frozen practical screening rules require top-16 overlap ≤13/16 with both matched",
              "window 9 and historical layers 0–15; bootstrap median overlap ≥14/16, fifth percentile",
              "≥12/16, and each half's overlap with the full selection ≥14/16. These are screening",
              "rules, not significance tests.", "",
              f"Meaningfully different: **{decision['meaningfully_different']}**. Stable: **{decision['stable']}**.",
              f"Prepare rank-8 configuration: **{decision['prepare_rank8_config']}**."]
    if decision["prepare_rank8_config"]:
        lines += ["", f"Prepared `{decision['rank_config']}`; actual adapter tensor count:",
                  f"**{decision['actual_trainable_parameters']:,}** trainable parameters. All 32 base layers remain."]
    else:
        lines += ["", "No new LoRA configuration is prepared because the candidate fails the frozen screening rules."]
    lines += ["", "## Outputs and limits", "",
              "- [All 32 importance scores](importance_scores.csv)",
              "- [Normalized importance and rank overlay](importance_overlay.png)",
              "- [Top-8/top-16 bootstrap inclusion frequencies](selection_stability.png)",
              "- [Signed flow measurements, independent axes](signed_flow_measurements.png)",
              "- [Full comparison and per-layer bootstrap frequencies](comparison.json)",
              "- [Actual submitted jobs](jobs.json)", "",
              "**Raw knockout magnitudes across window sizes are not directly equivalent.**",
              "Normalized profiles and ranks compare allocation patterns, not equal-sized causal interventions.",
              "Scores refer to windows centered at layers, not isolated-layer effects. Bootstrap stability",
              "describes this calibration cohort and does not establish external generalization.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--verify-environment", action="store_true", help="Also require/recheck external LLaVA code and image files")
    parser.add_argument("--window9-only", action="store_true", help="Report existing matched window9 scores/stability while measurements run")
    args = parser.parse_args()
    out = args.output_dir
    manifest = json.loads((out / "subset_manifest.json").read_text())
    external_environment_records = []
    for item in [manifest["calibration_source"], *manifest["window9_sources"].values(),
                 *manifest["measurement_code"]]:
        if item["path"].startswith("LLaVA-NeXT/") and not args.verify_environment:
            # External dependency is not needed to rescore saved measurements.
            # Preserve its original fingerprint; do not claim to revalidate it.
            external_environment_records.append({**item, "status": "external dependency not revalidated in offline analysis"})
            continue
        if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source artifact changed: {item['path']}")
    provenance = json.loads((out / "input_provenance.json").read_text())
    frozen_out = (ROOT / provenance["files"][0]["path"]).parent
    for item in provenance["files"]:
        if item["path"].startswith("datasets/scienceqa/images/") and not args.verify_environment:
            external_environment_records.append({**item, "status": "image not revalidated in offline analysis"})
            continue
        original = ROOT / item["path"]
        path = out / original.relative_to(frozen_out) if original.is_relative_to(frozen_out) else original
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Frozen matched input changed: {path}")
    cubes = {9: load_cube(out, manifest, 9)}
    parity = None
    if not args.window9_only:
        cubes[3] = load_cube(out, manifest, 3)
        parity = validate_parity(out, manifest)
    importance = {w: scores_from(c) for w, c in cubes.items()}
    stability, rho_ci, cross_bootstrap = bootstrap(cubes, manifest)
    scores = {w: {"importance": v.tolist(), "signed_flow_means": cubes[w].mean(axis=1).tolist(),
                  "top8": top(v, 8), "top16": top(v, 16),
                  "positive_score_layers": np.flatnonzero(v > 0).tolist(),
                  "cutoff_tie_counts": {k: int(np.sum(v == v[top(v, k)].min())) for k in (8, 16)},
                  "top16_vs_historical_0_15": overlap(list(range(16)), top(v, 16))} for w, v in importance.items()}
    summary = {"n": manifest["n"], "windows": scores, "stability": stability,
               "bootstrap": {"replicates": manifest["criteria"]["bootstrap_replicates"], "seed": manifest["criteria"]["bootstrap_seed"],
                             "sampling_unit": "question; paired indices across both flows and both widths; clip only after each resampled mean"},
               "external_environment_records": external_environment_records,
               "training_submitted": False, "criteria": manifest["criteria"], "parity_validation": parity,
               "caveat": "Raw knockout magnitudes across widths are not directly equivalent. Scores refer to overlapping windows centered at each layer, not isolated-layer causal effects. Bootstrap measures sensitivity within this 200-question calibration subset, not external generalization."}
    if args.window9_only:
        save_json(out / "window9_analysis.json", summary)
        print(json.dumps({"window9_top8": scores[9]["top8"], "window9_top16": scores[9]["top16"], "stability16": stability[9][16]}, indent=2))
        return
    comparisons = {k: overlap(scores[9][f"top{k}"], scores[3][f"top{k}"]) for k in (8, 16)}
    summary["comparison"] = {"spearman": float(spearmanr(importance[9], importance[3])[0]),
                             "spearman_bootstrap": rho_ci, "selection_overlap": comparisons,
                             "paired_bootstrap_cross_window_overlap": cross_bootstrap}
    criteria = manifest["criteria"]
    difference = comparisons[16]["overlap_count"] <= criteria["maximum_top16_overlap_for_difference"] and scores[3]["top16_vs_historical_0_15"]["overlap_count"] <= criteria["maximum_top16_overlap_for_difference"]
    s = stability[3][16]
    stable = (s["median_overlap"] >= criteria["minimum_bootstrap_median_overlap"] and
              s["fifth_percentile_overlap"] >= criteria["minimum_bootstrap_fifth_percentile_overlap"] and
              min(s["half_overlaps_with_full"]) >= criteria["minimum_half_overlap"])
    summary["decision"] = {"meaningfully_different": difference, "stable": stable,
                           "prepare_rank8_config": difference and stable,
                           "reason": "prepare_different_stable_candidate" if difference and stable else "no_candidate_unless_both_difference_and_stability_pass"}
    if difference and stable:
        rank_path = ROOT / "configs/ranks/flow_window3_pruned16_scienceqa.json"
        selected = scores[3]["top16"]
        spec = {f"layers.{i}.self_attn.{projection}": (8 if i in selected else 0)
                for i in range(32) for projection in ("q_proj", "v_proj")}
        if rank_path.exists() and json.loads(rank_path.read_text()) != spec:
            raise ValueError("Refusing to replace a different existing rank config")
        save_json(rank_path, spec)
        module_path = ROOT / "peft_lora/methods/region_guided/make_ranks.py"
        module_spec = importlib.util.spec_from_file_location("region_parameter_validation", module_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.validate_adapter(rank_path, out / "adapter_validation.json")
        summary["decision"].update(rank_config=str(rank_path.relative_to(ROOT)), actual_trainable_parameters=2097152)
    plots(out, cubes, importance, stability)
    with (out / "importance_scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "window9_importance", "window3_importance", "window9_rank", "window3_rank"])
        r9, r3 = rankdata(-importance[9]), rankdata(-importance[3])
        writer.writerows([i, importance[9][i], importance[3][i], r9[i], r3[i]] for i in range(32))
    save_json(out / "comparison.json", summary)
    write_report(out, summary)
    save_json(out / "status.json", {"state": "analysis_complete", "window9": "reused_matched_200", "window3": "measured_200_each_flow", "training": "not_submitted", "decision": summary["decision"]})
    print(json.dumps({"top8_w9": scores[9]["top8"], "top8_w3": scores[3]["top8"], "top16_w9": scores[9]["top16"], "top16_w3": scores[3]["top16"], "comparison": summary["comparison"], "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()
