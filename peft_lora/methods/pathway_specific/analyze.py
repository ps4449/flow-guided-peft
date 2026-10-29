#!/usr/bin/env python3
"""CPU-only analysis of full-cohort ScienceQA pathway-specific top-8 selection.

Calls the original flow_curve/layer_importance functions; never loads a model,
writes a rank configuration, submits a job, or replaces an output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "output/pathway_specific/scienceqa/train_correct_window9_n989_bootstrap5000_seed0_20261029"
FLOW_NAMES = ("Image___Question", "Question___Last")
NAMES = ("image_question", "question_last", "combined")
LABELS = {"image_question": "Image→Question", "question_last": "Question→Last",
          "combined": "Combined", "early_eight": "Early control (0–7)",
          "middle_eight": "Middle control (12–19)"}
CONTROLS = {"early_eight": list(range(8)), "middle_eight": list(range(12, 20))}
sys.dont_write_bytecode = True


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def ordered(score):
    return sorted(range(32), key=lambda layer: (-score[layer], layer))


def overlap(first, second):
    a, b = set(first), set(second)
    return {"count": len(a & b), "fraction": len(a & b) / 8,
            "jaccard": len(a & b) / len(a | b), "shared": sorted(a & b),
            "only_first": sorted(a - b), "only_second": sorted(b - a)}


def distribution(values):
    counts = Counter(int(x) for x in values)
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "fifth_percentile": float(np.percentile(values, 5)),
            "ninety_fifth_percentile": float(np.percentile(values, 95)),
            "ci95_percentile": np.percentile(values, [2.5, 97.5]).tolist(),
            "counts": {i: counts[i] for i in range(9)},
            "probabilities": {i: counts[i] / len(values) for i in range(9)}}


def load_original():
    sys.path.insert(0, str(ROOT / "peft_lora"))
    path = ROOT / "peft_lora/methods/flow_guided/make_ranks.py"
    spec = importlib.util.spec_from_file_location("original_flow_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data(original):
    source = ROOT / "output/information_flow/llava_v1_5_7b"
    paths = [Path(original.find_csv(str(source), "ScienceQA", "train_correct", flow, 9))
             for flow in FLOW_NAMES]
    calibration_path = ROOT / "datasets/SQA_train_correct_ScienceQA.csv"
    with calibration_path.open(newline="") as handle:
        calibration = {r["question_id"]: r for r in csv.DictReader(handle)}
    groups = []
    for flow, path in zip(FLOW_NAMES, paths):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_key = {(r["question_id"], int(r["layer"])): r for r in rows}
        ids = sorted({r["question_id"] for r in rows})
        expected = {(q, layer) for q in ids for layer in range(32)}
        if len(ids) != 989 or len(rows) != 989 * 32 or set(by_key) != expected:
            raise ValueError(f"Require exactly 989 complete unique questions × 32 centers: {path}")
        for (qid, layer), row in by_key.items():
            q = calibration[qid]
            if not (qid.startswith("train_") and row["goden answer"] == row["predicted answer"] == q["answer"]
                    and row["question"] == q["question"] and row["image"] == q["img_id"]
                    and row["block_desc"] == flow.replace("___", "->")):
                raise ValueError(f"Calibration metadata/correctness mismatch: {flow}/{qid}/{layer}")
            base, new, change = (float(row[k]) for k in ("base_score_first", "new_score_first", "relative diff first"))
            if not (0 < base <= 1 and 0 <= new <= 1 and
                    np.isclose(change, (new - base) * 100 / base, rtol=1e-10, atol=1e-10)):
                raise ValueError(f"Invalid probability or relative-change arithmetic: {flow}/{qid}/{layer}")
        groups.append((ids, by_key))
    if groups[0][0] != groups[1][0]:
        raise ValueError("Flow CSVs do not contain the same 989 questions")
    ids = groups[0][0]
    cube = np.array([[[float(rows[qid, layer]["relative diff first"]) for layer in range(32)]
                      for qid in ids] for _, rows in groups])
    if not np.isfinite(cube).all():
        raise ValueError("Non-finite observation")
    means = [original.flow_curve(str(p)) for p in paths]
    np.testing.assert_allclose(cube.mean(axis=1), means, rtol=1e-12, atol=1e-12)
    return ids, cube, means, paths, calibration_path


def score_means(original, means):
    return {"image_question": np.array(original.layer_importance([means[0]])),
            "question_last": np.array(original.layer_importance([means[1]])),
            "combined": np.array(original.layer_importance(means))}


def plot_results(out, scores, selections, stability):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = {"image_question": "#2563a6", "question_last": "#ce6a18", "combined": "#268264"}
    layers = np.arange(32)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for name in NAMES:
        axes[0].plot(layers, scores[name], marker="o", ms=3, color=colors[name], label=LABELS[name])
        axes[1].plot(layers, scores[name] / scores[name].sum(), marker="o", ms=3,
                     color=colors[name], label=LABELS[name])
    axes[0].set_ylabel("Importance (relative-change percentage points)")
    axes[0].set_title("ScienceQA train_correct: all 989 questions, window 9")
    axes[1].set_ylabel("Share of each ranking's total importance")
    axes[1].set_xlabel("Knockout window center / selected layer index")
    for ax in axes:
        ax.set_xticks(layers)
        ax.grid(alpha=.2)
        ax.legend()
    fig.text(.5, .012, "Overlapping window effects, not isolated-layer effects. Different pathway objectives do not establish downstream quality.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"importance_overlay.{suffix}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4))
    names = list(selections)
    matrix = np.array([[int(i in selections[name]) for i in layers] for name in names])
    ax.imshow(matrix, cmap=ListedColormap(["#f1f3f5", "#268264"]), vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    ax.set_xticks(layers)
    ax.set_xticks(np.arange(-.5, 32, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Layer index (green = selected; eight per row)")
    ax.set_title("Pathway rankings, existing combined top-8, and fixed position controls")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"layer_selection_comparison.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for j, name in enumerate(NAMES):
        s = stability[name]
        axes[0].plot(layers, s["inclusion_frequency"], marker="o", ms=3, color=colors[name], label=LABELS[name])
        axes[1].bar(np.arange(9) + (j - 1) * .25,
                    list(s["overlap_with_full_top8"]["probabilities"].values()), width=.25,
                    color=colors[name], label=LABELS[name])
    axes[0].set_xticks(layers)
    axes[0].set_ylim(-.03, 1.03)
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Bootstrap top-8 inclusion frequency")
    axes[0].set_title("5,000 paired question-bootstrap resamples, seed 0")
    axes[1].set_xticks(range(9))
    axes[1].set_xlabel("Overlap with that ranking's full-cohort top-8 (out of 8)")
    axes[1].set_ylabel("Bootstrap probability")
    for ax in axes:
        ax.grid(alpha=.15)
        ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"bootstrap_stability.{suffix}", dpi=180)
    plt.close(fig)


def write_report(out, result):
    fmt = lambda values: ", ".join(map(str, values))
    lines = ["# ScienceQA pathway-specific sparse selection", "",
             "Full train_correct window-9 data: **989 matched correctly answered questions per layer**,",
             "32 centers and 31,648 observations per pathway. No new inference, model evaluation, training,",
             "rank configuration, or GPU job was run. Existing outputs/configurations are unchanged.", "",
             "## Scoring and selected layers", "",
             "The unchanged `peft_lora/methods/flow_guided/make_ranks.py` supplies both `flow_curve`",
             "and `layer_importance`. Per-pathway importance is max(0, −mean relative probability change).",
             "Combined importance sums the two clipped pathway means. Average questions **before** clipping.",
             "Higher scores rank first; exact ties select the lower index. All reported layer lists are sorted.", "",
             "| Ranking | Top-8 indices | Selected score sum | All-32 score sum | Overlap with existing combined | Jaccard |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name in NAMES:
        r = result["rankings"][name]
        o = r["vs_existing_combined_top8"]
        lines.append(f"| {LABELS[name]} | {fmt(r['top8_sorted'])} | {r['selected_score_sum']:.6f} | {r['all32_score_sum']:.6f} | {o['count']}/8 | {o['jaccard']:.6f} |")
    lines += ["", "Scores are sums of relative-change percentage points, not model accuracies.",
              "Each ranking has its own objective, so its score total is not evidence of pathway superiority.", "",
              "## Position controls and scores under every objective", "",
              "Early is fixed at 0–7; middle is the centered eight of 32 layers, 12–19.",
              "These controls are specified without consulting the flow scores; they are not reselected by bootstrap.", "",
              "| Allocation | Sorted indices | Image→Question sum | Question→Last sum | Combined sum | Overlap / Jaccard with combined top-8 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name, r in result["allocations"].items():
        s, o = r["score_sums"], r["vs_existing_combined_top8"]
        lines.append(f"| {LABELS[name]} | {fmt(r['layers'])} | {s['image_question']:.6f} | {s['question_last']:.6f} | {s['combined']:.6f} | {o['count']}/8; {o['jaccard']:.6f} |")
    lines += ["", "## Bootstrap selection stability", "",
              "5,000 bootstrap samples of 989 **questions**, NumPy seed 0, with the same sampled IDs across",
              "both pathways. Each draw recomputes all means, clipping, and three rankings. Layers are not",
              "independent resampling units. Two fixed hash-ordered disjoint halves contain 494/495 questions.", "",
              "| Ranking | Median own-top-8 overlap | 5th / 95th percentiles | Exact top-8 frequency | Half overlaps |",
              "| --- | --- | --- | --- | --- |"]
    for name in NAMES:
        s = result["stability"][name]
        d = s["overlap_with_full_top8"]
        lines.append(f"| {LABELS[name]} | {d['median']:g}/8 | {d['fifth_percentile']:g}, {d['ninety_fifth_percentile']:g} | {d['probabilities'][8]:.2%} | {fmt(s['half_overlaps_with_full'])} |")
    lines += ["", "Overlap counts (all 5,000 draws; zero-count bins included):", "",
              "| Shared layers | Image→Question | Question→Last | Combined |", "| --- | --- | --- | --- |"]
    for count in range(9):
        values = [result["stability"][n]["overlap_with_full_top8"]["counts"][count] for n in NAMES]
        lines.append(f"| {count} | {values[0]} | {values[1]} | {values[2]} |")
    lines += ["", "## Cutoff ambiguity", "",
              "The gap uses the full-cohort eighth and ninth layers as fixed identities in every bootstrap draw.",
              "Its percentile interval describes calibration sampling sensitivity, not an independent-layer test.", "",
              "| Ranking | Eighth / ninth layers | Full score gap | Bootstrap gap 95% interval | Inclusion frequencies |",
              "| --- | --- | --- | --- | --- |"]
    for name in NAMES:
        c = result["stability"][name]["cutoff"]
        lines.append(f"| {LABELS[name]} | {c['eighth_layer']} / {c['ninth_layer']} | {c['score_gap']:.6f} | [{c['gap_ci95'][0]:.6f}, {c['gap_ci95'][1]:.6f}] | {c['eighth_inclusion_frequency']:.2%} / {c['ninth_inclusion_frequency']:.2%} |")
    lines += ["", "Ranks 6–11, all 32 inclusion frequencies, fixed-half selections, and both fixed-baseline and",
              "paired resampled-baseline overlap distributions are saved in `analysis.json` and CSVs.", "",
              "## Interpretation", "",
              "Practical screening criteria were saved before pathway ranking/bootstrap: distinct means ≤5/8",
              "overlap with existing combined top-8 (at least three replacements); stable means bootstrap",
              "median overlap ≥7/8, fifth percentile ≥6/8, and each half overlap ≥7/8. These are analysis",
              "screening choices, not a significance test or proof of generalization.", ""]
    for name in NAMES[:2]:
        d = result["decisions"][name]
        lines.append(f"- **{LABELS[name]}:** distinct={d['distinct']}, stable={d['stable']}, worth a controlled training comparison={d['passes_training_screen']}.")
    lines += ["", "Passing this screen motivates a hypothesis test against combined selection and position controls;",
              "it does not establish that either pathway is better. Do not claim accuracy, efficiency, or",
              "generalization gains without downstream evaluation. No new LoRA model was trained or evaluated.", "",
              "The scores describe overlapping nine-layer knockouts indexed by their center, not isolated-layer",
              "causal effects. At the left boundary windows are clipped: centers 0–4 all include layer 0.",
              "Bootstrap cannot remove window smoothing or disentangle pathway from layer-position effects.", "",
              "## Artifacts and reproduction", "",
              "- [All 32 scores, ranks, and inclusion frequencies](layer_rankings.csv)",
              "- [Allocation comparison and score totals](selection_comparison.csv)",
              "- [Overlaid importance curves](importance_overlay.png)",
              "- [Layer-selection comparison](layer_selection_comparison.png)",
              "- [Bootstrap frequencies and overlap distributions](bootstrap_stability.png)",
              "- [Full results](analysis.json), [provenance](manifest.json), [preservation checks](preservation.json)", "",
              "Reproduce into a fresh directory (existing directories are refused):", "",
              "```bash", "conda activate llava", "export MPLCONFIGDIR=/tmp/pathway-specific-matplotlib",
              "export XDG_CACHE_HOME=/tmp/pathway-specific-cache", "export PYTHONDONTWRITEBYTECODE=1",
              "python peft_lora/methods/pathway_specific/analyze.py --output-dir /tmp/scienceqa-pathways-reproduction",
              "```", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise ValueError(f"Refusing to overwrite existing output directory: {out}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    original = load_original()
    ids, cube, means, source_paths, calibration_path = load_data(original)
    protected_paths = [*source_paths, calibration_path,
                       ROOT / "peft_lora/methods/flow_guided/make_ranks.py", ROOT / "peft_lora/budget.py",
                       *sorted((ROOT / "configs/ranks").glob("*.json"))]
    for previous in (ROOT / "output/region_pilot", ROOT / "output/information_flow/llava_v1_5_7b/ScienceQA/window_sensitivity"):
        protected_paths.extend(sorted(p for p in previous.rglob("*") if p.is_file()))
    protected = [record(p) for p in dict.fromkeys(protected_paths)]
    criteria = {"maximum_combined_overlap_for_distinct": 5,
                "minimum_bootstrap_median_overlap": 7, "minimum_bootstrap_fifth_percentile": 6,
                "minimum_each_half_overlap": 7,
                "interpretation": "Practical screening choices saved before pathway scores/bootstrap; not a significance test"}
    half_order = sorted(range(len(ids)), key=lambda i: (hashlib.sha256(f"scienceqa-pathway-halves-v1|0|{ids[i]}".encode()).hexdigest(), ids[i]))
    halves = [half_order[:len(ids)//2], half_order[len(ids)//2:]]
    manifest = {"created_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                "branch": branch, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "n": len(ids), "observations_per_layer_per_pathway": 989, "window": 9,
                "question_ids_in_array_order": ids, "source_csvs": [record(p) for p in source_paths],
                "calibration_source": record(calibration_path), "scoring_implementation": record(ROOT / "peft_lora/methods/flow_guided/make_ranks.py"),
                "analysis_implementation": record(Path(__file__).resolve()),
                "bootstrap": {"replicates": 5000, "seed": 0, "unit": "question", "paired_across_pathways": True,
                              "generator": "numpy.default_rng(0).integers(0,989,size=989) once per draw"},
                "half_question_ids": [[ids[i] for i in indices] for indices in halves],
                "controls": CONTROLS, "criteria": criteria, "tie_break": "higher score then lower layer index",
                "protected_artifacts_before": protected, "versions": {"python": sys.version, "numpy": np.__version__},
                "gpu_jobs_submitted": False, "models_trained_or_evaluated": False}
    out.mkdir(parents=True)
    save_json(out / "manifest.json", manifest)
    scores = score_means(original, means)
    selections = {name: sorted(ordered(scores[name])[:8]) for name in NAMES}
    existing_path = ROOT / "configs/ranks/flow_pruned8_scienceqa.json"
    ranks = json.loads(existing_path.read_text())
    existing = sorted({int(k.split(".")[1]) for k, v in ranks.items() if v > 0})
    if existing != selections["combined"]:
        raise ValueError(f"Original full-data combined top-8 disagrees with existing config: {existing}")
    selections.update(CONTROLS)
    rankings = {name: {"scores": s.tolist(), "descending_layer_order": ordered(s),
                       "top8_rank_order": ordered(s)[:8], "top8_sorted": selections[name],
                       "selected_score_sum": float(s[selections[name]].sum()), "all32_score_sum": float(s.sum()),
                       "selected_score_fraction": float(s[selections[name]].sum()/s.sum()),
                       "vs_existing_combined_top8": overlap(selections[name], existing)} for name, s in scores.items()}
    allocations = {name: {"layers": layers, "fixed_position_control": name in CONTROLS,
                          "score_sums": {objective: float(s[layers].sum()) for objective, s in scores.items()},
                          "vs_existing_combined_top8": overlap(layers, existing)} for name, layers in selections.items()}
    b = manifest["bootstrap"]["replicates"]
    rng = np.random.default_rng(0)
    boot_scores = {name: np.empty((b, 32)) for name in NAMES}
    inclusion = {name: np.zeros((b, 32), dtype=bool) for name in NAMES}
    for draw in range(b):
        sampled = rng.integers(0, len(ids), size=len(ids))
        resampled_scores = score_means(original, cube[:, sampled, :].mean(axis=1))
        for name in NAMES:
            boot_scores[name][draw] = resampled_scores[name]
            inclusion[name][draw, ordered(resampled_scores[name])[:8]] = True
        if (draw + 1) % 1000 == 0:
            print(f"Bootstrap {draw+1}/{b}", flush=True)
    own_overlap, fixed_overlap, paired_overlap, stability, decisions = {}, {}, {}, {}, {}
    for name in NAMES:
        own_overlap[name] = inclusion[name][:, selections[name]].sum(axis=1)
        fixed_overlap[name] = inclusion[name][:, existing].sum(axis=1)
        paired_overlap[name] = (inclusion[name] & inclusion["combined"]).sum(axis=1)
        frequencies = inclusion[name].mean(axis=0)
        half_selections = [sorted(ordered(score_means(original, cube[:, part, :].mean(axis=1))[name])[:8]) for part in halves]
        order = ordered(scores[name]); eighth, ninth = order[7:9]
        gaps = boot_scores[name][:, eighth] - boot_scores[name][:, ninth]
        cutoff = {"eighth_layer": eighth, "ninth_layer": ninth,
                  "eighth_score": float(scores[name][eighth]), "ninth_score": float(scores[name][ninth]),
                  "score_gap": float(scores[name][eighth] - scores[name][ninth]),
                  "gap_ci95": np.percentile(gaps, [2.5, 97.5]).tolist(),
                  "probability_eighth_strictly_exceeds_ninth": float(np.mean(gaps > 0)),
                  "probability_exact_score_tie": float(np.mean(gaps == 0)),
                  "eighth_inclusion_frequency": float(frequencies[eighth]),
                  "ninth_inclusion_frequency": float(frequencies[ninth]),
                  "full_cutoff_exact_tie_count": int(np.sum(scores[name] == scores[name][eighth]))}
        stability[name] = {"overlap_with_full_top8": distribution(own_overlap[name]),
                           "overlap_with_fixed_existing_combined": distribution(fixed_overlap[name]),
                           "overlap_with_paired_resampled_combined": distribution(paired_overlap[name]),
                           "inclusion_frequency": frequencies.tolist(), "half_selections": half_selections,
                           "half_overlaps_with_full": [overlap(layers, selections[name])["count"] for layers in half_selections],
                           "cutoff": cutoff,
                           "boundary_layers_ranks6_to11": [{"rank": rank+1, "layer": layer, "score": float(scores[name][layer]),
                                                            "inclusion_frequency": float(frequencies[layer]),
                                                            "score_bootstrap_ci95": np.percentile(boot_scores[name][:,layer], [2.5,97.5]).tolist()}
                                                           for rank, layer in enumerate(order) if 5 <= rank <= 10]}
        s = stability[name]; d = s["overlap_with_full_top8"]
        distinct = overlap(selections[name], existing)["count"] <= criteria["maximum_combined_overlap_for_distinct"]
        stable = (d["median"] >= criteria["minimum_bootstrap_median_overlap"] and
                  d["fifth_percentile"] >= criteria["minimum_bootstrap_fifth_percentile"] and
                  min(s["half_overlaps_with_full"]) >= criteria["minimum_each_half_overlap"])
        decisions[name] = {"distinct": distinct, "stable": stable, "passes_training_screen": distinct and stable,
                           "overlap_with_early_control": overlap(selections[name], CONTROLS["early_eight"]),
                           "overlap_with_middle_control": overlap(selections[name], CONTROLS["middle_eight"])}
    result = {"n": len(ids), "window": 9, "bootstrap": manifest["bootstrap"], "criteria": criteria,
              "existing_combined_config": record(existing_path), "existing_combined_top8": existing,
              "rankings": rankings, "allocations": allocations, "stability": stability, "decisions": decisions,
              "control_overlap_with_resampled_combined": {name: distribution(inclusion["combined"][:, layers].sum(axis=1)) for name, layers in CONTROLS.items()},
              "models_trained_or_evaluated": False, "gpu_jobs_submitted": False,
              "limits": "Overlapping window-center scores; cohort-conditional bootstrap; no pathway superiority or downstream model performance established."}
    save_json(out / "analysis.json", result)
    write_csv(out / "layer_rankings.csv", [{"layer": layer, **{f"{name}_{field}": value for name in NAMES
              for field, value in (("importance", float(scores[name][layer])), ("rank", float(rankdata(-scores[name])[layer])),
                                   ("bootstrap_inclusion_frequency", stability[name]["inclusion_frequency"][layer]))}}
              for layer in range(32)])
    write_csv(out / "selection_comparison.csv", [{"allocation": name, "sorted_indices": ",".join(map(str, a["layers"])),
              **a["score_sums"], "combined_overlap": a["vs_existing_combined_top8"]["count"],
              "combined_jaccard": a["vs_existing_combined_top8"]["jaccard"]} for name, a in allocations.items()])
    write_csv(out / "bootstrap_overlaps.csv", [{"draw": i, **{f"{name}_{kind}": int(values[name][i])
              for name in NAMES for kind, values in (("own_full",own_overlap),("existing_combined",fixed_overlap),("paired_combined",paired_overlap))}}
              for i in range(b)])
    write_csv(out / "overlap_distributions.csv", [{"shared_layers": k, **{f"{name}_{kind}_count": s[key]["counts"][k]
              for name,s in stability.items() for kind,key in (("own_full","overlap_with_full_top8"),
              ("existing_combined","overlap_with_fixed_existing_combined"),("paired_combined","overlap_with_paired_resampled_combined"))}}
              for k in range(9)])
    np.savez_compressed(out / "bootstrap_scores_and_selections.npz", **{f"{name}_{kind}": values[name]
                        for name in NAMES for kind,values in (("scores",boot_scores),("selected",inclusion))})
    plot_results(out, scores, selections, stability)
    write_report(out, result)
    for item in protected:
        if record(ROOT / item["path"])["sha256"] != item["sha256"]:
            raise ValueError(f"Protected artifact changed: {item['path']}")
    save_json(out / "preservation.json", {"all_protected_artifacts_unchanged": True, "checked_artifact_count": len(protected),
              "scope": "Input CSVs, calibration CSV, original scoring/budget code, all pre-existing rank configs, prior region/window analyses",
              "output_directory_was_new": True, "models_trained_or_evaluated": False, "gpu_jobs_submitted": False})
    save_json(out / "status.json", {"state": "analysis_complete", "n": len(ids), "bootstrap_replicates": b,
              "decisions": decisions, "training": "not_submitted", "model_evaluation": "not_run"})
    print(json.dumps({"output": str(out), "rankings": rankings, "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
