#!/usr/bin/env python3
"""CPU-only paired analysis of existing ScienceQA predictions; new outputs only."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import scipy
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output/pathway_specific/scienceqa/paired_errors_seed0_20261029"
REPORT = ROOT / "docs/PATHWAY_TOP8_PAIRED_ERRORS.md"
NAMES = ("combined", "question_last", "middle")
LABELS = {"combined": "Combined", "question_last": "Question→Last", "middle": "Middle control"}
STEMS = ("flow_pruned8_scienceqa", "question_last_pruned8_scienceqa", "middle_pruned8_scienceqa")
EXPECTED = (1605, 1572, 1522)
JOBS = {"combined": ("47417", "47421"), "question_last": ("48610", "48619"), "middle": ("48611", "48623")}
PAIRS = tuple(combinations(range(3), 2))
MATCH_KEYS = ("seed", "model_path", "conv_mode", "train_file", "image_folder", "tag", "limit", "epochs",
              "batch_size", "grad_accum", "lr", "warmup_ratio", "save_every", "max_hours", "num_workers",
              "trainable_params", "total_steps")
sys.dont_write_bytecode = True


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.open()]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_evaluator():
    # This module only imports model/GPU packages inside its guarded main().
    path = ROOT / "peft_lora/eval_gqa.py"
    spec = importlib.util.spec_from_file_location("original_scienceqa_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contingency(a, b):
    return {"both_correct": int(np.sum(a & b)), "both_wrong": int(np.sum(~a & ~b)),
            "a_only_correct": int(np.sum(a & ~b)), "b_only_correct": int(np.sum(~a & b))}


def holm(pvalues):
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues)-rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_differences(correct, replicates, seed):
    # One question-index draw is shared by all models and all three contrasts.
    n = correct.shape[1]
    differences = np.array([correct[a].astype(np.int8)-correct[b].astype(np.int8) for a,b in PAIRS])
    rng = np.random.default_rng(seed)
    draws = np.empty((3, replicates))
    for start in range(0, replicates, 256):
        end = min(start + 256, replicates)
        indices = rng.integers(0, n, size=(end-start, n))
        draws[:, start:end] = differences[:, indices].mean(axis=2) * 100
    return np.percentile(draws, [2.5, 97.5], axis=1).T, draws


def examples_for(questions, prediction_maps, correct, pair_index):
    a, b = PAIRS[pair_index]
    result = {}
    for direction, mask in (("a_only_correct", correct[a] & ~correct[b]), ("b_only_correct", correct[b] & ~correct[a])):
        candidates = np.flatnonzero(mask).tolist()
        subjects = Counter(questions[i]["subject"] for i in candidates)
        chosen = []
        # One hash-chosen example from each of the three largest subject strata.
        # No manual selection by interesting wording or a preferred explanation.
        for subject, _ in sorted(subjects.items(), key=lambda item: (-item[1], item[0]))[:3]:
            pool = [i for i in candidates if questions[i]["subject"] == subject]
            index = min(pool, key=lambda i: (hashlib.sha256(f"scienceqa-paired-examples-v1|{pair_index}|{direction}|{questions[i]['question_id']}".encode()).hexdigest(), questions[i]["question_id"]))
            q = questions[index]
            chosen.append({"question_id": q["question_id"], "subject": subject, "grade": q["grade"],
                "disagreement_subject_count": subjects[subject], "question": q["question"], "full_prompt": q["text"],
                "image": str(Path("datasets/scienceqa/images") / q["image"]), "answer": q["answer"],
                "a_prediction": prediction_maps[a][q["question_id"]]["prediction"],
                "b_prediction": prediction_maps[b][q["question_id"]]["prediction"]})
        result[direction] = chosen
    return result


def plot(out, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.55, 1]})
    labels = [f"{LABELS[p['a']]} − {LABELS[p['b']]}" for p in pairs]
    keys = ("both_correct", "both_wrong", "a_only_correct", "b_only_correct")
    matrix = np.array([[p[k] for k in keys] for p in pairs])
    axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max(), aspect="auto")
    axes[0].set_xticks(range(4), ["Both correct", "Both wrong", "A-only correct", "B-only correct"])
    axes[0].set_yticks(range(3), labels)
    for i in range(3):
        for j in range(4):
            axes[0].text(j, i, f"{matrix[i,j]:,}\n{100*matrix[i,j]/2017:.1f}%", ha="center", va="center",
                         color="white" if matrix[i,j] > matrix.max()*.55 else "black")
    axes[0].set_title("Paired outcomes on the same 2,017 questions")
    for i,p in enumerate(pairs):
        center = p["accuracy_difference_pp"]
        lower, upper = p["paired_bootstrap_ci95_pp"]
        axes[1].errorbar(center, i, xerr=[[center-lower], [upper-center]], fmt="o", capsize=5, color="#286b95")
        axes[1].annotate(f"{center:+.2f} [{lower:+.2f}, {upper:+.2f}]", (center, i), xytext=(0, 13), textcoords="offset points", ha="center", fontsize=9)
    axes[1].set_yticks(range(3), ["A − B"]*3)
    axes[1].set_ylim(2.5,-.55)
    axes[1].axvline(0,color="gray",linestyle="--",linewidth=1)
    axes[1].set_xlabel("Accuracy difference (percentage points)")
    axes[1].set_title("95% paired-question bootstrap intervals")
    axes[1].grid(axis="x", alpha=.2)
    fig.text(.5,.012,"Fixed seed-0 models: question-sampling uncertainty only; training-seed uncertainty is not estimated.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.045,1,1))
    for suffix in ("png", "pdf"):
        fig.savefig(out/f"paired_errors.{suffix}",dpi=180)
    plt.close(fig)


def report(path, out, result, questions, correct):
    pairs = result["pairs"]
    lines = ["# ScienceQA seed-0 paired error analysis", "",
        "All three files contain the same **2,017 unique held-out question IDs and exact gold labels**.",
        "Predictions were aligned by ID and rescored with the unchanged evaluator's",
        "`answer.strip().rstrip('.').lower()` rule; no option extraction or rescoring change was introduced.",
        "Verified totals: Combined **1,605**, Question→Last **1,572**, middle control **1,522** correct.", "",
        "## Paired comparison", "",
        "A and B follow the order in the first column. Positive differences favor A.", "",
        "| A − B | Both correct | Both wrong | A-only | B-only | Δ pp [95% paired CI] | Exact McNemar p | Holm p |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for p in pairs:
        lo,hi = p["paired_bootstrap_ci95_pp"]
        lines.append(f"| {LABELS[p['a']]} − {LABELS[p['b']]} | {p['both_correct']} | {p['both_wrong']} | {p['a_only_correct']} | {p['b_only_correct']} | {p['accuracy_difference_pp']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p['mcnemar_exact_p']:.6g} | {p['mcnemar_holm_p']:.6g} |")
    lines += ["", f"Intervals use **{result['bootstrap']['replicates']:,} ordinary paired question-bootstrap draws**, seed {result['bootstrap']['seed']};",
        "each draw resamples 2,017 question IDs with replacement, using the same indices across models.",
        "Reported endpoints are percentile 2.5/97.5; they are marginal 95% intervals, not simultaneous family-wise intervals.",
        "Both-wrong includes cases with different incorrect predictions.", "",
        "Exact two-sided McNemar tests use a Binomial(discordant count, 0.5) null on A-only/B-only outcomes;",
        "no chi-square approximation or continuity correction is used. See the",
        "[exact-test documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html) and",
        "[SciPy binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).",
        "Holm adjustment covers the three overall pairwise tests only.", "",
        "## Subject and grade differences", "",
        "Percentages and signed differences below are descriptive; tiny groups should not be generalized."]
    for field in ("subject", "grade"):
        lines += ["", f"### {field.title()}", "",
            "| Group | n | Combined % | Question→Last % | Middle % | C−Q pp | C−M pp | Q−M pp |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        values = sorted({q[field] for q in questions}, key=lambda s: int(s.removeprefix("grade")) if field=="grade" else s)
        for value in values:
            mask = np.array([q[field]==value for q in questions]);acc = correct[:,mask].mean(axis=1)*100
            lines.append(f"| {value} | {int(mask.sum())} | {acc[0]:.2f} | {acc[1]:.2f} | {acc[2]:.2f} | {acc[0]-acc[1]:+.2f} | {acc[0]-acc[2]:+.2f} | {acc[1]-acc[2]:+.2f} |")
    lines += ["", "## Illustrative disagreements", "",
        "For each direction, examples were selected by a fixed SHA-256 rule within the largest disagreement-subject strata.",
        "The table shows the first example per direction; full prompts, images, and additional subject examples are in `analysis.json`.",
        "These illustrate observed disagreements, not an unbiased estimate of error mechanisms.", "",
        "| A / B | Who succeeds | Question / subject / grade | Gold; A output; B output |",
        "| --- | --- | --- | --- |"]
    for p in pairs:
        for direction, winner in (("a_only_correct","A"),("b_only_correct","B")):
            selected = p["examples"][direction]
            if not selected:
                continue
            e = selected[0]
            stem = e["question"].replace("|","\\|").replace("\n"," ")
            a = e["a_prediction"].replace("|","\\|").replace("\n"," ")
            b = e["b_prediction"].replace("|","\\|").replace("\n"," ")
            lines.append(f"| {LABELS[p['a']]} / {LABELS[p['b']]} | {winner} | {e['question_id']}: {stem} ({e['subject']}, {e['grade']}) | {e['answer']}; {a}; {b} |")
    lines += ["", "## Original Slurm and configuration provenance", "",
        "Slurm accounting timestamps are **Asia/Singapore (UTC+08:00)**, not file modification times.", "",
        "| Model | Stage / job | Start SGT | End SGT | Elapsed | State / exit |",
        "| --- | --- | --- | --- | --- | --- |"]
    by_id = {r["job_id"]:r for r in result["provenance"]["slurm"]["records"]}
    for name in NAMES:
        for stage,job in zip(("train","eval"),JOBS[name]):
            r = by_id[job]
            lines.append(f"| {LABELS[name]} | {stage} / {job} | {r['started_at'].replace('T',' ')} | {r['ended_at'].replace('T',' ')} | {r['elapsed']} | {r['state']} / {r['exit_code']} |")
    lines += ["", "All six jobs ran on TC2N01 with one GPU and 30 GB RAM; training requested eight CPUs and evaluation four.",
        "The baseline's original submission commands explicitly identify `flow_pruned8_scienceqa`, seed 0 and the ScienceQA files.",
        "Shared saved training settings match: LLaVA-1.5-7B, vicuna_v1, one epoch, microbatch 4 × accumulation 4,",
        "lr 2e-4, 3% warmup, 388 steps, and 1,048,576 trainable parameters. New adapters verify rank 8, alpha 16, dropout 0.05.",
        "The frozen new-run trainer, evaluator, shared loader and datasets match baseline artifact-introduction commit",
        "`6efc769401506c0a2b5c5416578f8f0f05602891`; verified against the recorded run revision, not current branch files.",
        "The curated branch includes later teammate source changes. Historical source consistency does not attest the baseline runtime revision:",
        "its checkpoint is absent, and its config omits the newer inactive flow-gate fields. The metadata-addition commit",
        "is not proof of the checked-out source revision at training time. New runs record commit",
        "`8734991cd5c80eff45b09eb954d6d849ef495d58`. Original submitted paths are retained alongside relocation mappings in the existing manifest.", "",
        "## Uncertainty and scope", "",
        "These are comparisons of **three fixed seed-0 trained artifacts**. Bootstrap intervals and McNemar p-values describe",
        "evaluation-question uncertainty under the paired/i.i.d.-question sampling model. They do not measure training-seed",
        "variation, certify performance on another population, or establish pathway-level statistical superiority across seeds.",
        "Correlated questions would weaken the i.i.d. approximation. No subgroup significance tests or causal error explanations are claimed.",
        "No inference, training, checkpoint changes, or GPU jobs were performed. Existing experiment artifacts passed hash-preservation checks.", "",
        f"[Comparison table](../{out.relative_to(ROOT)}/comparison.csv) · "
        f"[Subject/grade details](../{out.relative_to(ROOT)}/subgroup_differences.csv) · "
        f"[Paired-error visualization](../{out.relative_to(ROOT)}/paired_errors.png) · "
        f"[Per-question audit](../{out.relative_to(ROOT)}/paired_correctness.csv) · "
        f"[Full results and provenance](../{out.relative_to(ROOT)}/analysis.json)", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--slurm-audit", type=Path, help="Optional saved sacct snapshot; default: retained analysis provenance")
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out, report_path = args.output_dir.resolve(), args.report.resolve()
    if out.exists() or report_path.exists():
        raise ValueError("Refusing to overwrite existing analysis/report; choose fresh destinations")
    evaluator = load_evaluator()
    question_file = ROOT / "datasets/scienceqa/test_questions.jsonl"
    questions = jsonl(question_file)
    ids = [q["question_id"] for q in questions]
    if len(ids)!=2017 or len(set(ids))!=2017:
        raise ValueError("Expected 2017 unique held-out question IDs")
    question_map = {q["question_id"]:q for q in questions}
    source_paths = [question_file, ROOT/"datasets/scienceqa/train_llava.json", ROOT/"peft_lora/train_lora.py",
                    ROOT/"peft_lora/eval_gqa.py", ROOT/"peft_lora/common.py", ROOT/"output/pathway_top8_seed0/submission_manifest.json",
                    ROOT/"configs/ranks/uniform.json"]
    prediction_maps, correct, models = [], [], {}
    configs = []
    orders = []
    for index,(name,stem) in enumerate(zip(NAMES,STEMS)):
        run_dir = ROOT/"output/peft"/stem/"scienceqa_train_llava/seed0"
        pred_file, metric_file = run_dir/"eval_scienceqa/predictions.jsonl", run_dir/"eval_scienceqa/metrics.json"
        predictions = jsonl(pred_file)
        order = [p["question_id"] for p in predictions]
        if len(order)!=2017 or len(set(order))!=2017 or set(order)!=set(ids):
            raise ValueError(f"Duplicate/missing/foreign question IDs: {pred_file}")
        lookup = {p["question_id"]:p for p in predictions}
        if any(lookup[q]["answer"] != question_map[q]["answer"] for q in ids):
            raise ValueError(f"Gold labels differ from held-out source: {name}")
        flags = np.array([evaluator.normalize(lookup[q]["prediction"])==evaluator.normalize(question_map[q]["answer"]) for q in ids])
        metrics = read(metric_file)
        if int(flags.sum())!=EXPECTED[index] or metrics["correct"]!=EXPECTED[index] or metrics["total"]!=2017 or metrics["accuracy"]!=round(flags.mean()*100,2):
            raise ValueError(f"Recomputed metrics mismatch: {name}")
        for field in ("subject","grade"):
            for value in {q[field] for q in questions}:
                mask = np.array([q[field]==value for q in questions]); saved = metrics[f"by_{field}"][value]
                if saved["total"]!=int(mask.sum()) or saved["accuracy"]!=round(flags[mask].mean()*100,2):
                    raise ValueError(f"Saved subgroup metric mismatch: {name}/{field}/{value}")
        cfg = read(run_dir/"train_config.json"); configs.append(cfg)
        rank_file = ROOT/cfg["ranks"]; ranks = read(rank_file)
        selected = sorted({int(k.split('.')[1]) for k,v in ranks.items() if v>0})
        source_paths.extend([pred_file,metric_file,run_dir/"train_config.json",run_dir/"train_log.jsonl",rank_file])
        adapter_dir = run_dir/"adapter"
        if adapter_dir.exists():
            source_paths.extend(p for p in adapter_dir.iterdir() if p.is_file())
            adapter_cfg = read(adapter_dir/"adapter_config.json")
            if (adapter_cfg["r"],adapter_cfg["lora_alpha"],adapter_cfg["lora_dropout"])!=(8,16,.05):
                raise ValueError(f"Saved adapter config mismatch: {name}")
        models[name] = {"run_dir": str(run_dir.relative_to(ROOT)), "predictions": record(pred_file), "metrics": record(metric_file),
            "training_config": cfg, "rank_configuration": record(rank_file), "selected_layers": selected,
            "correct": int(flags.sum()), "total": 2017, "accuracy_percent": float(flags.mean()*100),
            "adapter_available": adapter_dir.exists(), "raw_prediction_changed_by_normalization": sum(lookup[q]["prediction"]!=evaluator.normalize(lookup[q]["prediction"]) for q in ids)}
        prediction_maps.append(lookup);correct.append(flags);orders.append(order)
    correct = np.array(correct)
    if any(any(cfg.get(k)!=configs[0].get(k) for k in MATCH_KEYS) for cfg in configs[1:]):
        raise ValueError("Shared saved training settings do not match")
    protected = [record(p) for p in dict.fromkeys(source_paths)]
    manifest = read(ROOT/"output/pathway_top8_seed0/submission_manifest.json")
    # Historical manifests keep their original absolute paths. Verify Git blobs
    # from the recorded training revision, never another live worktree's files.
    source_commit = manifest["git_commit"]
    submitted_matches = []
    baseline_commit = "6efc769401506c0a2b5c5416578f8f0f05602891"
    baseline_source = {}
    relatives = ("peft_lora/train_lora.py", "peft_lora/eval_gqa.py", "peft_lora/common.py",
                 "datasets/scienceqa/train_llava.json", "datasets/scienceqa/test_questions.jsonl")
    for relative in relatives:
        item = next(item for item in manifest["submitted_files"] if item["path"].endswith("/" + relative))
        blob = subprocess.check_output(["git", "show", f"{source_commit}:{relative}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != item["sha256"]:
            raise ValueError(f"Recorded training revision does not match frozen input: {relative}")
        submitted_matches.append({**item, "matches_recorded_commit": True,
                                  "matches_current": blob == (ROOT / relative).read_bytes()})
        baseline_blob = subprocess.check_output(["git", "show", f"{baseline_commit}:{relative}"], cwd=ROOT)
        baseline_source[relative] = {"identical_to_recorded_run_source": baseline_blob == blob,
                                    "identical_to_current": baseline_blob == (ROOT / relative).read_bytes(),
                                    "sha256": hashlib.sha256(baseline_blob).hexdigest()}
    slurm = read(args.slurm_audit) if args.slurm_audit else read(OUT / "analysis.json")["provenance"]["slurm"]
    if {r["job_id"] for r in slurm["records"]}!={j for jobs in JOBS.values() for j in jobs}:
        raise ValueError("Incomplete original Slurm accounting snapshot")
    for job in slurm["records"]:
        if job["state"]!="COMPLETED" or job["exit_code"]!="0:0":
            raise ValueError(f"Unsuccessful original job: {job['job_id']}")
    intervals, draws = bootstrap_differences(correct,args.bootstrap_replicates,args.seed)
    pairs = []
    for index,(a,b) in enumerate(PAIRS):
        c = contingency(correct[a],correct[b]);discordant=c["a_only_correct"]+c["b_only_correct"]
        p = float(binomtest(c["a_only_correct"],discordant,p=.5,alternative="two-sided").pvalue) if discordant else 1.0
        pairs.append({"a":NAMES[a],"b":NAMES[b],"n":2017,**c,"a_correct":int(correct[a].sum()),"b_correct":int(correct[b].sum()),
            "accuracy_difference_pp":float((correct[a].mean()-correct[b].mean())*100),
            "paired_bootstrap_ci95_pp":intervals[index].tolist(),"mcnemar_exact_p":p,
            "examples":examples_for(questions,prediction_maps,correct,index)})
    for pair,p in zip(pairs,holm([p["mcnemar_exact_p"] for p in pairs])):
        pair["mcnemar_holm_p"]=p
    groups = []
    for field in ("subject","grade"):
        for value in sorted({q[field] for q in questions}):
            mask=np.array([q[field]==value for q in questions])
            for a,b in PAIRS:
                groups.append({"field":field,"group":value,"n":int(mask.sum()),"a":NAMES[a],"b":NAMES[b],
                    "a_correct":int(correct[a,mask].sum()),"b_correct":int(correct[b,mask].sum()),
                    "a_accuracy_percent":float(correct[a,mask].mean()*100),"b_accuracy_percent":float(correct[b,mask].mean()*100),
                    "accuracy_difference_pp":float((correct[a,mask].mean()-correct[b,mask].mean())*100),
                    **contingency(correct[a,mask],correct[b,mask])})
    result={"created_at":datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),"n":2017,
        "models":models,"pairs":pairs,"subgroups":groups,
        "verification":{"unique_ids_match":True,"exact_gold_labels_match":True,"original_prediction_orders_identical":orders[0]==orders[1]==orders[2],
                        "normalization":"Original evaluator.normalize: answer.strip().rstrip('.').lower()", "saved_overall_and_group_metrics_reproduce":True},
        "bootstrap":{"replicates":args.bootstrap_replicates,"seed":args.seed,"resampling_unit":"question","paired_across_models":True,
                     "interval":"2.5/97.5 percentile; marginal 95%; same draw indices for all contrasts","question_order_sha256":hashlib.sha256('\n'.join(ids).encode()).hexdigest()},
        "example_selection":"For each pair/direction, hash-chosen question in each of three largest disagreement-subject strata; no manual cherry-picking",
        "provenance":{"analysis_commit":git("rev-parse","HEAD"),"analysis_implementation":record(Path(__file__).resolve()),
            "slurm":slurm,"matched_training_fields":list(MATCH_KEYS),"new_run_submitted_source_checks":submitted_matches,
            "baseline_artifact_addition_commit":baseline_commit,"baseline_source_comparison":baseline_source,
            "baseline_runtime_commit":"Not recorded; artifact-addition commit is not runtime-attestation",
            "new_run_recorded_commit":manifest["git_commit"],"baseline_config_missing_inactive_fields":[k for k in ("flow_gates","gate_lr") if k not in configs[0]],
            "preserved_inputs":protected,"packages":{"numpy":np.__version__,"scipy":scipy.__version__}},
        "uncertainty":"Conditional evaluation-question uncertainty for fixed seed-0 artifacts; no training-seed uncertainty estimated; no superiority claim across seeds",
        "gpu_jobs_submitted":False,"inference_performed":False}
    out.mkdir(parents=True)
    write_csv(out/"comparison.csv",[{**{k:p[k] for k in ("a","b","n","a_correct","b_correct","both_correct","both_wrong","a_only_correct","b_only_correct","accuracy_difference_pp")},
        "ci95_lower_pp":p["paired_bootstrap_ci95_pp"][0],"ci95_upper_pp":p["paired_bootstrap_ci95_pp"][1],
        "mcnemar_exact_p":p["mcnemar_exact_p"],"mcnemar_holm_p":p["mcnemar_holm_p"]} for p in pairs])
    write_csv(out/"subgroup_differences.csv",groups)
    write_csv(out/"paired_correctness.csv",[{"question_id":q["question_id"],"subject":q["subject"],"grade":q["grade"],"answer":q["answer"],
        **{f"{name}_{field}":value for model,name in enumerate(NAMES) for field,value in (
            ("prediction",prediction_maps[model][q["question_id"]]["prediction"]),
            ("normalized_prediction",evaluator.normalize(prediction_maps[model][q["question_id"]]["prediction"])),
            ("correct",int(correct[model,i])))}} for i,q in enumerate(questions)])
    plot(out,pairs)
    report(report_path,out,result,questions,correct)
    for item in protected:
        if record(ROOT/item["path"])["sha256"]!=item["sha256"]:
            raise ValueError(f"Existing experiment input changed during analysis: {item['path']}")
    result["verification"]["existing_inputs_unchanged"]=True
    save_json(out/"analysis.json",result)
    print(json.dumps({"output":str(out),"report":str(report_path),"pairs":[{k:v for k,v in p.items() if k!='examples'} for p in pairs]},indent=2))


if __name__=="__main__":
    main()
