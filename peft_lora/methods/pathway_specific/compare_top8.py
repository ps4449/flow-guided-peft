#!/usr/bin/env python3
"""CPU-only four-checkpoint comparison; requires completed early evaluation.

Extracted from the deployed early-control analysis. No scheduling, training or
inference. Every output/report destination must be new. Historical run paths in
manifests are evidence, never locations to write or fetch missing files from.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from itertools import combinations
import importlib.util
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
MODEL_NAMES = ("combined", "question_last", "middle", "early")
LABELS = ("Combined-flow", "Question→Last", "Fixed middle", "Fixed early")
STEMS = ("flow_pruned8_scienceqa", "question_last_pruned8_scienceqa", "middle_pruned8_scienceqa")
EXPECTED_CORRECT = (1605, 1572, 1522)
PAIRS = ((0, 3),) + tuple(pair for pair in combinations(range(4), 2) if pair != (0, 3))
MATCH_KEYS = ("seed", "model_path", "conv_mode", "train_file", "image_folder", "tag", "limit", "epochs",
              "batch_size", "grad_accum", "lr", "warmup_ratio", "save_every", "max_hours", "num_workers",
              "trainable_params", "total_steps")


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def read(path):
    return json.loads(path.read_text())


def atomic_save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def local_rank_path(value):
    path = Path(value)
    original = Path("/home/msai/swarangi001/flow-guided-peft")
    if path.is_absolute():
        if not path.is_relative_to(original):
            raise ValueError(f"Unexpected historical rank path: {path}")
        path = path.relative_to(original)
    return ROOT / path


def evaluator():
    spec = importlib.util.spec_from_file_location("unchanged_scienceqa_evaluator", ROOT / "peft_lora/eval_gqa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def original_runs():
    return [ROOT / "output/peft" / stem / "scienceqa_train_llava/seed0" for stem in STEMS]

def load_predictions(run_dir, questions, expected_correct=None):
    raw = [json.loads(line) for line in (run_dir / "eval_scienceqa/predictions.jsonl").open()]
    by_id = {p["question_id"]: p for p in raw}
    gold = {q["question_id"]: q for q in questions}
    if len(questions) != 2017 or len(gold) != 2017:
        raise ValueError("Expected 2017 unique held-out question IDs")
    if len(raw) != 2017 or len(by_id) != 2017 or set(by_id) != set(gold):
        raise ValueError(f"Prediction IDs do not match all 2017 unique questions: {run_dir}")
    normalize = evaluator().normalize
    for qid, prediction in by_id.items():
        if prediction["answer"] != gold[qid]["answer"]:
            raise ValueError(f"Exact gold-label disagreement: {run_dir}/{qid}")
    flags = [normalize(by_id[q["question_id"]]["prediction"]) == normalize(q["answer"]) for q in questions]
    metrics = read(run_dir / "eval_scienceqa/metrics.json")
    correct = sum(flags)
    if metrics["total"] != 2017 or metrics["correct"] != correct or metrics["accuracy"] != round(100*correct/2017, 2):
        raise ValueError(f"Saved metrics disagree with original normalization: {run_dir}")
    if expected_correct is not None and correct != expected_correct:
        raise ValueError(f"Unexpected historical score: {run_dir}")
    return flags, metrics

def final_comparison():
    import csv
    import numpy as np
    from scipy.stats import binomtest
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    manifest=read(EARLY/"submission_manifest.json")
    questions=[json.loads(line) for line in (ROOT/"datasets/scienceqa/test_questions.jsonl").open()]
    directories=original_runs()+[EARLY]
    configs=[read(directory/"train_config.json") for directory in directories]
    if any(any(cfg.get(k) != configs[0].get(k) for k in MATCH_KEYS) for cfg in configs[1:]):
        raise ValueError("Shared training configurations do not match")
    if any(cfg["seed"] != 0 or cfg["trainable_params"] != 1048576 for cfg in configs):
        raise ValueError("Expected seed 0 and 1,048,576 parameters")
    flags=[];model_rows=[]
    for i,(name,directory) in enumerate(zip(MODEL_NAMES,directories)):
        matched,metrics=load_predictions(directory,questions,EXPECTED_CORRECT[i] if i<3 else None)
        flags.append(matched)
        config=read(directory/"train_config.json");rank_path=local_rank_path(config["ranks"])
        rank_spec=read(rank_path);layers=sorted({int(key.split('.')[1]) for key,value in rank_spec.items() if value>0})
        expected_layers = ([0,1,2,3,4,9,10,11], list(range(9,17)), list(range(12,20)), list(range(8)))[i]
        expected_spec = {f"layers.{layer}.self_attn.{projection}": 8 if layer in expected_layers else 0 for layer in range(32) for projection in ("q_proj", "v_proj")}
        if rank_spec != expected_spec:
            raise ValueError(f"Unexpected rank configuration: {rank_path}")
        entries=[json.loads(line) for line in (directory/"train_log.jsonl").open()]
        model_rows.append({"model":name,"seed":0,"selected_layers":layers,"correct":sum(matched),"total":2017,
            "accuracy":100*sum(matched)/2017,"trainable_parameters":config["trainable_params"],
            "trainer_logged_minutes":entries[-1]["minutes"],"adapter":str(directory/"adapter"),
            "predictions":str(directory/"eval_scienceqa/predictions.jsonl"),"metrics":str(directory/"eval_scienceqa/metrics.json"),
            "rank_configuration":str(rank_path)})
    correct=np.array(flags,dtype=bool);differences=np.array([correct[a].astype(np.int8)-correct[b].astype(np.int8) for a,b in PAIRS])
    rng=np.random.default_rng(0);draws=np.empty((len(PAIRS),20000))
    for start in range(0,20000,256):
        end=min(start+256,20000);indices=rng.integers(0,2017,size=(end-start,2017))
        draws[:,start:end]=differences[:,indices].mean(axis=2)*100
    intervals=np.percentile(draws,[2.5,97.5],axis=1).T
    pairs=[]
    for index,(a,b) in enumerate(PAIRS):
        ac,bc=correct[a],correct[b];a_only=int(np.sum(ac&~bc));b_only=int(np.sum(~ac&bc));discordant=a_only+b_only
        pairs.append({"a":MODEL_NAMES[a],"b":MODEL_NAMES[b],"primary":index==0,
            "both_correct":int(np.sum(ac&bc)),"both_wrong":int(np.sum(~ac&~bc)),"a_only_correct":a_only,"b_only_correct":b_only,
            "accuracy_difference_pp":(a_only-b_only)*100/2017,"ci95_lower_pp":float(intervals[index,0]),"ci95_upper_pp":float(intervals[index,1]),
            "mcnemar_exact_p":float(binomtest(a_only,discordant,.5,alternative="two-sided").pvalue) if discordant else 1.0})
    order=np.argsort([p["mcnemar_exact_p"] for p in pairs]);maximum=0.0
    for rank,index in enumerate(order):
        maximum=max(maximum,(len(pairs)-rank)*pairs[index]["mcnemar_exact_p"])
        pairs[index]["mcnemar_holm_all_six_p"]=min(1.0,maximum)
    def table(path,rows):
        with path.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    OUT.mkdir(parents=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    table(OUT/"summary.csv",[{**r,"selected_layers":",".join(map(str,r["selected_layers"]))} for r in model_rows])
    table(OUT/"paired_comparison.csv",pairs)
    fig,ax=plt.subplots(figsize=(9,5));values=[r["accuracy"] for r in model_rows]
    bars=ax.bar(LABELS,values,color=["#286ca3","#d17a22","#7e8793","#27856d"])
    for bar,r in zip(bars,model_rows):
        ax.text(bar.get_x()+bar.get_width()/2,r["accuracy"]+1,f"{r['accuracy']:.2f}%\n{r['correct']}/2017",ha="center",fontsize=10)
    ax.set_ylim(0,100);ax.set_ylabel("Accuracy (%)");ax.set_title("ScienceQA: fixed seed-0 checkpoints, same 2,017 questions")
    ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(OUT/"accuracy.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,3.8));matrix=np.array([[int(i in r["selected_layers"]) for i in range(32)] for r in model_rows])
    ax.imshow(matrix,cmap=ListedColormap(["#f1f3f5","#27856d"]),vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(32));ax.set_yticks(range(4),LABELS)
    ax.set_xticks(np.arange(-.5,32,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True);ax.grid(which="minor",color="white",linewidth=1.5)
    ax.tick_params(which="minor",bottom=False,left=False);ax.set_xlabel("Transformer layer (green = rank-8 q_proj/v_proj LoRA)")
    ax.set_title("Eight selected layers; all 32 original transformer layers preserved")
    fig.tight_layout();fig.savefig(OUT/"selected_layers.png",dpi=180);plt.close(fig)
    output={"created_at":now(),"models":model_rows,"paired_comparisons":pairs,"primary_comparison":["combined","early"],
        "bootstrap":{"replicates":20000,"seed":0,"unit":"paired question","same_draws_for_all_models":True,"interval":"2.5/97.5 marginal percentile"},
        "verification":{"same_2017_unique_ids":True,"exact_gold_labels_agree":True,"normalization":"unchanged evaluator.normalize: strip().rstrip('.').lower()"},
        "source_commit":manifest["source_commit"],"packages":manifest["packages"],
        "uncertainty":"Fixed-checkpoint evaluation-question significance does not quantify training-seed reliability; seed 0 is exploratory",
        "historical_predictions_rescored":True}
    atomic_save(OUT/"final_comparison.json",output)
    lines=["# Final ScienceQA top-8 comparison", "", "**Prespecified primary comparison: combined-flow versus fixed early layers 0–7.**", "",
        "All four fixed checkpoints use seed 0, rank-8 q_proj/v_proj LoRA, alpha 16, dropout 0.05, and 1,048,576 trainable parameters.",
        "All 32 base layers remain. The matched protocol uses the original LLaVA-1.5-7B, corrected multimodal batching, uppercase option-letter answers,",
        "vicuna_v1, one epoch, batch 4 × accumulation 4, LR 2e-4, 3% warmup, bf16 AdamW and cosine scheduling for 388 updates.", "",
        "| Fixed checkpoint | Layers | Correct / total | Accuracy | Trainer minutes |", "| --- | --- | --- | --- | --- |"]
    for label,r in zip(LABELS,model_rows):
        lines.append(f"| {label} | {r['selected_layers']} | {r['correct']}/2017 | {r['accuracy']:.2f}% | {r['trainer_logged_minutes']} |")
    primary=pairs[0]
    lines += ["", "## Primary paired result", "",
        f"Combined − early: **{primary['accuracy_difference_pp']:+.3f} percentage points**, 95% paired-question bootstrap interval "
        f"**[{primary['ci95_lower_pp']:+.3f}, {primary['ci95_upper_pp']:+.3f}]**; exact two-sided McNemar p = **{primary['mcnemar_exact_p']:.6g}**.",
        f"Both correct: **{primary['both_correct']}**; both wrong: **{primary['both_wrong']}**; Combined-only correct: **{primary['a_only_correct']}**; early-only correct: **{primary['b_only_correct']}**.", "",
        "## All paired contrasts", "", "Positive differences favor A. Primary contrast was declared before early training; the others are secondary/exploratory.", "",
        "| A − B | Both correct | Both wrong | A-only | B-only | Δ pp [95% CI] | Exact McNemar p | Holm p (all six) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for p in pairs:
        lines.append(f"| {p['a']} − {p['b']} | {p['both_correct']} | {p['both_wrong']} | {p['a_only_correct']} | {p['b_only_correct']} | {p['accuracy_difference_pp']:+.3f} [{p['ci95_lower_pp']:+.3f}, {p['ci95_upper_pp']:+.3f}] | {p['mcnemar_exact_p']:.6g} | {p['mcnemar_holm_all_six_p']:.6g} |")
    job_text=" / ".join(manifest["jobs"].get(stage,{}).get("job_id","unknown") for stage in ("train","eval"))
    lines += ["", "Predictions are aligned by the same 2,017 unique question IDs and exact gold labels. The original answer normalization is unchanged.",
        "Intervals use 20,000 paired question resamples with seed 0 and are marginal 95% percentile intervals, not simultaneous intervals.",
        "McNemar uses the exact two-sided binomial test on discordant questions; all-six Holm values are additionally reported.", "",
        "## Interpretation and limits", "",
        f"For these fixed seed-0 artifacts, Combined answers {primary['a_only_correct']-primary['b_only_correct']:+d} more questions correctly than early-layer LoRA.",
        "These tests describe evaluation-question uncertainty conditional on these trained checkpoints, under an i.i.d.-question sampling approximation.",
        "They do **not** measure training-seed variation or establish reliable method superiority across training runs. One seed is exploratory.",
        "A position-control comparison can inform this project hypothesis without proving that flow rankings identify isolated causal layers.", "",
        "## Artifacts and provenance", "",
        f"Source: `dev` at `{manifest['source_commit']}`; exact source-file hashes and package versions are in the new experiment manifest.",
        "Uncommitted launcher/helper files are recorded by hash. The historical baseline's exact runtime revision was not recorded, as documented in the previous paired audit.",
        f"Early training / evaluation job IDs: **{job_text}**. GPU work ran exclusively inside Slurm allocations with the unchanged TC2 resources.",
        "Final adapter: `outputs/final_early8_seed0/adapter/`; resumable `checkpoint.pt` is removed by the unchanged trainer after success.",
        "Predictions and metrics: `outputs/final_early8_seed0/eval_scienceqa/`. Historical artifacts were read only; IDs, labels, rank maps and saved metrics were checked.", "",
        f"[Summary CSV](../{OUT.relative_to(ROOT)}/summary.csv) · [Paired comparisons](../{OUT.relative_to(ROOT)}/paired_comparison.csv) · "
        f"[Accuracy chart](../{OUT.relative_to(ROOT)}/accuracy.png) · [Selected layers](../{OUT.relative_to(ROOT)}/selected_layers.png) · "
        f"[Full analysis](../{OUT.relative_to(ROOT)}/final_comparison.json) · [Submission manifest](../outputs/final_early8_seed0/submission_manifest.json)", ""]
    REPORT.write_text("\n".join(lines))
    print(json.dumps({"early_correct":model_rows[3]["correct"],"early_accuracy":model_rows[3]["accuracy"],"primary":primary,"report":str(REPORT)},indent=2),flush=True)

def main():
    global OUT, EARLY, REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-run", type=Path, default=ROOT / "outputs/final_early8_seed0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    EARLY, OUT, REPORT = args.early_run.resolve(), args.output_dir.resolve(), args.report.resolve()
    if OUT.exists() or REPORT.exists():
        raise ValueError("Refusing to overwrite existing output/report")
    if not OUT.is_relative_to(ROOT) or REPORT.parent != ROOT / "docs":
        raise ValueError("Use an output within this checkout and a fresh report in docs/")
    required = ("train_config.json", "train_log.jsonl", "eval_scienceqa/predictions.jsonl", "eval_scienceqa/metrics.json")
    missing = [name for name in required if not (EARLY / name).is_file()]
    if missing:
        parser.error("Early evaluation is pending or incomplete; no comparison written. Missing: " + ", ".join(missing))
    final_comparison()


if __name__ == "__main__":
    main()
