#!/usr/bin/env python3
"""Matched ScienceQA attention-knockout window sensitivity (prepare/measure).

Uses the unchanged InformationFlow.py functions for model inputs, token spans,
generation, and attention masks. No adapters are loaded and no training occurs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/information_flow/llava_v1_5_7b/ScienceQA/window_sensitivity/train_correct_n200_seed0"
SOURCE = ROOT / "output/information_flow/llava_v1_5_7b/ScienceQA/train_correct"
FLOWS = {"Image___Question": "Image->Question", "Question___Last": "Question->Last"}
sys.path.insert(0, str(ROOT))


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def record(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def window_expression():
    tree = ast.parse((ROOT / "InformationFlow.py").read_text())
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "layerlist" for t in node.targets)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one original layerlist expression")
    node = matches[0]
    return compile(ast.fix_missing_locations(ast.Expression(node.value)), "InformationFlow.py", "eval"), node


def actual_window(center, width):
    # Evaluate the original AST, so a future implementation change cannot silently
    # leave this boundary audit or the measurement runner using a different rule.
    code, _ = window_expression()
    return eval(code, {"layer": center, "args": SimpleNamespace(window=width),
                       "model": SimpleNamespace(config=SimpleNamespace(num_hidden_layers=32))})


def source_rows():
    result, paths = {}, {}
    for flow in FLOWS:
        (path,) = SOURCE.glob(f"{flow}/*_window9_{flow}.csv")
        paths[flow] = path
        by_question = defaultdict(list)
        for row in read_csv(path):
            by_question[row["question_id"]].append(row)
        result[flow] = by_question
    return result, paths


def prepare(args):
    out = args.output_dir
    sources, paths = source_rows()
    eligible = []
    calibration_path = ROOT / "datasets/SQA_train_correct_ScienceQA.csv"
    calibration = {r["question_id"]: r for r in read_csv(calibration_path)}
    for qid in sorted(set.intersection(*(set(v) for v in sources.values()))):
        if qid not in calibration:
            continue
        row = calibration[qid]
        valid = True
        for by_question in sources.values():
            rows = by_question[qid]
            valid &= len(rows) == 32 and {int(r["layer"]) for r in rows} == set(range(32))
            valid &= all(r["goden answer"] == r["predicted answer"] == row["answer"]
                         and r["image"] == row["img_id"] and r["question"] == row["question"] for r in rows)
        valid &= (ROOT / "datasets/scienceqa/images" / row["img_id"]).is_file()
        if valid:
            eligible.append(qid)
    ordered = sorted(eligible, key=lambda q: (hashlib.sha256(f"flow-window-sensitivity-v1|{args.seed}|{q}".encode()).hexdigest(), q))
    selected = ordered[:args.n]
    if len(selected) != args.n:
        raise ValueError(f"Need {args.n} complete correct examples; found {len(selected)}")
    # Declared before any window-3 result exists. Evaluate top-k stability with
    # paired question bootstraps, preserving dependence across flows and windows.
    criteria = {"meaningful_difference": "top16 overlap <= 13 with both matched window9 top16 and historical layers 0..15",
                "maximum_top16_overlap_for_difference": 13,
                "stability": "window3 top16 bootstrap median overlap >= 14/16, fifth percentile >= 12/16, and each fixed disjoint half overlap >= 14/16 with its full-sample top16",
                "bootstrap_replicates": 2000, "bootstrap_seed": 0,
                "minimum_bootstrap_median_overlap": 14, "minimum_bootstrap_fifth_percentile_overlap": 12,
                "minimum_half_overlap": 14,
                "note": "Predeclared practical screening thresholds, not a significance test; no downstream test-set outcomes used"}
    manifest = {"method": "matched_attention_knockout_window_sensitivity", "n": args.n, "seed": args.seed,
                "eligible_complete_correct_questions": len(eligible),
                "selection_rule": "ascending SHA256('flow-window-sensitivity-v1|seed|question_id'), tie by ID; first n",
                "question_ids": selected, "subject_counts": dict(Counter(calibration[q]["subject"] for q in selected)),
                "calibration_source": record(calibration_path), "window9_sources": {k: record(v) for k, v in paths.items()},
                "measurement_code": [record(ROOT / p) for p in ["InformationFlow.py", "methods.py", "utils.py", "LLaVA-NeXT/llava/model/builder.py"]],
                "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "inference_settings": {"model_path": "liuhaotian/llava-v1.5-7b", "model_base": None,
                    "conv_mode": "vicuna_v1", "dtype": "float16 (original information-flow builder default; not the PEFT bf16 loader)",
                    "attn_implementation_argument": None, "temperature": 0, "top_p": None, "num_beams": 1,
                    "base_max_new_tokens": 128, "knockout_max_new_tokens": 1, "use_cache": True,
                    "batch_size": 1, "image_aspect_ratio_override": None,
                    "prompt_and_preprocessing": "unchanged InformationFlow.CustomDataset; source prompt column; original token-range helper"},
                "criteria": criteria}
    target = out / "subset_manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("Frozen manifest differs; choose a new output directory")
    save_json(target, manifest)
    write_csv(out / "calibration_questions.csv", [calibration[q] for q in selected])
    for flow in FLOWS:
        write_csv(out / "window9" / f"{flow}.csv", [r for q in selected for r in sources[flow][q]])
    windows = [{"center": center, "window": width, "layers": actual_window(center, width),
                "actual_width": len(actual_window(center, width))} for width in (9, 3) for center in range(32)]
    _, node = window_expression()
    save_json(out / "boundary_audit.json", {"implementation": record(ROOT / "InformationFlow.py"),
              "source_line": node.lineno, "expression": ast.unparse(node.value), "windows": windows,
              "center4_to_center5_window9": {"removed": [0], "added": [9], "both_widths": 9},
              "interpretation": "The sharp change aligns with layer0 leaving the window. Overlapping nonlinear interventions cannot attribute it uniquely to layer0; layer9 also enters. Boundary windows are clipped rather than shifted to retain constant width."})
    save_json(out / "status.json", {"state": "prepared", "window9": "reused_matched_200", "window3": "not_measured", "training": "not_submitted"})
    print(json.dumps({"n": args.n, "eligible": len(eligible), "subjects": manifest["subject_counts"], "output": str(out)}, indent=2))


def measure(args):
    import copy
    import torch
    import transformers
    import InformationFlow as original
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    out, flow = args.output_dir, args.flow
    manifest = json.loads((out / "subset_manifest.json").read_text())
    for item in manifest["measurement_code"]:
        if record(ROOT / item["path"])["sha256"] != item["sha256"]:
            raise ValueError(f"Frozen measurement source changed: {item['path']}")
    questions = read_csv(out / "calibration_questions.csv")
    assert [r["question_id"] for r in questions] == manifest["question_ids"]
    reference = {(r["question_id"], int(r["layer"])): r for r in read_csv(out / "window9" / f"{flow}.csv")}
    destination = out / "window3" / f"{flow}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = read_csv(destination) if destination.exists() else []
    counts = Counter(r["question_id"] for r in existing)
    complete = {q for q, n in counts.items() if n == 32 and {int(r["layer"]) for r in existing if r["question_id"] == q} == set(range(32))}
    if set(counts) - complete:
        # A killed job may leave an incomplete last question; retain only fully
        # measured questions, which can be checked against all 32 required layers.
        existing = [r for r in existing if r["question_id"] in complete]
        write_csv(destination, existing, list(reference.values())[0].keys())
    assert complete <= set(manifest["question_ids"])
    todo = [r for r in questions if r["question_id"] not in complete]
    if not todo:
        print(f"Already complete: {flow}: {len(complete)}/{manifest['n']}")
        return
    torch.manual_seed(0)
    model_path = manifest["inference_settings"]["model_path"]
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, device_map="auto", attn_implementation=None)
    model.eval()
    assert len(model.model.layers) == 32
    dataset = {r["question_id"]: r for r in questions}
    rows = [{**r, "q_id": r["question_id"]} for r in todo]
    loader = original.create_data_loader(rows, str(ROOT / "datasets/scienceqa/images"), 1, 2,
                                         tokenizer, processor, model.config, "ScienceQA", "vicuna_v1")
    original_args = SimpleNamespace(certain_part_image=False)
    parity_path = out / "window3" / f"{flow}_parity.json"
    parity = json.loads(parity_path.read_text()) if parity_path.exists() else []
    parity = [r for r in parity if r["question_id"] in complete]
    meta_path = out / "window3" / f"{flow}_run.json"
    run = {"state": "running", "flow": flow, "window": 3, "started_at_unix": time.time(),
           "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"), "node": platform.node(),
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "transformers": transformers.__version__,
           "actual_model_dtype": str(model.dtype), "actual_attention_implementation": model.config._attn_implementation,
           "parameters_trainable_during_inference": "no gradients; no adapters", "resumed_complete_questions": len(complete),
           "parity_tolerance": {"rtol": 0.001, "atol": 0.00001}, "runner": record(Path(__file__))}
    save_json(meta_path, run)
    fields = list(next(iter(reference.values())))
    started = time.monotonic()
    with destination.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not existing:
            writer.writeheader()
        for attempt_index, (item, row) in enumerate(zip(loader, rows)):
            ids, images, sizes, prompts, mask = item
            qid = row["question_id"]
            inputs = {"inputs": ids.cuda(), "images": [x.cuda() for x in images], "image_sizes": sizes,
                      "do_sample": False, "temperature": 0, "top_p": None, "num_beams": 1,
                      "max_new_tokens": 128, "use_cache": True, "return_dict_in_generate": True,
                      "output_scores": True, "pad_token_id": tokenizer.eos_token_id}
            base, predicted, token, shape = original.run_original(model, inputs, tokenizer, model_name, args=original_args)
            if predicted != row["answer"]:
                raise ValueError(f"Previously correct calibration question changed prediction: {qid}: {predicted!r}")
            saved_base = float(reference[(qid, 0)]["base_score_first"])
            check = {"question_id": qid, "base_saved": saved_base, "base_current": base,
                     "base_abs_diff": abs(base - saved_base), "window9_checks": []}
            if not math.isclose(base, saved_base, rel_tol=0.001, abs_tol=0.00001):
                save_json(out / "window3" / f"{flow}_parity_failure.json", check)
                raise ValueError(f"Base-probability parity failed for {qid}: saved={saved_base}, current={base}")
            a, b = FLOWS[flow].split("->")
            source = original.blockdesc2range(a, dataset, qid, inputs["inputs"], shape, tokenizer, model_name)
            target = original.blockdesc2range(b, dataset, qid, inputs["inputs"], shape, tokenizer, model_name)
            pairs = [(t, s) for s in source for t in target]
            inputs["max_new_tokens"] = 1

            def knockout(center, width):
                config = {layer: copy.deepcopy(pairs) for layer in actual_window(center, width)}
                return original.trace_with_attn_block_llava(model, inputs, config, token, FLOWS[flow], model_name).cpu().item()

            # Repeat six saved window-9 values at the start of every allocation,
            # including resumed runs and the boundary transition under study.
            if attempt_index < 2:
                for center in (0, 4, 5):
                    current = knockout(center, 9)
                    saved = float(reference[(qid, center)]["new_score_first"])
                    check["window9_checks"].append({"center": center, "saved": saved, "current": current,
                                                     "abs_diff": abs(saved - current)})
                    if not math.isclose(saved, current, rel_tol=0.001, abs_tol=0.00001):
                        save_json(out / "window3" / f"{flow}_parity_failure.json", check)
                        raise ValueError(f"Window9 knockout parity failed for {qid} at center {center}")
            measured = []
            for center in range(32):
                new = knockout(center, 3)
                measured.append({"question_id": qid, "image": row["img_id"], "goden answer": row["answer"],
                    "predicted answer": predicted, "question": row["question"], "block_desc": FLOWS[flow], "layer": center,
                    "base_score_first": base, "new_score_first": new, "relative diff first": (new - base) * 100.0 / base})
            writer.writerows(measured)
            handle.flush()
            parity.append(check)
            complete.add(qid)
            save_json(parity_path, parity)
            run.update(completed_questions=len(complete), elapsed_seconds=time.monotonic() - started)
            save_json(meta_path, run)
            print(f"{flow}: {len(complete)}/{manifest['n']} questions, {run['elapsed_seconds']:.1f}s", flush=True)
    run.update(state="complete", completed_at_unix=time.time())
    save_json(meta_path, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--n", type=int, default=200)
    prep.add_argument("--seed", type=int, default=0)
    measurement = sub.add_parser("measure")
    measurement.add_argument("--flow", choices=list(FLOWS), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "measure": measure}[args.command](args)


if __name__ == "__main__":
    main()
