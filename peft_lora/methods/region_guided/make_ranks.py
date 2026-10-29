#!/usr/bin/env python3
"""Offline analysis for the ScienceQA two-region LoRA pilot.

By default, reuse the repository's ScienceQA train_correct flow aggregation and
existing flow-pruned rank map. Export plots, scores, a trainer-compatible rank
map, and provenance. --validate-adapter counts real PEFT adapter tensors on the
full model architecture without loading pretrained weights or requiring a GPU.
This script never submits jobs or trains a model.

Usage (from the repository root):
  python peft_lora/methods/region_guided/make_ranks.py [--validate-adapter]
Ranks go to configs/ranks/region_pruned16_scienceqa.json; analysis artifacts
go to output/region_pilot/scienceqa/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
FLOW_DIR = ROOT / "output/information_flow/llava_v1_5_7b"
EXISTING_RANKS = ROOT / "configs/ranks/flow_pruned16_scienceqa.json"
BASELINE = ROOT / "output/peft/flow_pruned16_scienceqa/scienceqa_train_llava/seed0"
RANK_OUTPUT = ROOT / "configs/ranks/region_pruned16_scienceqa.json"


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_record(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def repository_scores() -> tuple[dict[int, float], list[dict], dict]:
    # Reuse the exact aggregation that generated flow_pruned16_scienceqa.json.
    sys.path.insert(0, str(ROOT / "peft_lora"))
    flow = import_file("pilot_flow_ranks", ROOT / "peft_lora/methods/flow_guided/make_ranks.py")
    curves, sources = {}, []
    for name in ("Image___Question", "Question___Last"):
        path = Path(flow.find_csv(str(FLOW_DIR), "ScienceQA", "train_correct", name, 9))
        curves[name] = flow.flow_curve(str(path))
        with path.open() as f:
            counts = {i: 0 for i in range(32)}
            for row in csv.DictReader(f):
                counts[int(row["layer"])] += 1
        if set(counts.values()) == {0} or min(counts.values()) == 0:
            raise ValueError(f"Missing layer observations in {path}")
        sources.append({**file_record(path), "observations_by_layer": counts})
    importance = flow.layer_importance(list(curves.values()))
    return dict(enumerate(importance)), sources, curves


def validate_adapter(rank_path: Path, output: Path) -> None:
    """Insert adapters using the unchanged trainer; count actual parameter tensors.

    Meta tensors retain the checkpoint's exact architecture and tensor shapes
    without allocating ~14 GB of base weights. Materialize the adapter matrices
    alone on CPU so the trainable count also covers allocated parameter storage.
    """
    import torch
    import peft
    import transformers
    from huggingface_hub import hf_hub_download
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    from transformers import CLIPVisionModel

    sys.path.insert(0, str(ROOT / "peft_lora"))
    trainer = import_file("pilot_trainer", ROOT / "peft_lora/train_lora.py")
    config_path = Path(hf_hub_download("liuhaotian/llava-v1.5-7b", "config.json", local_files_only=True))
    config = LlavaConfig(**json.loads(config_path.read_text()))
    config.delay_load = True
    with torch.device("meta"):
        model = LlavaLlamaForCausalLM(config)
        tower = model.get_vision_tower()
        tower.vision_tower = CLIPVisionModel(tower.config)
        tower.is_loaded = True
    base_layers = tuple(id(layer) for layer in model.get_model().layers)
    model.requires_grad_(False)
    ranks = json.loads(rank_path.read_text())
    lora_config, expected = trainer.make_peft_config(ranks, 388)
    model = peft.get_peft_model(model, lora_config)
    trainer.check_ranks(model, expected)
    assert tuple(id(layer) for layer in model.get_model().layers) == base_layers
    modules = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            assert name.startswith("base_model.model.model.layers."), name
            module.lora_A["default"].to_empty(device="cpu")
            module.lora_B["default"].to_empty(device="cpu")
            module.reset_lora_parameters("default", True)
            modules.append({"name": name, "rank": module.r["default"],
                            "alpha": module.lora_alpha["default"],
                            "A_shape": list(module.lora_A["default"].weight.shape),
                            "B_shape": list(module.lora_B["default"].weight.shape)})
    params = [{"name": name, "shape": list(p.shape), "numel": p.numel(), "device": str(p.device)}
              for name, p in model.named_parameters() if p.requires_grad]
    count = sum(p["numel"] for p in params)
    assert count == 2_097_152, count
    assert len(modules) == 32 and len(params) == 64
    assert all(".lora_" in p["name"] and p["device"] == "cpu" for p in params)
    result = {"status": "passed", "rank_file": file_record(rank_path.resolve()),
              "validation_mode": "Full checkpoint architecture on meta; actual PEFT adapter tensors allocated and initialized on CPU; pretrained weights not loaded; no forward/training run",
              "checkpoint_config": {"snapshot": config_path.parent.name, "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
              "base_transformer_layers_before": len(base_layers), "base_transformer_layers_after": len(model.get_model().layers),
              "base_layer_objects_preserved": True, "trainable_parameters": count,
              "expected_trainable_parameters": 2_097_152, "adapter_modules": modules,
              "trainable_tensors": params,
              "versions": {"torch": torch.__version__, "peft": peft.__version__, "transformers": transformers.__version__}}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Actual trainable adapter parameters: {count:,}; 32 base layers preserved; 32 q/v adapter modules.")


def _to_number(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Score must be finite; got {value!r}")
    return number


def load_scores(path: Path) -> dict[int, float]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            for name in ("scores", "flow_scores", "layer_scores", "data"):
                if name in payload:
                    payload = payload[name]
                    break
        if isinstance(payload, list):
            if all(isinstance(x, (int, float)) for x in payload):
                scores = {i: _to_number(x) for i, x in enumerate(payload)}
            elif all(isinstance(x, dict) for x in payload):
                scores = {}
                for item in payload:
                    l = item.get("layer", item.get("layer_idx", item.get("layer_id")))
                    score = next((item[k] for k in ("score", "flow_score", "importance", "value") if k in item), None)
                    if l is None or score is None:
                        raise ValueError(f"Cannot parse layer/score from {item!r}")
                    l = int(l)
                    if l in scores:
                        raise ValueError(f"Duplicate layer {l}")
                    scores[l] = _to_number(score)
            else:
                raise ValueError("JSON list must be numbers or layer-score dictionaries")
        elif isinstance(payload, dict):
            scores = {int(key): _to_number(value) for key, value in payload.items()}
        else:
            raise ValueError("Unexpected JSON structure")
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV is missing a header")
            layer_key = next((k for k in ("layer", "layer_idx", "layer_id") if k in reader.fieldnames), None)
            score_key = next((k for k in ("score", "flow_score", "importance", "value") if k in reader.fieldnames), None)
            if not layer_key or not score_key:
                raise ValueError(f"Expected layer + score columns, found {reader.fieldnames}")
            scores = {}
            for row in reader:
                l = int(row[layer_key])
                if l in scores:
                    raise ValueError(f"Duplicate layer {l}")
                scores[l] = _to_number(row[score_key])
    else:
        raise ValueError("Expected a JSON or CSV input")

    expected = set(range(32))
    if set(scores) != expected:
        raise ValueError(f"Expected exactly layers 0..31; missing={sorted(expected-set(scores))}; extra={sorted(set(scores)-expected)}")
    return scores


def parse_existing_layers(value: str | None) -> list[int] | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        x = json.loads(path.read_text())
        if isinstance(x, dict):
            x = x.get("selected_layers", x.get("layers", x.get("top16", x)))
            if isinstance(x, dict):
                x = sorted({int(re.fullmatch(r"layers\.(\d+)\.self_attn\.[qv]_proj", k).group(1))
                            if k.startswith("layers.") else int(k) for k, v in x.items() if int(v) > 0})
        selection = [int(i) for i in x]
    else:
        selection = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(selection) != 16 or len(set(selection)) != 16 or not all(0 <= i < 32 for i in selection):
        raise ValueError(f"Expected 16 distinct indices in 0..31, got {selection}")
    return sorted(selection)


def select_two_regions(importance: dict[int, float], length: int = 8, gap: int = 1) -> tuple[list[int], list[tuple[int, int]], float]:
    """Select two disjoint contiguous length=8 windows, with at least `gap` omitted layers.

    Maximize sum of per-layer importance, tie-break by smallest (start1, start2).
    """
    if length < 1 or gap < 1:
        raise ValueError("Region length and gap must both be positive")
    candidates = []
    for a in range(33 - length):
        for b in range(a + length + gap, 33 - length):
            total = sum(importance[i] for i in range(a, a + length)) + sum(importance[i] for i in range(b, b + length))
            candidates.append((-total, a, b))
    if not candidates:
        raise ValueError("Cannot fit two regions with given length/gap")
    _, a, b = min(candidates)
    selected = list(range(a, a + length)) + list(range(b, b + length))
    return selected, [(a, a + length - 1), (b, b + length - 1)], -min(candidates)[0]


def plot_scores(importance: dict[int, float], top16: list[int], regions: list[tuple[int, int]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = list(range(32))
    y = [importance[i] for i in x]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.plot(x, y, marker="o", markersize=4, linewidth=1.7, label="Flow importance")
    ax.scatter(top16, [importance[i] for i in top16], color="#dc7b16", marker="*", s=115, zorder=3, label="Existing top-16")
    for i, (start, end) in enumerate(regions):
        ax.axvspan(start - .45, end + .45, color="#249f80", alpha=.16, label="Selected contiguous regions" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xlim(-.6, 31.6)
    ax.set_xlabel("Transformer layer index (0–31)")
    ax.set_ylabel("Flow importance (higher means more important)")
    ax.set_title("Flow scores: independent top-16 vs two contiguous 8-layer regions")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, help="Optional external JSON/CSV; default: derive the real ScienceQA importance")
    parser.add_argument("--important-direction", choices=("high", "low"),
                        help="Whether the highest or lowest *raw* score indicates greatest importance. Verify semantics from code.")
    parser.add_argument("--existing-top16", help="Comma-separated layer indices OR path to JSON existing selection; omit to infer from raw scores")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/region_pilot/scienceqa")
    parser.add_argument("--rank-output", type=Path, default=RANK_OUTPUT,
                        help="Trainer-compatible module rank map (default: configs/ranks/region_pruned16_scienceqa.json)")
    parser.add_argument("--min-gap", type=int, default=1, help="Minimum number of skipped layers between 8-layer regions (default: 1)")
    parser.add_argument("--validate-adapter", action="store_true", help="Validate actual adapter parameter tensors without a GPU; needs the existing llava environment and cached configs")
    args = parser.parse_args()
    if args.min_gap < 1:
        parser.error("--min-gap must be >= 1")
    curves, sources = {}, []
    if args.scores:
        if args.important_direction is None:
            parser.error("External scores require --important-direction")
        raw = load_scores(args.scores)
        sources = [file_record(args.scores.resolve())]
    else:
        if args.important_direction == "low":
            parser.error("The repository's aggregated importance is higher-is-more-important")
        raw, sources, curves = repository_scores()
        args.important_direction = "high"
        args.existing_top16 = args.existing_top16 or str(EXISTING_RANKS)
    importance = {i: (raw[i] if args.important_direction == "high" else -raw[i]) for i in range(32)}
    inferred_top16 = sorted(sorted(range(32), key=lambda i: (-importance[i], i))[:16])
    provided_top16 = parse_existing_layers(args.existing_top16)
    top16 = provided_top16 or inferred_top16
    if not args.scores and top16 != inferred_top16:
        raise ValueError("Repository importance does not reproduce the existing flow-pruned allocation")
    selected, regions, total = select_two_regions(importance, gap=args.min_gap)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlap = sorted(set(selected) & set(top16))
    result = {
        "method": "two_contiguous_regions_max_total_flow",
        "score_file": str(args.scores) if args.scores else "flow_scores.json (derived from source CSVs)",
        "score_sources": sources,
        "importance_definition": "sum over Image->Question and Question->Last of max(0, -mean(relative diff first over train_correct questions)); knockout window=9; higher is more important" if not args.scores else "external scores transformed according to score_direction",
        "score_direction": args.important_direction,
        "source_top16": "existing run configuration" if provided_top16 else "inferred from scores; verify against trainer",
        "top16_existing": top16,
        "top16_inferred": inferred_top16,
        "region_bounds_inclusive": regions,
        "region_selected_layers": selected,
        "overlap_count_with_top16": len(overlap),
        "overlap_fraction": len(overlap) / 16,
        "overlap_layers": overlap,
        "removed_layers": sorted(set(top16) - set(selected)),
        "added_layers": sorted(set(selected) - set(top16)),
        "min_gap": args.min_gap,
        "tie_break": "lexicographically smallest (first_start, second_start) among equal objective values",
        "jaccard_vs_top16": len(overlap) / len(set(selected) | set(top16)),
        "identical_to_top16": set(selected) == set(top16),
        "nearly_identical_to_top16": len(overlap) >= 14,
        "training_decision": "skip_nearly_identical" if len(overlap) >= 14 else "eligible_for_one_seed0_pilot",
        "near_identity_rule": "At least 14/16 shared layers (at most two substitutions); decided before any training",
        "importance_sum_region": total,
        "importance_sum_top16": sum(importance[i] for i in top16),
        "lora_rank_on_selected_layers": 8,
        "number_of_lora_layers": 16,
        "expected_adapter_parameters": 2097152,
        "rank_config": str(args.rank_output.resolve().relative_to(ROOT)) if args.rank_output.resolve().is_relative_to(ROOT) else str(args.rank_output.resolve()),
        "note": "The rank_config uses the project's trainer format. make_peft_config filters rank-zero entries before constructing exact target module names, so omitted layers receive no adapters. No base layers are removed.",
    }
    result["importance_retained_fraction"] = total / result["importance_sum_top16"] if result["importance_sum_top16"] else None
    (args.output_dir / "region_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "rank_by_layer_manifest.json").write_text(json.dumps({str(i): (8 if i in selected else 0) for i in range(32)}, indent=2) + "\n")
    rank_path = args.rank_output
    rank_path.parent.mkdir(parents=True, exist_ok=True)
    rank_path.write_text(json.dumps({f"layers.{i}.self_attn.{proj}": (8 if i in selected else 0)
                                     for i in range(32) for proj in ("q_proj", "v_proj")}, indent=2) + "\n")
    (args.output_dir / "flow_scores.json").write_text(json.dumps({"scores": [raw[i] for i in range(32)],
             "important_direction": args.important_direction, "sources": sources}, indent=2) + "\n")
    with (args.output_dir / "flow_scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "score", *curves, "existing_top16", "region_selected"])
        for i in range(32):
            writer.writerow([i, raw[i], *(curve[i] for curve in curves.values()), int(i in top16), int(i in selected)])
    if not args.scores:
        config = json.loads((BASELINE / "train_config.json").read_text())
        (args.output_dir / "baseline_seed0_train_config.json").write_text(json.dumps(config, indent=2) + "\n")
        planned = {**config, "ranks": result["rank_config"], "out_dir": None}
        (args.output_dir / "planned_seed0_train_config.json").write_text(json.dumps(planned, indent=2) + "\n")
        metrics = json.loads((BASELINE / "eval_scienceqa/metrics.json").read_text())
        (args.output_dir / "baseline_seed0_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (args.output_dir / "provenance.json").write_text(json.dumps({
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "files": [file_record(p) for p in [EXISTING_RANKS, BASELINE / "train_config.json", BASELINE / "eval_scienceqa/metrics.json",
                      ROOT / "datasets/scienceqa/train_llava.json", ROOT / "datasets/scienceqa/test_questions.jsonl",
                      ROOT / "peft_lora/methods/flow_guided/make_ranks.py", ROOT / "peft_lora/train_lora.py", ROOT / "peft_lora/eval_gqa.py", ROOT / "peft_lora/common.py",
                      Path(__file__).resolve(), rank_path.resolve()]]}, indent=2) + "\n")
    plot_scores(importance, top16, regions, args.output_dir / "flow_regions_vs_top16.png")
    if args.validate_adapter:
        validate_adapter(rank_path, args.output_dir / "adapter_validation.json")
    print(json.dumps({k: result[k] for k in ("top16_existing", "region_bounds_inclusive", "region_selected_layers", "overlap_count_with_top16", "identical_to_top16", "jaccard_vs_top16", "expected_adapter_parameters")}, indent=2))
    if result["identical_to_top16"]:
        print("\nWARNING: Region allocation matches existing top-16 exactly; this pilot would be redundant.")
    elif result["overlap_count_with_top16"] >= 14:
        print("\nCAUTION: Region allocation differs in only 1–2 selected layers; likely a low-information pilot.")
    print(f"\nSaved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
