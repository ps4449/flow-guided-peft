"""Check whether attention can stand in for the Image->Question knockout, before using it in a training loss.

The knockout graphs (InformationFlow.py) show that on wrongly answered questions, the question tokens take in
less from the image in layers 5-15. A training loss cannot run knockouts, but it can use attention. This script
measures, with the zero-shot model, how much attention the question tokens give the image tokens in every layer
(summed over the 576 image tokens, averaged over heads and question tokens), on the same right/wrong files:
  1. Per layer: average image attention on correct vs wrong answers, with a 95% bootstrap interval of the gap.
     If attention is a good stand-in, it should also be lower on wrong answers in layers 5-15.
  2. Per question: Spearman correlation between image attention and the knockout drop (layers 5-15).
     Negative = more attention goes with a bigger drop when the link is cut, i.e. attention tracks the knockout.

Only the questions in the knockout CSVs are used, so both measures cover the same questions.
Writes output/information_flow/<model>/<dataset>/attention_check/{correct,wrong}.csv and summary.json.

Usage (GPU, from the repo root; see scripts/tc2_attention_check.sh):
  python scripts/check_attention_proxy.py --correct datasets/GQA_train_correct_All.csv --wrong datasets/GQA_train_wrong_All.csv
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "peft_lora"))
from common import build_prompt, load_image, load_model, tokenize  # noqa: E402
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402

GQA_SUFFIX = " \nAnswer the question using a single word or phrase."  # as in InformationFlow.py
FLOW_LAYERS = range(5, 16)  # where the knockout gap is


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def knockout_effects(path):
    """{question_id: mean knockout change over FLOW_LAYERS} from an InformationFlow.py CSV."""
    per_question = defaultdict(dict)
    for row in read_csv(path):
        per_question[row["question_id"]][int(row["layer"])] = float(row["relative diff first"])
    return {q: sum(layers[l] for l in FLOW_LAYERS) / len(FLOW_LAYERS) for q, layers in per_question.items()}


def question_positions(input_ids, question, tokenizer, image_dim):
    """Positions of the question tokens in the model's input after the image token is expanded."""
    image_pos = (input_ids == IMAGE_TOKEN_INDEX).nonzero()[0].item()
    after = input_ids[image_pos + 1:].tolist()
    toks = tokenizer.convert_ids_to_tokens(after)
    text = "".join(toks).replace("▁", " ")
    start_char = text.index(question)  # ValueError if not found (e.g. byte tokens); the caller skips it
    loc, start, end = 0, None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if start is None and loc > start_char:
            start = i
        if loc >= start_char + len(question):
            end = i + 1
            break
    return image_pos, [image_pos + image_dim + i for i in range(start, end)]


def image_attention(model, tokenizer, image_processor, row, image_folder):
    """Per layer: attention from the question tokens to the image tokens (sum over image, mean over heads/tokens)."""
    human = DEFAULT_IMAGE_TOKEN + "\n" + (row.get("prompt") or row["question"] + GQA_SUFFIX)
    input_ids = tokenize(build_prompt(human), tokenizer)
    image_file = row["img_id"] if os.path.splitext(row["img_id"])[1] else row["img_id"] + ".jpg"
    image, size = load_image(os.path.join(image_folder, image_file), image_processor, model.config)
    with torch.inference_mode():
        out = model(input_ids=input_ids.unsqueeze(0).cuda(), images=image.unsqueeze(0).cuda(), image_sizes=[size],
                    modalities=["image"], output_attentions=True, return_dict=True)
    if out.attentions is None:
        sys.exit("The model returned no attention weights; load it with eager attention")
    total = out.attentions[0].shape[-1]
    image_dim = total - (len(input_ids) - 1)
    image_pos, q_pos = question_positions(input_ids, row["question"], tokenizer, image_dim)
    q_pos = torch.tensor(q_pos, device=out.attentions[0].device)
    return [layer_attn[0][:, q_pos, image_pos:image_pos + image_dim].float().sum(-1).mean().item()
            for layer_attn in out.attentions]


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correct", required=True)
    parser.add_argument("--wrong", required=True)
    parser.add_argument("--image-folder", default=None, help="default: datasets/gqa/images, or scienceqa for SQA_ files")
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--flow-dir", default="output/information_flow/llava_v1_5_7b")
    parser.add_argument("--limit", type=int, default=None, help="first N questions per file (for tests)")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    dataset = os.path.splitext(os.path.basename(args.correct))[0].split("_")[-1]  # All or ScienceQA
    image_folder = args.image_folder or ("datasets/scienceqa/images" if dataset == "ScienceQA" else "datasets/gqa/images")
    out_dir = os.path.join(args.flow_dir, dataset, "attention_check")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer, model, image_processor = load_model(args.model_path)
    model.eval()

    results = {}
    for label, path in [("correct", args.correct), ("wrong", args.wrong)]:
        knock = glob.glob(os.path.join(args.flow_dir, dataset, f"train_{label}", "Image___Question", "*.csv"))
        effects = knockout_effects(knock[0]) if knock else {}
        rows = [r for r in read_csv(path) if not effects or r["question_id"] in effects][: args.limit]
        per_question, skipped = {}, 0
        for i, row in enumerate(rows):
            try:
                per_question[row["question_id"]] = image_attention(model, tokenizer, image_processor, row, image_folder)
            except ValueError:
                skipped += 1
            if (i + 1) % 100 == 0:
                print(f"{label}: {i + 1}/{len(rows)}", flush=True)
        with open(os.path.join(out_dir, f"{label}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question_id", "layer", "image_attention"])
            for q, values in per_question.items():
                writer.writerows([q, layer, round(v, 6)] for layer, v in enumerate(values))
        results[label] = (per_question, effects, skipped)
        print(f"{label}: {len(per_question)} questions ({skipped} skipped: question tokens not found)", flush=True)

    right, wrong = results["correct"][0], results["wrong"][0]
    n_layers = len(next(iter(right.values())))
    rng = random.Random(0)
    summary = {"dataset": dataset, "n_correct": len(right), "n_wrong": len(wrong), "layers": []}
    print("\nImage attention of the question tokens (share of their attention that goes to the image)")
    print("layer  correct   wrong   gap(w-c)   95% CI of gap")
    for layer in range(n_layers):
        r = [v[layer] for v in right.values()]
        w = [v[layer] for v in wrong.values()]
        gap = sum(w) / len(w) - sum(r) / len(r)
        boots = sorted(sum(rng.choices(w, k=len(w))) / len(w) - sum(rng.choices(r, k=len(r))) / len(r)
                       for _ in range(args.n_boot))
        lo, hi = boots[int(0.025 * args.n_boot)], boots[int(0.975 * args.n_boot) - 1]
        mark = " *" if lo > 0 or hi < 0 else ""
        print(f"{layer:5d} {sum(r) / len(r):8.4f} {sum(w) / len(w):7.4f} {gap:9.4f}   [{lo:7.4f}, {hi:7.4f}]{mark}")
        summary["layers"].append({"layer": layer, "correct": sum(r) / len(r), "wrong": sum(w) / len(w),
                                  "gap": gap, "ci": [lo, hi]})

    # per question: does more image attention (layers 5-15) go with a bigger knockout drop?
    print(f"\nPer question, layers {FLOW_LAYERS.start}-{FLOW_LAYERS.stop - 1}: Spearman(image attention, knockout change)")
    for label in ("correct", "wrong"):
        per_question, effects, _ = results[label]
        common_ids = [q for q in per_question if q in effects]
        if not common_ids:
            print(f"  {label}: no knockout CSV found")
            continue
        att = [sum(per_question[q][l] for l in FLOW_LAYERS) / len(FLOW_LAYERS) for q in common_ids]
        rho = spearman(att, [effects[q] for q in common_ids])
        summary[f"spearman_{label}"] = rho
        print(f"  {label}: rho = {rho:.3f} over {len(common_ids)} questions (negative = attention tracks the knockout)")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_dir}/correct.csv, wrong.csv and summary.json")


if __name__ == "__main__":
    main()
