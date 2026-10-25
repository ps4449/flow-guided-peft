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

--method attn_grad scores attention x gradient instead (gradient of the first answer token's log-probability),
which measures how much the answer uses each link, not only how much the model looks; it writes to attention_check_attn_grad/.

Usage (GPU, from the repo root; see scripts/tc2_attention_check.sh):
  python scripts/check_attention_proxy.py --correct datasets/GQA_train_correct_All.csv --wrong datasets/GQA_train_wrong_All.csv
  ... --method attn_grad --labels wrong     # attention x gradient, wrong answers only
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


def image_attention(model, tokenizer, image_processor, row, image_folder, method="attention"):
    """Per layer: attention from the question tokens to the image tokens (sum over image, mean over heads/tokens).

    method="attn_grad" scores attention x gradient instead: each weight times the gradient of the log-probability
    of the model's first answer token (the token the knockout scores) with respect to it. Positive = this link
    raises the answer's probability, so like the knockout it measures use, not only looking.
    """
    human = DEFAULT_IMAGE_TOKEN + "\n" + (row.get("prompt") or row["question"] + GQA_SUFFIX)
    input_ids = tokenize(build_prompt(human), tokenizer)
    image_file = row["img_id"] if os.path.splitext(row["img_id"])[1] else row["img_id"] + ".jpg"
    image, size = load_image(os.path.join(image_folder, image_file), image_processor, model.config)
    inputs = dict(input_ids=input_ids.unsqueeze(0).cuda(), images=image.unsqueeze(0).cuda(), image_sizes=[size],
                  modalities=["image"], output_attentions=True, return_dict=True)
    if method == "attention":
        with torch.inference_mode():
            out = model(**inputs)
    else:
        with torch.enable_grad():
            out = model(**inputs)
            if out.attentions is not None:
                for layer_attn in out.attentions:
                    layer_attn.retain_grad()
                log_probs = out.logits[0, -1].float().log_softmax(-1)
                log_probs[log_probs.argmax()].backward()
    if out.attentions is None:
        sys.exit("The model returned no attention weights; load it with eager attention")
    total = out.attentions[0].shape[-1]
    image_dim = total - (len(input_ids) - 1)
    image_pos, q_pos = question_positions(input_ids, row["question"], tokenizer, image_dim)
    q_pos = torch.tensor(q_pos, device=out.attentions[0].device)
    scores = []
    for layer_attn in out.attentions:
        weights = layer_attn.detach() if method == "attention" else layer_attn.detach() * layer_attn.grad
        scores.append(weights[0][:, q_pos, image_pos:image_pos + image_dim].float().sum(-1).mean().item())
    return scores


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
    parser.add_argument("--method", choices=["attention", "attn_grad"], default="attention",
                        help="attn_grad: attention x gradient of the answer's log-probability")
    parser.add_argument("--labels", nargs="+", choices=["correct", "wrong"], default=["correct", "wrong"],
                        help="which files to run; the correct-vs-wrong table needs both")
    args = parser.parse_args()

    dataset = os.path.splitext(os.path.basename(args.correct))[0].split("_")[-1]  # All or ScienceQA
    image_folder = args.image_folder or ("datasets/scienceqa/images" if dataset == "ScienceQA" else "datasets/gqa/images")
    out_dir = os.path.join(args.flow_dir, dataset, "attention_check" + ("" if args.method == "attention" else "_attn_grad"))
    os.makedirs(out_dir, exist_ok=True)

    tokenizer, model, image_processor = load_model(args.model_path)
    model.eval()
    if args.method == "attn_grad":
        # no weight gradients; make the image features need a gradient so the attention weights get one
        model.requires_grad_(False)
        model.get_model().mm_projector.register_forward_hook(lambda module, inp, out: out.requires_grad_())

    results = {}
    paths = {"correct": args.correct, "wrong": args.wrong}
    for label in args.labels:
        path = paths[label]
        knock = glob.glob(os.path.join(args.flow_dir, dataset, f"train_{label}", "Image___Question", "*.csv"))
        effects = knockout_effects(knock[0]) if knock else {}
        rows = [r for r in read_csv(path) if not effects or r["question_id"] in effects][: args.limit]
        per_question, skipped = {}, 0
        for i, row in enumerate(rows):
            try:
                per_question[row["question_id"]] = image_attention(model, tokenizer, image_processor, row, image_folder,
                                                                   args.method)
            except ValueError:
                skipped += 1
            if (i + 1) % 100 == 0:
                print(f"{label}: {i + 1}/{len(rows)}", flush=True)
        with open(os.path.join(out_dir, f"{label}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question_id", "layer", "image_" + args.method])
            for q, values in per_question.items():
                writer.writerows([q, layer, f"{v:.6g}"] for layer, v in enumerate(values))
        results[label] = (per_question, effects, skipped)
        print(f"{label}: {len(per_question)} questions ({skipped} skipped: question tokens not found)", flush=True)

    summary = {"dataset": dataset, "method": args.method, "layers": []}
    summary.update({f"n_{label}": len(results[label][0]) for label in results})
    if len(results) == 2:
        compare(results["correct"][0], results["wrong"][0], args, summary)

    # per question: does more image attention (layers 5-15) go with a bigger knockout drop?
    print(f"\nPer question, layers {FLOW_LAYERS.start}-{FLOW_LAYERS.stop - 1}: Spearman(image {args.method}, knockout change)")
    for label in results:
        per_question, effects, _ = results[label]
        common_ids = [q for q in per_question if q in effects]
        if not common_ids:
            print(f"  {label}: no knockout CSV found")
            continue
        att = [sum(per_question[q][l] for l in FLOW_LAYERS) / len(FLOW_LAYERS) for q in common_ids]
        rho = spearman(att, [effects[q] for q in common_ids])
        summary[f"spearman_{label}"] = rho
        print(f"  {label}: rho = {rho:.3f} over {len(common_ids)} questions (negative = it tracks the knockout)")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {', '.join(f'{out_dir}/{label}.csv' for label in results)} and summary.json")


def compare(right, wrong, args, summary):
    """Per layer: mean on correct vs wrong answers, with a 95% bootstrap interval of the gap."""
    n_layers = len(next(iter(right.values())))
    rng = random.Random(0)
    print(f"\nImage {args.method} of the question tokens (attention: share of their attention that goes to the image)")
    print("layer     correct       wrong    gap(w-c)   95% CI of gap")
    for layer in range(n_layers):
        r = [v[layer] for v in right.values()]
        w = [v[layer] for v in wrong.values()]
        gap = sum(w) / len(w) - sum(r) / len(r)
        boots = sorted(sum(rng.choices(w, k=len(w))) / len(w) - sum(rng.choices(r, k=len(r))) / len(r)
                       for _ in range(args.n_boot))
        lo, hi = boots[int(0.025 * args.n_boot)], boots[int(0.975 * args.n_boot) - 1]
        mark = " *" if lo > 0 or hi < 0 else ""
        print(f"{layer:5d} {sum(r) / len(r):10.3e} {sum(w) / len(w):10.3e} {gap:10.3e}   [{lo:10.3e}, {hi:10.3e}]{mark}")
        summary["layers"].append({"layer": layer, "correct": sum(r) / len(r), "wrong": sum(w) / len(w),
                                  "gap": gap, "ci": [lo, hi]})


if __name__ == "__main__":
    main()
