"""Baseline layer choices for pruned LoRA, to compare with the flow-chosen layers (flow_pruned16).

Each method scores the 32 LLM layers without the information-flow knockout, then rank 8 goes on the
--keep highest-scoring layers and none elsewhere (same format and size as make_ranks.py --keep):
  grad        mean gradient norm of q_proj + v_proj on training questions (IST, EMNLP 2024 Findings; LISA, NeurIPS 2024)
  spectrum    heavy-tail alpha of the q_proj / v_proj weight spectrum, higher = less well trained
              (AlphaLoRA, EMNLP 2024); uses the weights only
  similarity  angular distance between a layer's input and output at the last token, higher = the layer
              changes the representation more (Gromov et al., ICLR 2025)

Writes configs/ranks/<method>_pruned<keep>_<dataset>.json for each method and the scores to
output/layer_baselines/<dataset>/scores.json. Needs a GPU (one forward + backward per question).

Usage (from the repo root):
  python peft_lora/methods/layer_baselines/make_ranks.py --train-file datasets/scienceqa/train_llava.json
"""
import argparse
import json
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import budget
from common import load_model
from train_lora import GQATrain, collate

METHODS = ["grad", "spectrum", "similarity"]


def alpha_hill(weight):
    """Power-law exponent of the eigenvalues of W^T W, Hill estimator on the top half (as in AlphaLoRA)."""
    eigs = torch.linalg.svdvals(weight.float()).pow(2).sort().values  # ascending
    k = len(eigs) // 2
    top, xmin = eigs[-k:], eigs[-k - 1]
    return 1 + k / torch.log(top / xmin).sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--train-file", default="datasets/scienceqa/train_llava.json")
    parser.add_argument("--image-folder", default=None, help="default: images/ next to the training file")
    parser.add_argument("--num-questions", type=int, default=512, help="random training questions for grad and similarity")
    parser.add_argument("--keep", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.image_folder = args.image_folder or os.path.join(os.path.dirname(args.train_file), "images")
    dataset_name = "scienceqa" if "scienceqa" in args.train_file else "gqa"

    tokenizer, model, image_processor = load_model(args.model_path)
    model.config.use_cache = False
    model.requires_grad_(False)
    layers = model.get_model().layers
    targets = [[getattr(layer.self_attn, m).weight for m in budget.TARGET_MODULES] for layer in layers]
    for weights in targets:
        for w in weights:
            w.requires_grad_(True)

    scores = {"spectrum": [sum(alpha_hill(w) for w in weights) / len(weights) for weights in targets]}
    print("Spectrum done")

    # grad and similarity from the same forward + backward pass per question
    ends = {}  # layer -> (input, output) at the last token
    def hook(i):
        def record(module, args, kwargs, out):
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            out = out[0] if isinstance(out, tuple) else out
            ends[i] = (hidden[0, -1].detach().float(), out[0, -1].detach().float())
        return record

    for i, layer in enumerate(layers):
        layer.register_forward_hook(hook(i), with_kwargs=True)
    with open(args.train_file) as f:
        rows = json.load(f)
    rows = random.Random(args.seed).sample(rows, min(args.num_questions, len(rows)))
    dataset = GQATrain(rows, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode)

    grad_sum = [0.0] * budget.NUM_LAYERS
    dist_sum = [0.0] * budget.NUM_LAYERS
    for n in range(len(dataset)):
        batch = collate([dataset[n]], tokenizer.pad_token_id)
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        model(**batch).loss.backward()
        for i, weights in enumerate(targets):
            grad_sum[i] += math.sqrt(sum(w.grad.float().pow(2).sum().item() for w in weights))
            for w in weights:
                w.grad = None
            x, y = ends[i]
            cos = torch.nn.functional.cosine_similarity(x, y, dim=0).clamp(-1, 1).item()
            dist_sum[i] += math.acos(cos) / math.pi
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(dataset)} questions", flush=True)
    scores["grad"] = [g / len(dataset) for g in grad_sum]
    scores["similarity"] = [d / len(dataset) for d in dist_sum]

    out_dir = os.path.join("output/layer_baselines", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    chosen = {}
    for method in METHODS:
        chosen[method] = sorted(sorted(range(budget.NUM_LAYERS), key=lambda l: (-scores[method][l], l))[: args.keep])
        ranks = {f"layers.{l}.self_attn.{m}": budget.AVG_RANK if l in chosen[method] else 0
                 for l in range(budget.NUM_LAYERS) for m in budget.TARGET_MODULES}
        path = f"configs/ranks/{method}_pruned{args.keep}_{dataset_name}.json"
        with open(path, "w") as f:
            json.dump(ranks, f, indent=2)
        print(f"Wrote {path}: layers {chosen[method]}, {budget.num_params(ranks):,} params")
    with open(os.path.join(out_dir, "scores.json"), "w") as f:
        json.dump({"num_questions": len(dataset), "seed": args.seed, "keep": args.keep,
                   "scores": scores, "chosen": chosen}, f, indent=2)

    print("layer  " + "  ".join(f"{m:>10}" for m in METHODS))
    for l in range(budget.NUM_LAYERS):
        print(f"{l:5d}  " + "  ".join(f"{scores[m][l]:10.4g}" for m in METHODS))


if __name__ == "__main__":
    main()
