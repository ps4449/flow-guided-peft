"""LoRA fine-tuning of LLaVA-1.5-7B on the GQA train sample, shared by all methods.

The method is set only by its rank file in configs/ranks/:
  LoRA     {"layers.<i>.self_attn.<q|k|v|o>_proj": <rank>, "layers.<i>.mlp.<gate|up|down>_proj": <rank>, ...}
           Only the listed LLM modules get LoRA (rank 0 = none). lora_alpha = 2 * rank everywhere.
  AdaLoRA  {"adalora": {"init_r": 12, "target_r": 8, "target_modules": ["q_proj", "v_proj"], ...}}
           Other keys in "adalora" are passed on to peft.AdaLoraConfig.

The vision tower and projector stay frozen. The adapter is saved to
output/peft/<rank file name>/<settings>/seed<k>/adapter/, where <settings> lists the options
changed from their defaults (e.g. "lr1e-4_ep2"), or "default". Add --tag to name a change that
is not an option (e.g. new training data). Before the 6 h limit the job saves a
checkpoint and stops; submitting the same command again resumes from it.

--flow-gates adds a learnable knob per head on the Image->Question attention in layers 5-15
(FlowGates in common.py, 352 parameters, own learning rate --gate-lr), saved as adapter/flow_gates.pt.

--variant changes the LoRA itself, same ranks and layers (LoRA only):
  dora      weight = magnitude * direction, LoRA on the direction (DoRA, ICML 2024); adds one
            magnitude vector per module (4,096 parameters each)
  pissa     A and B start from the top singular vectors of each weight, the rest stays frozen (PiSSA,
            NeurIPS 2024). Saved converted to a plain LoRA of twice the rank, so it loads on the original model.
  loraplus  B learns --loraplus-ratio times faster than A (LoRA+, ICML 2024)

Usage:
  python peft_lora/train_lora.py --ranks configs/ranks/<method>.json --seed 0 [--limit 100]
  ... --train-file datasets/scienceqa/train_llava.json    # ScienceQA -> <settings> = scienceqa_train_llava
  ... --flow-gates                                        # -> <settings> gets "flowgates"
  ... --variant dora                                      # -> <settings> gets "dora"
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
from peft import AdaLoraConfig, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from common import FlowGates, build_prompt, load_image, load_model, tokenize
from llava.constants import IGNORE_INDEX

MODULE_RE = re.compile(r"layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)")
RESUME_EXIT_CODE = 3

if "rank_pattern" not in LoraConfig.__dataclass_fields__:
    sys.exit("This peft version has no per-layer ranks. Run: pip install peft==0.12.0")


class GQATrain(Dataset):
    def __init__(self, rows, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.rows = rows
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        row = self.rows[index]
        human, answer = row["conversations"][0]["value"], row["conversations"][1]["value"]
        prefix = tokenize(build_prompt(human, None, self.conv_mode), self.tokenizer)
        input_ids = tokenize(build_prompt(human, answer, self.conv_mode), self.tokenizer)
        labels = input_ids.clone()
        labels[: len(prefix)] = IGNORE_INDEX  # loss only on the answer tokens and </s>
        image, size = load_image(os.path.join(self.image_folder, row["image"]), self.image_processor, self.model_config)
        return input_ids, labels, image, size

    def __len__(self):
        return len(self.rows)


def collate(batch, pad_id):
    input_ids, labels, images, sizes = zip(*batch)
    attention_mask = pad_sequence([torch.ones(len(x), dtype=torch.bool) for x in input_ids], batch_first=True)
    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_id),
        "labels": pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
        "attention_mask": attention_mask,
        "images": torch.stack(images),
        "image_sizes": list(sizes),
        "modalities": ["image"] * len(batch),  # LLaVA-NeXT requires one entry per sample
    }


def make_peft_config(spec, total_steps, variant="lora"):
    """Returns the peft config and, for LoRA, the expected {module: rank}."""
    if "adalora" in spec:
        if variant != "lora":
            sys.exit("--variant works with LoRA rank files only")
        cfg = dict(spec["adalora"])
        names = "|".join(cfg.pop("target_modules"))
        target = rf"model\.layers\.\d+\.(self_attn|mlp)\.({names})"
        return AdaLoraConfig(target_modules=target, total_step=total_steps, **cfg), None

    ranks = {k: int(r) for k, r in spec.items() if int(r) > 0}
    unknown = [k for k in ranks if not MODULE_RE.fullmatch(k)]
    if unknown:
        sys.exit(f"Unknown module names in the rank file, e.g. {unknown[:3]}")
    values = list(ranks.values())
    default_r = max(set(values), key=values.count)
    config = LoraConfig(
        r=default_r,
        lora_alpha=2 * default_r,
        lora_dropout=0.05,
        # full names, so the CLIP vision tower (which also has layers.<i>.self_attn.q_proj) is never matched
        target_modules="|".join(re.escape("model." + k) for k in ranks),
        rank_pattern={re.escape(k): r for k, r in ranks.items()},
        alpha_pattern={re.escape(k): 2 * r for k, r in ranks.items()},
        use_dora=variant == "dora",
        init_lora_weights="pissa" if variant == "pissa" else True,
    )
    return config, ranks


def check_ranks(model, expected):
    got = {}
    for name, module in model.named_modules():
        if hasattr(module, "r") and isinstance(module.r, dict) and "default" in module.r:
            got[re.search(r"layers\.\d+\..*", name).group(0)] = module.r["default"]
    if got != expected:
        wrong = sorted(set(got.items()) ^ set(expected.items()))[:5]
        sys.exit(f"LoRA ranks do not match the rank file, e.g. {wrong}")


# options that change the training result, with their short names in the run folder
SETTING_NAMES = {"lr": "lr", "epochs": "ep", "batch_size": "bs", "grad_accum": "ga", "warmup_ratio": "wu",
                 "train_file": "", "limit": "limit", "flow_gates": "flowgates", "gate_lr": "glr",
                 "variant": "", "loraplus_ratio": "lpr"}
                 # a different train file shows as its name


def short_value(value):
    if isinstance(value, bool):  # a switch shows as its name only
        return ""
    if isinstance(value, float) and value < 0.01:
        return f"{value:.0e}".replace("e-0", "e-")  # 0.0001 -> 1e-4
    if isinstance(value, str):  # a training file: its name, plus the dataset folder when it is not GQA
        folder, stem = os.path.basename(os.path.dirname(value)), os.path.splitext(os.path.basename(value))[0]
        return stem if folder in ("gqa", "") else f"{folder}_{stem}"  # e.g. scienceqa_train_llava
    return f"{value:g}" if isinstance(value, float) else str(value)


def settings_name(args, parser):
    """Folder name from the options that differ from their defaults, e.g. "lr1e-4_ep2"."""
    parts = [args.tag] if args.tag else []
    parts += [f"{short}{short_value(getattr(args, key))}" for key, short in SETTING_NAMES.items()
              if getattr(args, key) != parser.get_default(key)]
    return "_".join(parts) or "default"


def save_checkpoint(path, model, optimizer, scheduler, step, gates):
    torch.save({
        "adapter": get_peft_model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "gates": gates.state_dict() if gates else None,
    }, path + ".tmp")
    os.replace(path + ".tmp", path)  # never leave a half-written checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--train-file", default="datasets/gqa/train_20000_llava.json")
    parser.add_argument("--image-folder", default=None, help="default: images/ next to the training file")
    parser.add_argument("--out-dir", default=None, help="overrides the automatic run folder")
    parser.add_argument("--tag", default=None, help="extra label for the run folder, e.g. a data version")
    parser.add_argument("--limit", type=int, default=None, help="train on the first N questions only (for tests)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-every", type=int, default=200, help="optimizer steps between checkpoints")
    parser.add_argument("--max-hours", type=float, default=5.5, help="stop with a checkpoint after this long")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--flow-gates", action="store_true", help="learnable Image->Question gates in layers 5-15")
    parser.add_argument("--gate-lr", type=float, default=1e-2, help="learning rate of the flow gates")
    parser.add_argument("--variant", default="lora", choices=["lora", "dora", "pissa", "loraplus"])
    parser.add_argument("--loraplus-ratio", type=float, default=16.0, help="B / A learning rate for --variant loraplus")
    args = parser.parse_args()
    args.image_folder = args.image_folder or os.path.join(os.path.dirname(args.train_file), "images")
    start = time.time()

    name = os.path.splitext(os.path.basename(args.ranks))[0]
    out_dir = args.out_dir or os.path.join("output/peft", name, settings_name(args, parser), f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    adapter_dir = os.path.join(out_dir, "adapter")
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    if os.path.exists(adapter_dir):
        print(f"Already trained: {adapter_dir}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.ranks) as f:
        spec = json.load(f)
    with open(args.train_file) as f:
        rows = json.load(f)[: args.limit]

    tokenizer, model, image_processor = load_model(args.model_path)
    model.config.use_cache = False
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    gates = FlowGates(model) if args.flow_gates else None

    bs, accum = args.batch_size, args.grad_accum
    steps_per_epoch = math.ceil(len(rows) / bs) // accum
    total_steps = steps_per_epoch * args.epochs

    peft_config, expected_ranks = make_peft_config(spec, total_steps, args.variant)
    model = get_peft_model(model, peft_config)
    if expected_ranks is not None:
        check_ranks(model, expected_ranks)
    pissa_init_dir = os.path.join(out_dir, "pissa_init")
    if args.variant == "pissa" and not os.path.exists(pissa_init_dir):
        # the starting A and B, needed to save the result as a plain LoRA (the SVD gives the same start on resume)
        model.peft_config["default"].init_lora_weights = True
        model.save_pretrained(pissa_init_dir)
        model.peft_config["default"].init_lora_weights = "pissa"
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    if args.variant == "loraplus":
        groups = [{"params": [p for n, p in named if "lora_B" not in n]},
                  {"params": [p for n, p in named if "lora_B" in n], "lr": args.lr * args.loraplus_ratio}]
    else:
        groups = [{"params": params}]
    if gates:  # the gates start at 0 and need a larger step size than LoRA to move in one epoch
        groups.append({"params": list(gates.parameters()), "lr": args.gate_lr})
        params = params + list(gates.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"Method: {name}  Seed: {args.seed}  Trainable params: {n_params:,}  Steps: {total_steps}"
          + ("  Flow gates: on" if gates else "") + (f"  Variant: {args.variant}" if args.variant != "lora" else ""))

    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    step = 0
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if gates:
            gates.load_state_dict(state["gates"])
        step = state["step"]
        print(f"Resumed from step {step}")

    dataset = GQATrain(rows, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode)
    sample = dataset[0]
    print("Target of sample 0:", repr(tokenizer.decode(sample[0][sample[1] != IGNORE_INDEX])))

    model.train()
    is_adalora = "adalora" in spec
    log = open(os.path.join(out_dir, "train_log.jsonl"), "a")
    while step < total_steps:
        # the data order depends only on seed and epoch, so a resumed run skips exactly the batches already seen
        epoch, step_in_epoch = divmod(step, steps_per_epoch)
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(args.seed * 1000 + epoch)).tolist()
        order = order[step_in_epoch * accum * bs: steps_per_epoch * accum * bs]
        loader = DataLoader(dataset, batch_size=bs, sampler=order, num_workers=args.num_workers,
                            collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
        running, logged_steps = 0.0, 0  # loss summed over the optimizer steps since the last log line
        for i, batch in enumerate(loader):
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            if gates:
                gates.set_spans(batch["input_ids"], batch["labels"], batch["attention_mask"])
            loss = model(**batch).loss / accum
            loss.backward()
            running += loss.item()
            if (i + 1) % accum:
                continue

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            if is_adalora:
                model.base_model.update_and_allocate(step)  # AdaLoRA prunes ranks from the gradients
            optimizer.zero_grad(set_to_none=True)
            step += 1
            logged_steps += 1

            if step % 10 == 0 or step == total_steps:
                entry = {"step": step, "epoch": epoch, "loss": round(running / logged_steps, 4),
                         "lr": scheduler.get_last_lr()[0], "minutes": round((time.time() - start) / 60, 1)}
                if gates:  # mean gate per layer (layers 5-15)
                    entry["gates"] = [round(g, 3) for g in gates.log_scale.mean(1).tolist()]
                print(json.dumps(entry), flush=True)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                running, logged_steps = 0.0, 0

            if step == total_steps:
                break
            if step % args.save_every == 0:
                save_checkpoint(ckpt_path, model, optimizer, scheduler, step, gates)
            if time.time() - start > args.max_hours * 3600:
                save_checkpoint(ckpt_path, model, optimizer, scheduler, step, gates)
                print(f"Stopped at step {step}/{total_steps} before the time limit. Submit the same command again to resume.")
                sys.exit(RESUME_EXIT_CODE)

    if args.variant == "pissa":
        model.save_pretrained(adapter_dir, path_initial_model_for_weight_conversion=pissa_init_dir)
    else:
        model.save_pretrained(adapter_dir)
    if gates:
        torch.save(gates.state_dict(), os.path.join(adapter_dir, "flow_gates.pt"))
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump({**vars(args), "trainable_params": n_params, "total_steps": total_steps}, f, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
