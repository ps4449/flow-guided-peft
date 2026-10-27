"""ScienceQA AdaLoRA: published regularization/pruning conditions.

Preserves the validated training operations of the archived closure engine.
Use explicit config, seed and a new output directory; see experiments/README.md.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/adalora_reproduction"))
sys.path.insert(1, str(ROOT / "experiments/scienceqa_adalora"))

import numpy as np
import torch
from peft import AdaLoraConfig, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from common import build_prompt, load_image, load_model, tokenize
from budget import validate_budget as validate_shared_budget
import copy

def validate_budget(spec, model):
    ref = copy.deepcopy(spec)
    ref["adalora"]["target_r"] = 8
    result = validate_shared_budget(ref, model)
    expected = 64 * spec["adalora"]["target_r"]
    result["final_rank_sum"] = expected
    result["target_effective_parameters_nominal"] = expected * (result["per_rank_AB_cost"] + 1)
    result["note"] = "Mechanism diagnostic; fixed-rank-12 arms use a larger final budget than the main benchmark"
    return result
from adalora_runtime import (configure, snapshot_loss, ranks, allocator_state,
                             restore_allocator, raw_adapter_state, export_full_shape)
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

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
        if not torch.equal(prefix, input_ids[:len(prefix)]):
            raise ValueError(f'Prompt/answer token boundary mismatch for {row["id"]}')
        labels = input_ids.clone()
        labels[: len(prefix)] = IGNORE_INDEX  # loss only on the answer tokens and </s>
        if not bool((labels != IGNORE_INDEX).any()):
            raise ValueError(f'No answer labels for {row["id"]}')
        image_count = int((input_ids == IMAGE_TOKEN_INDEX).sum())
        if image_count != 1 or len(input_ids) + 575 > self.tokenizer.model_max_length:
            raise ValueError(f'Image-token expansion would truncate sample {row["id"]}')
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
    }


def make_peft_config(spec, total_steps):
    """Returns the peft config and, for LoRA, the expected {module: rank}."""
    if "adalora" in spec:
        return configure(spec['adalora'], total_steps, AdaLoraConfig), None

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


def save_checkpoint(path, model, optimizer, scheduler, step, contract):
    is_adalora = model.peft_config['default'].peft_type == 'ADALORA'
    torch.save({
        "format_version": 2,
        "contract": contract,
        "adapter": raw_adapter_state(model),
        "allocator": allocator_state(model) if is_adalora else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "rng": (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()),
    }, path + ".tmp")
    os.replace(path + ".tmp", path)  # never leave a half-written checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--train-file", default="datasets/gqa/train_20000_llava.json")
    parser.add_argument("--image-folder", default="datasets/gqa/images")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="train on the first N questions only (for tests)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-every", type=int, default=200, help="optimizer steps between checkpoints")
    parser.add_argument("--max-hours", type=float, default=5.5, help="stop with a checkpoint after this long")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--no-pruning', action='store_true', help='Mechanism diagnostic: keep all initial rank-12 components')
    parser.add_argument('--stop-after', type=int, default=None, help='diagnostic: checkpoint and exit 3 at this optimizer step')
    args = parser.parse_args()
    start = time.time()

    name = os.path.splitext(os.path.basename(args.ranks))[0]
    out_dir = args.out_dir or os.path.join("output/peft", name, f"seed{args.seed}" + (f"_limit{args.limit}" if args.limit else ""))
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
    if not rows or args.batch_size < 1 or args.grad_accum < 1 or args.epochs < 1:
        raise ValueError('Need nonempty data and positive batch/accumulation/epochs')
    is_adalora = 'adalora' in spec
    if not is_adalora or args.batch_size != 1 or args.grad_accum != 16:
        raise ValueError('Closure arms require AdaLoRA parameterization and microbatch 1 / accumulation 16')
    if spec['adalora']['init_r'] != 12 or spec['adalora']['target_r'] != (12 if args.no_pruning else 8):
        raise ValueError('Pruning switch/config mismatch')
    reg_weight = spec['adalora']['orth_reg_weight']
    if reg_weight not in (0.0, 0.5):
        raise ValueError('Only predeclared regularization weights are allowed')
    # ScienceQA has 6218 rows. Use the existing shuffled-order slice below,
    # which keeps 388 complete updates (6208 rows), matching the shared loop.

    tokenizer, model, image_processor = load_model(args.model_path)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Shared precision is BF16; this GPU does not support it')
    budget_report = validate_budget(spec, model)
    print('Budget:', json.dumps(budget_report), flush=True)
    with open(os.path.join(out_dir, 'budget.json'), 'w') as f:
        json.dump(budget_report, f, indent=2)
    tokenizer.model_max_length = args.max_length
    model.config.tokenizer_model_max_length = args.max_length
    model.config.use_cache = False
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    bs, accum = args.batch_size, args.grad_accum
    steps_per_epoch = math.ceil(len(rows) / bs) // accum
    total_steps = steps_per_epoch * args.epochs
    if total_steps < 1:
        raise ValueError('No complete optimizer updates in selected data')

    peft_config, expected_ranks = make_peft_config(spec, total_steps)
    model = get_peft_model(model, peft_config)
    if expected_ranks is not None:
        check_ranks(model, expected_ranks)
    if args.no_pruning:
        model.peft_config['default'].rank_pattern = {
            n: [True] * p.shape[0]
            for n, p in model.base_model.model.named_parameters() if 'lora_E.default' in n
        }
        assert len(model.peft_config['default'].rank_pattern) == 64
    params = [p for p in model.parameters() if p.requires_grad]
    if any(not any(tag in n for tag in ('lora_A.', 'lora_B.', 'lora_E.'))
           for n, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError('Unexpected trainable base or vision parameters')
    for p in params:
        p.data = p.data.float()
    n_params = sum(p.numel() for p in params)
    print(f"Method: {name}  Seed: {args.seed}  Trainable params: {n_params:,}  Steps: {total_steps}")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    contract = dict(spec=spec, model=args.model_path, batch_size=bs, grad_accum=accum,
                    total_steps=total_steps, lr=args.lr, warmup_ratio=args.warmup_ratio,
                    max_length=args.max_length, seed=args.seed, conv_mode=args.conv_mode, no_pruning=args.no_pruning,
                    data_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())

    step = 0
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        if state.get('format_version') != 2 or state.get('contract') != contract:
            raise RuntimeError('Checkpoint format/settings differ; use a fresh output directory')
        result = model.load_state_dict(state['adapter'], strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f'Unexpected adapter keys: {result.unexpected_keys}')
        expected_keys = set(raw_adapter_state(model))
        if set(state['adapter']) != expected_keys:
            raise RuntimeError('Adapter checkpoint is incomplete')
        if is_adalora:
            restore_allocator(model, state['allocator'])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        py_rng, np_rng, cpu_rng, cuda_rng = state['rng']
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        print(f"Resumed from step {step}")

    dataset = GQATrain(rows, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode)
    sample = dataset[0]
    print("Target of sample 0:", repr(tokenizer.decode(sample[0][sample[1] != IGNORE_INDEX])))

    # Runtime counts are attached only after dataset/prompt sanity checks.
    import functools, inspect
    import batch_guard
    row_stats = batch_guard.install(model)
    base_for_guard = model.get_base_model()
    guarded = base_for_guard.prepare_inputs_labels_for_multimodal
    signature_for_guard = inspect.signature(guarded)
    target_rows = total_steps * accum * bs
    @functools.wraps(guarded)
    def count_training(*a, **kw):
        if signature_for_guard.bind_partial(*a, **kw).arguments.get('labels') is None:
            return guarded.__wrapped__(*a, **kw)
        result = guarded(*a, **kw)
        if row_stats['input_rows'] == target_rows:
            assert row_stats['language_rows'] == target_rows
            assert row_stats['supervised_tokens'] == 2 * target_rows
            from pathlib import Path
            Path(out_dir, 'batch_integrity.json').write_text(json.dumps(row_stats, indent=2) + '\n')
        return result
    base_for_guard.prepare_inputs_labels_for_multimodal = count_training
    model.train()
    model.get_base_model().get_vision_tower().eval()
    captured, restore_forward = snapshot_loss(model) if is_adalora else ({}, lambda: None)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    log = open(os.path.join(out_dir, "train_log.jsonl"), "a")
    while step < total_steps:
        # the data order depends only on seed and epoch, so a resumed run skips exactly the batches already seen
        epoch, step_in_epoch = divmod(step, steps_per_epoch)
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(args.seed * 1000 + epoch)).tolist()
        order = order[step_in_epoch * accum * bs: steps_per_epoch * accum * bs]
        loader = DataLoader(dataset, batch_size=bs, sampler=order, num_workers=args.num_workers,
                            collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
                            generator=torch.Generator().manual_seed(args.seed * 1000 + epoch + 12345))
        running = 0.0
        running_task = 0.0
        tick = time.perf_counter()
        for i, batch in enumerate(loader):
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            # Generic PeftModel.forward bypasses AdaLoraModel.forward in PEFT
            # 0.12. Call the tuner explicitly to include orthogonal regularization.
            # PEFT 0.12 rejects orth_reg_weight=0 in its tuner forward.
            # The underlying model computes the identical task loss without that regularizer.
            outputs = model.base_model(**batch) if reg_weight > 0 else model.get_base_model()(**batch)
            loss = outputs.loss / accum
            if reg_weight > 0 and not float(outputs.loss.detach()) > captured['task']:
                raise RuntimeError('AdaLoRA regularization did not contribute to training loss')
            if reg_weight == 0 and abs(float(outputs.loss.detach()) - captured['task']) > 1e-7:
                raise RuntimeError('Off-arm task loss differs unexpectedly')
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            running += loss.item()
            running_task += captured['task'] / accum if is_adalora else loss.item()
            if (i + 1) % accum:
                continue

            if is_adalora and any(p.grad is None for p in params):
                raise RuntimeError('Missing AdaLoRA gradients')
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not bool(torch.isfinite(grad_norm)):
                raise RuntimeError('Non-finite gradient norm')
            optimizer.step()
            scheduler.step()
            step += 1
            if is_adalora and not args.no_pruning:
                model.base_model.update_and_allocate(step)  # AdaLoRA prunes ranks from the gradients
            optimizer.zero_grad(set_to_none=True)

            if is_adalora or step % 10 == 0 or step == total_steps:
                torch.cuda.synchronize()
                entry = {"step": step, "epoch": epoch, "loss": round(running, 4), "lr": scheduler.get_last_lr()[0],
                         "task_loss": running_task, "regularization": running-running_task,
                         "grad_norm": float(grad_norm),
                         "seconds": time.perf_counter() - tick,
                         "minutes": round((time.time() - start) / 60, 1)}
                if is_adalora:
                    entry.update(ranks(model))
                print(json.dumps(entry), flush=True)
                log.write(json.dumps(entry) + "\n")
                log.flush()
            running = 0.0
            running_task = 0.0
            tick = time.perf_counter()

            if step == total_steps:
                break
            if step % args.save_every == 0:
                save_checkpoint(ckpt_path, model, optimizer, scheduler, step, contract)
            if time.time() - start > args.max_hours * 3600 or step == args.stop_after:
                save_checkpoint(ckpt_path, model, optimizer, scheduler, step, contract)
                print(f"Stopped at step {step}/{total_steps} before the time limit. Submit the same command again to resume.")
                sys.exit(RESUME_EXIT_CODE)

    restore_forward()
    model.eval()
    # Save one fixed training input/logit reference for fresh-process reload checks.
    reference = collate([dataset[0]], tokenizer.pad_token_id)
    reference.pop('labels')
    torch.save(reference, os.path.join(out_dir, 'reload_input.pt'))
    reference = {k: v.cuda() if torch.is_tensor(v) else v for k, v in reference.items()}
    with torch.no_grad():
        expected = model.get_base_model()(**reference, use_cache=False).logits[:, -8:, :].float().cpu()
    torch.save(expected, os.path.join(out_dir, 'reload_logits.pt'))
    if is_adalora:
        final = ranks(model)
        expected_final_rank = 64 * spec['adalora']['target_r']
        if not model.peft_config['default'].rank_pattern or final['retained_rank_sum'] != expected_final_rank:
            raise RuntimeError(f'Diagnostic final effective rank budget differs from expected: {final}')
        export_full_shape(model, adapter_dir)
    else:
        model.save_pretrained(adapter_dir)
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump({**vars(args), "trainable_params": n_params, "total_steps": total_steps,
                   "budget": budget_report, "precision": "base_bf16_adapter_fp32",
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "final_rank": ranks(model) if is_adalora else None}, f, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
