"""GQA testdev-balanced accuracy of LLaVA-1.5-7B, with or without a LoRA adapter.

Writes predictions.jsonl and metrics.json to <adapter's run folder>/eval/, or to
output/peft/zero_shot/eval/ without an adapter. Predictions are saved as they are made,
so a stopped job continues where it left off when submitted again.

Usage:
  python peft_lora/eval_gqa.py [--adapter output/peft/<method>/seed<k>/adapter] [--limit 100]
"""
import argparse
import json
import os

import torch
from tqdm import tqdm

from common import build_prompt, load_image, load_model, tokenize
from llava.constants import DEFAULT_IMAGE_TOKEN


def normalize(answer):
    return answer.strip().rstrip(".").lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--questions", default="datasets/gqa/testdev_questions.jsonl")
    parser.add_argument("--image-folder", default="datasets/gqa/images")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="evaluate the first N questions only (for tests)")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    run_dir = os.path.dirname(args.adapter.rstrip("/")) if args.adapter else "output/peft/zero_shot"
    out_dir = args.out_dir or os.path.join(run_dir, "eval" + (f"_limit{args.limit}" if args.limit else ""))
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "predictions.jsonl")

    with open(args.questions) as f:
        questions = [json.loads(line) for line in f][: args.limit]
    done = set()
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            done = {json.loads(line)["question_id"] for line in f}
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"{len(questions)} questions, {len(done)} already done")

    if todo:
        tokenizer, model, image_processor = load_model(args.model_path)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        model.eval()

        with open(pred_path, "a") as f, torch.inference_mode():
            for q in tqdm(todo):
                image, size = load_image(os.path.join(args.image_folder, q["image"]), image_processor, model.config)
                prompt = build_prompt(DEFAULT_IMAGE_TOKEN + "\n" + q["text"], None, args.conv_mode)
                input_ids = tokenize(prompt, tokenizer).unsqueeze(0).cuda()
                output_ids = model.generate(input_ids, images=image.unsqueeze(0).cuda(), image_sizes=[size],
                                            do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True)
                prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                f.write(json.dumps({"question_id": q["question_id"], "prediction": prediction, "answer": q["answer"]}) + "\n")
                f.flush()

    with open(pred_path) as f:
        rows = [json.loads(line) for line in f]
    correct = sum(normalize(r["prediction"]) == normalize(r["answer"]) for r in rows)
    metrics = {"accuracy": round(100 * correct / len(rows), 2), "correct": correct, "total": len(rows),
               "adapter": args.adapter or "zero-shot"}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
