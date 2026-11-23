"""Build the GQA train/testdev files for LoRA training and evaluation.

Reads the GQA question JSONs and writes, into datasets/gqa/:
  train_<N>_llava.json   N sampled train questions in LLaVA conversation format
  needed_train.txt       images used by the sampled train questions

  testdev_questions.jsonl  testdev-balanced questions, answers and GQA question types for evaluation
  needed_testdev.txt     images used by testdev-balanced

The sample is fixed by --seed, so everyone gets the same training questions.

Usage:
  python scripts/prepare_gqa_splits.py --questions-dir <folder with the JSONs> [--n-train 20000] [--seed 0]
"""
import argparse
import json
import os
import random

PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."


def capitalize(answer):
    """LLaVA-1.5 answers with a capital first letter ("Yes", "Right"); GQA answers are lowercase.
    Matching LLaVA's style keeps fine-tuning from spending updates on letter case
    (the evaluation lowercases answers anyway)."""
    return answer[:1].upper() + answer[1:]


def image_list(image_ids):
    return "\n".join(f"images/{i}.jpg" for i in sorted(image_ids)) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--out-dir", default="datasets/gqa")
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(os.path.join(args.questions_dir, "train_balanced_questions.json")) as f:
        train = json.load(f)
    qids = sorted(train)
    random.Random(args.seed).shuffle(qids)
    sample = qids[: args.n_train]

    llava = [
        {
            "id": qid,
            "image": f"{train[qid]['imageId']}.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\n" + train[qid]["question"] + PROMPT_SUFFIX},
                {"from": "gpt", "value": capitalize(train[qid]["answer"])},
            ],
        }
        for qid in sample
    ]
    train_images = {train[qid]["imageId"] for qid in sample}
    del train

    with open(os.path.join(args.questions_dir, "testdev_balanced_questions.json")) as f:
        testdev = json.load(f)
    testdev_images = {q["imageId"] for q in testdev.values()}
    testdev_rows = [
        {"question_id": qid, "image": f"{q['imageId']}.jpg", "text": q["question"] + PROMPT_SUFFIX, "answer": q["answer"],
         "structural": q["types"]["structural"], "semantic": q["types"]["semantic"]}
        for qid, q in sorted(testdev.items())
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"train_{args.n_train}_llava.json"), "w") as f:
        json.dump(llava, f)
    with open(os.path.join(args.out_dir, "needed_train.txt"), "w") as f:
        f.write(image_list(train_images))
    with open(os.path.join(args.out_dir, "testdev_questions.jsonl"), "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in testdev_rows)
    with open(os.path.join(args.out_dir, "needed_testdev.txt"), "w") as f:
        f.write(image_list(testdev_images))

    print(f"train: {len(llava)} questions, {len(train_images)} images")
    print(f"testdev: {len(testdev)} questions, {len(testdev_images)} images")


if __name__ == "__main__":
    main()
