"""Build the ScienceQA-IMG files for LoRA training and evaluation (a second dataset next to GQA).

ScienceQA is multiple-choice school science (diagrams, maps, photos). LLaVA-1.5 was not trained on it.
Only the questions that come with an image are kept. Reads the Hugging Face copy
(derek-thomas/ScienceQA, images stored inside the parquet files) and writes, into datasets/scienceqa/:
  images/<split>_<index>.png   the images
  train_llava.json             training questions in LLaVA conversation format (answer = option letter)
  train_questions.jsonl        the same training questions for eval_gqa.py (zero-shot on train)
  val_questions.jsonl          validation questions for eval_gqa.py
  test_questions.jsonl         test questions for eval_gqa.py

The prompt follows LLaVA-1.5's ScienceQA evaluation:
  Context: <hint>        (only when the question has one)
  <question>
  A. <choice>
  B. <choice>
  Answer with the option's letter from the given choices directly.

Usage (on your Mac, needs `pip install pyarrow`):
  huggingface-cli download derek-thomas/ScienceQA --repo-type dataset --local-dir ~/Downloads/scienceqa
  python3 scripts/prepare_scienceqa.py --data-dir ~/Downloads/scienceqa
"""
import argparse
import glob
import json
import os

import pyarrow.parquet as pq

PROMPT_SUFFIX = "\nAnswer with the option's letter from the given choices directly."
LETTERS = "ABCDEFGHIJ"
SPLITS = {"train": "train", "validation": "val", "test": "test"}
TYPE_FIELDS = ["subject", "category", "grade"]


def prompt_text(row):
    lines = [f"Context: {row['hint']}"] if row["hint"].strip() else []
    lines.append(row["question"])
    lines += [f"{LETTERS[i]}. {choice}" for i, choice in enumerate(row["choices"])]
    return "\n".join(lines) + PROMPT_SUFFIX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="folder with data/<split>-*.parquet")
    parser.add_argument("--out-dir", default="datasets/scienceqa")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)

    for hf_split, split in SPLITS.items():
        (path,) = glob.glob(os.path.join(os.path.expanduser(args.data_dir), "data", f"{hf_split}-*.parquet"))
        questions = []
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            if not (row["image"] and row["image"]["bytes"]):
                continue  # text-only question
            image = f"{split}_{index}.png"
            with open(os.path.join(args.out_dir, "images", image), "wb") as f:
                f.write(row["image"]["bytes"])
            questions.append({"question_id": f"{split}_{index}", "image": image, "text": prompt_text(row),
                              "answer": LETTERS[row["answer"]], "question": row["question"],
                              **{field: row[field] for field in TYPE_FIELDS}})

        with open(os.path.join(args.out_dir, f"{split}_questions.jsonl"), "w") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")
        if split == "train":
            llava = [{"id": q["question_id"], "image": q["image"],
                      "conversations": [{"from": "human", "value": "<image>\n" + q["text"]},
                                        {"from": "gpt", "value": q["answer"]}]}
                     for q in questions]
            with open(os.path.join(args.out_dir, "train_llava.json"), "w") as f:
                json.dump(llava, f)
        print(f"{split}: {len(questions)} questions with an image")


if __name__ == "__main__":
    main()
