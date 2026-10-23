"""Build the right/wrong question files for the information-flow knockouts on failures.

The paper's CSVs (datasets/GQA_val_correct_*) only hold questions LLaVA answers correctly.
These two files come from our 20k training questions, split by the zero-shot predictions:
  datasets/GQA_train_wrong_All.csv    --n questions zero-shot LLaVA gets wrong
  datasets/GQA_train_correct_All.csv  --n questions it gets right, with the same mix of
                                      structural types (verify/query/...) as the wrong file
Every question type is used ("All"); the Image/Question/Last flows only need the question,
image and answer. All the images are already in datasets/gqa/images (training set).

Needs the zero-shot predictions on the training questions (see scripts/make_mistakes_split.py)
and the original GQA train JSON for the lowercase answers and question types.

Usage:
  python3 scripts/make_flow_failure_split.py --questions-dir ~/Downloads/gqa_questions [--n 1000] [--seed 0]
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

COLUMNS = ["question_id", "img_id", "question", "answer", "type_detailed", "type_semantic", "type_structural"]


def normalize(answer):
    return answer.strip().rstrip(".").lower()


def write_csv(path, qids, train):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for qid in qids:
            q = train[qid]
            writer.writerow([qid, q["imageId"], q["question"], q["answer"],
                             q["types"]["detailed"], q["types"]["semantic"], q["types"]["structural"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-dir", required=True, help="folder with train_balanced_questions.json")
    parser.add_argument("--predictions", default="output/peft/zero_shot/eval_train20000/predictions.jsonl")
    parser.add_argument("--n", type=int, default=1000, help="questions per file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="datasets")
    args = parser.parse_args()

    with open(args.predictions) as f:
        correct = {p["question_id"]: normalize(p["prediction"]) == normalize(p["answer"]) for p in map(json.loads, f)}
    with open(os.path.join(args.questions_dir, "train_balanced_questions.json")) as f:
        train = {qid: q for qid, q in json.load(f).items() if qid in correct}

    rng = random.Random(args.seed)
    wrong = rng.sample(sorted(q for q, ok in correct.items() if not ok), args.n)

    # right answers: same number per structural type as the wrong sample
    right_by_type = defaultdict(list)
    for qid in sorted(q for q, ok in correct.items() if ok):
        right_by_type[train[qid]["types"]["structural"]].append(qid)
    wrong_types = Counter(train[qid]["types"]["structural"] for qid in wrong)
    right = [qid for t, n in sorted(wrong_types.items()) for qid in rng.sample(right_by_type[t], n)]
    rng.shuffle(right)

    write_csv(os.path.join(args.out_dir, "GQA_train_wrong_All.csv"), wrong, train)
    write_csv(os.path.join(args.out_dir, "GQA_train_correct_All.csv"), right, train)
    print(f"Wrote {len(wrong)} wrong and {len(right)} right questions to {args.out_dir}/GQA_train_*_All.csv")
    print("Structural types (same in both): " + ", ".join(f"{t} {n}" for t, n in sorted(wrong_types.items())))


if __name__ == "__main__":
    main()
