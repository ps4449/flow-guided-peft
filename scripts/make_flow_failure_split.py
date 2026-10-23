"""Build the right/wrong question files for the information-flow knockouts on failures.

The paper's CSVs (datasets/GQA_val_correct_*) only hold questions LLaVA answers correctly.
These files come from training questions, split by the zero-shot predictions:
  GQA (default, from our 20k training questions):
    datasets/GQA_train_wrong_All.csv    --n questions zero-shot LLaVA gets wrong
    datasets/GQA_train_correct_All.csv  --n questions it gets right, with the same mix of
                                        structural types (verify/query/...) as the wrong file
  ScienceQA (--dataset scienceqa, from the 6,218 training questions):
    datasets/SQA_train_wrong_ScienceQA.csv / datasets/SQA_train_correct_ScienceQA.csv
                                        same, matched by subject (natural/social/language science);
                                        a "prompt" column holds the full ScienceQA prompt
Every question type is used; the Image/Question/Last flows only need the question, image and answer.
Answers are lowercase, as InformationFlow.py compares them with its lowercased prediction.

Needs the zero-shot predictions on the training questions:
  GQA:       output/peft/zero_shot/eval_train20000/ (see scripts/make_mistakes_split.py)
  ScienceQA: output/peft/zero_shot/eval_scienceqa_train/
             (sbatch --export=ALL,EXTRA="--questions datasets/scienceqa/train_questions.jsonl" scripts/tc2_eval.sh)
GQA also needs the original GQA train JSON for the lowercase answers and question types.

Usage:
  python3 scripts/make_flow_failure_split.py --questions-dir ~/Downloads/gqa_questions [--n 1000] [--seed 0]
  python3 scripts/make_flow_failure_split.py --dataset scienceqa [--n 1000] [--seed 0]
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

GQA_COLUMNS = ["question_id", "img_id", "question", "answer", "type_detailed", "type_semantic", "type_structural"]
SQA_COLUMNS = ["question_id", "img_id", "question", "answer", "prompt", "subject", "category", "grade"]


def normalize(answer):
    return answer.strip().rstrip(".").lower()


def load_gqa(args):
    """{question_id: CSV row} and the field the right sample is matched on."""
    with open(os.path.join(args.questions_dir, "train_balanced_questions.json")) as f:
        train = json.load(f)
    with open("datasets/gqa/train_20000_questions.jsonl") as f:
        ids = [json.loads(line)["question_id"] for line in f]
    rows = {qid: {"question_id": qid, "img_id": train[qid]["imageId"], "question": train[qid]["question"],
                  "answer": train[qid]["answer"], "type_detailed": train[qid]["types"]["detailed"],
                  "type_semantic": train[qid]["types"]["semantic"], "type_structural": train[qid]["types"]["structural"]}
            for qid in ids}
    return rows, "type_structural", GQA_COLUMNS, "GQA_train_{}_All.csv"


def load_scienceqa(args):
    with open("datasets/scienceqa/train_questions.jsonl") as f:
        questions = [json.loads(line) for line in f]
    # InformationFlow.py finds the question by its text in the prompt tokens; line breaks and non-ASCII
    # characters become byte tokens there, so those questions (~4%) cannot be located and are left out
    questions = [q for q in questions if "\n" not in q["question"] and q["question"].isascii()]
    rows = {q["question_id"]: {"question_id": q["question_id"], "img_id": q["image"], "question": q["question"],
                               "answer": q["answer"].lower(), "prompt": q["text"], "subject": q["subject"],
                               "category": q["category"], "grade": q["grade"]}
            for q in questions}
    return rows, "subject", SQA_COLUMNS, "SQA_train_{}_ScienceQA.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gqa", "scienceqa"], default="gqa")
    parser.add_argument("--questions-dir", help="GQA only: folder with train_balanced_questions.json")
    parser.add_argument("--predictions", default=None, help="default: the zero-shot predictions on the training questions")
    parser.add_argument("--n", type=int, default=1000, help="questions per file (at most the number of wrong answers)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="datasets")
    args = parser.parse_args()
    if args.dataset == "gqa" and not args.questions_dir:
        parser.error("--questions-dir is needed for GQA")

    rows, match_field, columns, file_name = load_gqa(args) if args.dataset == "gqa" else load_scienceqa(args)
    predictions = args.predictions or {"gqa": "output/peft/zero_shot/eval_train20000/predictions.jsonl",
                                       "scienceqa": "output/peft/zero_shot/eval_scienceqa_train/predictions.jsonl"}[args.dataset]
    with open(predictions) as f:
        correct = {p["question_id"]: normalize(p["prediction"]) == normalize(p["answer"])
                   for p in map(json.loads, f) if p["question_id"] in rows}

    rng = random.Random(args.seed)
    wrong_ids = sorted(q for q, ok in correct.items() if not ok)
    wrong = rng.sample(wrong_ids, min(args.n, len(wrong_ids)))

    # right answers: same number per type as the wrong sample
    right_by_type = defaultdict(list)
    for qid in sorted(q for q, ok in correct.items() if ok):
        right_by_type[rows[qid][match_field]].append(qid)
    wrong_types = Counter(rows[qid][match_field] for qid in wrong)
    right = [qid for t, n in sorted(wrong_types.items()) for qid in rng.sample(right_by_type[t], n)]
    rng.shuffle(right)

    for label, qids in [("wrong", wrong), ("correct", right)]:
        with open(os.path.join(args.out_dir, file_name.format(label)), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows[qid] for qid in qids)
    print(f"{len(wrong_ids)} wrong of {len(correct)}; wrote {len(wrong)} wrong and {len(right)} right questions "
          f"to {args.out_dir}/{file_name.format('{wrong,correct}')}")
    print(f"{match_field} (same in both): " + ", ".join(f"{t} {n}" for t, n in sorted(wrong_types.items())))


if __name__ == "__main__":
    main()
