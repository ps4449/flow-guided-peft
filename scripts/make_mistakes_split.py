"""Build a training file focused on the questions zero-shot LLaVA gets wrong.

Step 1 (Mac or cluster): turn the training file into questions eval_gqa.py can answer
  python3 scripts/make_mistakes_split.py questions
  -> datasets/gqa/train_20000_questions.jsonl

Step 2 (cluster GPU): answer them with zero-shot LLaVA
  sbatch --export=ALL,EXTRA="--questions datasets/gqa/train_20000_questions.jsonl --out-dir output/peft/zero_shot/eval_train20000" scripts/tc2_eval.sh

Step 3 (Mac or cluster): keep every wrong question plus a sample of right ones (so it does not forget)
  python3 scripts/make_mistakes_split.py build [--right-ratio 1.0] [--seed 0]
  -> datasets/gqa/train_mistakes_llava.json

Uses only the 20k training questions already sampled, so no new images are needed.
"""
import argparse
import json
import random

PREFIX = "<image>\n"


def normalize(answer):
    return answer.strip().rstrip(".").lower()


def make_questions(args):
    with open(args.train_file) as f:
        rows = json.load(f)
    with open(args.questions, "w") as f:
        for r in rows:
            human, answer = r["conversations"][0]["value"], r["conversations"][1]["value"]
            f.write(json.dumps({"question_id": r["id"], "image": r["image"],
                                "text": human[len(PREFIX):] if human.startswith(PREFIX) else human,
                                "answer": answer}) + "\n")
    print(f"Wrote {len(rows)} questions to {args.questions}")


def build(args):
    with open(args.train_file) as f:
        rows = {r["id"]: r for r in json.load(f)}
    with open(args.predictions) as f:
        preds = [json.loads(line) for line in f]
    if len(preds) != len(rows):
        print(f"Warning: {len(preds)} predictions for {len(rows)} training questions")

    wrong = sorted(p["question_id"] for p in preds if normalize(p["prediction"]) != normalize(p["answer"]))
    right = sorted(p["question_id"] for p in preds if normalize(p["prediction"]) == normalize(p["answer"]))
    n_right = min(len(right), round(args.right_ratio * len(wrong)))
    rng = random.Random(args.seed)
    keep = wrong + rng.sample(right, n_right)
    rng.shuffle(keep)

    with open(args.out, "w") as f:
        json.dump([rows[qid] for qid in keep], f)
    print(f"Zero-shot accuracy on the training questions: {100 * len(right) / len(preds):.2f}%")
    print(f"Wrote {len(keep)} questions to {args.out}: {len(wrong)} wrong + {n_right} right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["questions", "build"])
    parser.add_argument("--train-file", default="datasets/gqa/train_20000_llava.json")
    parser.add_argument("--questions", default="datasets/gqa/train_20000_questions.jsonl")
    parser.add_argument("--predictions", default="output/peft/zero_shot/eval_train20000/predictions.jsonl")
    parser.add_argument("--right-ratio", type=float, default=1.0, help="right questions kept per wrong one")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="datasets/gqa/train_mistakes_llava.json")
    args = parser.parse_args()
    make_questions(args) if args.step == "questions" else build(args)


if __name__ == "__main__":
    main()
