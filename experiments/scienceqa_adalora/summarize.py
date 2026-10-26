"""Recompute three-seed ScienceQA results from predictions. Read-only unless --write is explicit."""
import argparse
import json
import statistics
from pathlib import Path
from validate_run import validate_run

def summarize(root):
    runs = {}
    for seed in (0, 1, 2):
        path = Path(root) / f'seed{seed}'
        runs[str(seed)] = {'path': str(path), **validate_run(path)}
        assert runs[str(seed)]['seed'] == seed
    assert len({r['regularization'] for r in runs.values()}) == 1
    assert len({r['final_rank_sum'] for r in runs.values()}) == 1
    common = set.intersection(*(set(r['evaluations']) for r in runs.values()))
    assert common, 'No shared evaluation split across all three seeds'
    summary = {}
    for split in sorted(common):
        values = [r['evaluations'][split]['accuracy'] for r in runs.values()]
        summary[split] = {'mean_accuracy': statistics.mean(values), 'sample_std_pp': statistics.stdev(values), 'seeds': [0, 1, 2]}
    return {'runs': runs, 'summary': summary, 'scope': 'Regularization-off chosen after exploratory seed-0 results; no new model selection or training.'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/peft/adalora/scienceqa_regoff_confirm'))
    parser.add_argument('--write', type=Path, help='Optional new summary file; refuses overwrite')
    args = parser.parse_args()
    result = summarize(args.root)
    if args.write:
        with args.write.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result['summary'], indent=2))
