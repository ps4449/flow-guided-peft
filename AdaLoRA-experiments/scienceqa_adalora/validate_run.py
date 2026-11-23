"""Validate ScienceQA AdaLoRA artifacts without a GPU; raw historical metadata is preserved."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(path):
    return json.loads(Path(path).read_text())

def normalize(text):
    return text.strip().rstrip('.').lower()

def validate_run(folder, historical=False):
    folder = Path(folder)
    config = read(folder / 'train_config.json')
    adapter = read(folder / 'adapter/adapter_config.json')
    assert config['train_file'] == 'datasets/scienceqa/train_llava.json'
    assert config['image_folder'] == 'datasets/scienceqa/images'
    assert config['batch_size'] == 1 and config['grad_accum'] == 16
    assert config['epochs'] == 1 and config['lr'] == 2e-4 and config['warmup_ratio'] == .03
    assert config['seed'] in (0, 1, 2)
    expected_rows = (config.get('limit') or 6218) // 16 * 16
    assert config['total_steps'] == expected_rows // 16
    target = adapter['target_r'] * 64
    assert target in (512, 768) and adapter['init_r'] == 12
    assert adapter['orth_reg_weight'] in (0, .5)
    mask = read(folder / 'adapter/rank_mask.json')
    assert len(mask) == 64 and sum(map(sum, mask.values())) == target
    logs = [json.loads(line) for line in (folder / 'train_log.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs] == list(range(1, config['total_steps'] + 1))
    assert all(math.isfinite(r['task_loss']) and math.isfinite(r['grad_norm']) for r in logs)
    assert config['final_rank']['retained_rank_sum'] == target
    if adapter['orth_reg_weight'] == 0:
        assert all(abs(r['regularization']) < 1e-7 for r in logs)
    else:
        assert all(r['regularization'] > 0 for r in logs)
    if target == 768:
        assert all(r['retained_rank_sum'] == 768 for r in logs)
    integrity = folder / 'batch_integrity.json'
    if integrity.exists():
        counts = read(integrity)
        assert counts['input_rows'] == counts['language_rows'] == expected_rows
        assert counts['supervised_tokens'] == 2 * expected_rows
    else:
        assert historical, 'New runs must contain batch_integrity.json'
    assert read(folder / 'reload_check.json')['passed']
    evaluations = {}
    for path in folder.glob('*/metrics.json'):
        split = 'val' if path.parent.name in ('eval_val', 'eval_scienceqa_val') else 'test'
        refs = {r['question_id']: r['answer'] for r in map(json.loads, (ROOT / f'datasets/scienceqa/{split}_questions.jsonl').read_text().splitlines())}
        rows = list(map(json.loads, (path.parent / 'predictions.jsonl').read_text().splitlines()))
        assert len(rows) == len({r['question_id'] for r in rows}) == len(refs)
        assert {r['question_id']: r['answer'] for r in rows} == refs
        assert all(normalize(r['prediction']) in 'abcde' and len(normalize(r['prediction'])) == 1 for r in rows)
        correct = sum(normalize(r['prediction']) == normalize(r['answer']) for r in rows)
        metric = read(path)
        assert metric['correct'] == correct and metric['total'] == len(refs)
        assert metric['accuracy'] == round(100 * correct / len(refs), 2)
        evaluations[path.parent.name] = {'correct': correct, 'total': len(refs), 'accuracy': 100 * correct / len(refs)}
    return {'seed': config['seed'], 'regularization': adapter['orth_reg_weight'], 'final_rank_sum': target, 'training_rows': expected_rows, 'evaluations': evaluations}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--historical', action='store_true', help='Allow original batch-1 runs predating row-counter sidecars')
    args = parser.parse_args()
    print(json.dumps(validate_run(args.run, args.historical), indent=2))
