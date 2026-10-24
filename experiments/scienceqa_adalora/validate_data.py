"""Validate shared ScienceQA data/images and optionally the regenerated parquet outputs."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepared',type=Path);args=parser.parse_args()
    manifest=json.loads((ROOT/'experiments/scienceqa_adalora/input_manifest.json').read_text())
    for name,digest in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    folder=ROOT/'datasets/scienceqa'
    ids={}
    train=json.loads((folder/'train_llava.json').read_text());assert len(train)==6218
    images=args.prepared/'images' if args.prepared else folder/'images'
    for split,count in [('train',6218),('val',2097),('test',2017)]:
        name=f'{split}_questions.jsonl'
        rows=[json.loads(s) for s in (folder/name).read_text().splitlines()]
        assert len(rows)==len({r['question_id'] for r in rows})==count
        ids[split]={r['question_id'] for r in rows}
        for q in rows:
            assert q['answer'] in 'ABCDE' and len(q['answer'])==1
            assert (images/q['image']).is_file(),q['image']
        if args.prepared:
            other=[json.loads(s) for s in (args.prepared/name).read_text().splitlines()]
            assert other==rows,f'Regenerated {name} differs from team data'
    assert not ids['train']&ids['test'] and not ids['train']&ids['val'] and not ids['val']&ids['test']
    questions={q['question_id']:q for q in map(json.loads,(folder/'train_questions.jsonl').read_text().splitlines())}
    for r in train:
        q=questions[r['id']]
        assert r['image']==q['image']
        assert r['conversations'][0]['value']=='<image>\n'+q['text']
        assert r['conversations'][1]['value']==q['answer']
    if args.prepared:assert json.loads((args.prepared/'train_llava.json').read_text())==train
    assert len(list(images.glob('*.png')))==10332
    print('SCIENCEQA_DATA_VALIDATED: train6218 / val2097 / test2017, 10332 images',flush=True)
if __name__=='__main__':main()
