"""Create the exact separate capitalization dataset; never overwrite shared data."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'datasets/gqa/train_20000_llava.json')
    parser.add_argument('--output',type=Path,default=ROOT/'datasets/gqa/train_20000_capfirst_diagnostic.json')
    args=parser.parse_args()
    assert args.source.resolve()!=args.output.resolve(), 'Use a separate output file'
    rows=json.loads(args.source.read_text())
    assert len(rows)==20000
    for row in rows:
        answer=row['conversations'][1]['value']
        assert answer
        row['conversations'][1]['value']=answer[:1].upper()+answer[1:]
    content=(json.dumps(rows,ensure_ascii=False,indent=2)+'\n').encode()
    expected=json.loads((HERE/'source_manifest.json').read_text())['capitalized_data_sha256']
    assert hashlib.sha256(content).hexdigest()==expected, 'Training data differs from the completed experiment'
    if args.output.exists():
        assert args.output.read_bytes()==content, 'Refusing to overwrite different data'
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_bytes(content)
    print('CAPFIRST_DATA_VERIFIED',len(rows),expected)

if __name__=='__main__':main()
