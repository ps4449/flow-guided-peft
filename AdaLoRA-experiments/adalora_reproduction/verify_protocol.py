"""Check frozen sources and optional runtime versions without loading a model."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-env',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,name
    if args.check_env:
        for package,expected in manifest['training_environment'].items():
            actual=importlib.metadata.version(package)
            assert actual==expected,f'{package}: expected {expected}, found {actual}'
    print('FROZEN_ADALORA_PROTOCOL_VERIFIED')

if __name__=='__main__':main()
