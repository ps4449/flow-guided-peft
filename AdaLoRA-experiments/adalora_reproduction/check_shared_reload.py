"""Verify the shared loop's adapter export against a fixed training input."""
import argparse
import json
from pathlib import Path
import torch
from peft import PeftModel
from common import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    _, base, _ = load_model('liuhaotian/llava-v1.5-7b')
    model = PeftModel.from_pretrained(base, str(run / 'adapter'), is_trainable=False)
    model.eval()
    batch = torch.load(run / 'reload_input.pt', map_location='cpu')
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    expected = torch.load(run / 'reload_logits.pt', map_location='cpu')
    with torch.no_grad():
        actual = model.get_base_model()(**batch, use_cache=False).logits[:, -8:, :].float().cpu()
    passed = bool(torch.isfinite(actual).all() and torch.allclose(actual, expected, atol=0.03, rtol=0.01))
    result = dict(passed=passed, max_abs_difference=float((actual-expected).abs().max()),
                  scope='last eight token positions on one training input; not an accuracy evaluation',
                  atol=0.03, rtol=0.01)
    (run / 'reload_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('Shared-loop checkpoint reload differs; stop before full training')


if __name__ == '__main__':
    main()
