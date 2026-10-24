"""Static Uniform control through the unchanged AdaLoRA engine, with row-count instrumentation."""
import functools,inspect,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"experiments/adalora_reproduction"))
import adalora_train_engine as engine
import batch_guard
original=engine.get_peft_model

def attach(*args,**kwargs):
    model=original(*args,**kwargs)
    stats=batch_guard.install(model)
    base=model.get_base_model();checked=base.prepare_inputs_labels_for_multimodal
    signature=inspect.signature(checked)
    argv=sys.argv
    folder=Path(argv[argv.index('--out-dir')+1])
    limit=int(argv[argv.index('--limit')+1]) if '--limit' in argv else 6218
    target=(limit//16)*16
    @functools.wraps(checked)
    def measured(*a,**kw):
        # The engine creates an unlabelled reload reference after training.
        if signature.bind_partial(*a,**kw).arguments.get("labels") is None:
            return checked.__wrapped__(*a,**kw)
        result=checked(*a,**kw)
        if stats['input_rows']==target:
            assert stats['language_rows']==target and stats['supervised_tokens']==2*target,stats
            (folder/'batch_integrity.json').write_text(json.dumps({**stats,'scope':'Training forwards only; excludes subsequent reload-reference forward'},indent=2)+'\n')
        return result
    base.prepare_inputs_labels_for_multimodal=measured
    return model
engine.get_peft_model=attach
if __name__=='__main__':
    if '--help' not in sys.argv and ('--batch-size' not in sys.argv or sys.argv[sys.argv.index('--batch-size')+1]!='1'):raise ValueError('This control requires microbatch 1')
    engine.main()
