"""Fail immediately if multimodal preparation silently drops training rows."""
import functools,inspect,json
from pathlib import Path

def install(model):
    base=model.get_base_model();original=base.prepare_inputs_labels_for_multimodal
    signature=inspect.signature(original)
    stats=dict(forward_calls=0,input_rows=0,language_rows=0,supervised_tokens=0,all_batch_sizes_match=True)
    @functools.wraps(original)
    def checked(*args,**kwargs):
        bound=signature.bind_partial(*args,**kwargs)
        ids=bound.arguments.get('input_ids')
        expected=int(ids.shape[0])
        result=original(*args,**kwargs)
        embeds,labels=result[4],result[5]
        assert embeds is not None and labels is not None,'Expected multimodal supervised training batch'
        assert embeds.shape[0]==labels.shape[0]==expected,('Multimodal batch truncated',expected,embeds.shape[0],labels.shape[0])
        stats['forward_calls']+=1;stats['input_rows']+=expected;stats['language_rows']+=int(embeds.shape[0])
        stats['supervised_tokens']+=int((labels!=-100).sum())
        return result
    base.prepare_inputs_labels_for_multimodal=checked
    return stats

def save(stats,folder):
    assert stats['input_rows']==stats['language_rows']==6208,stats
    assert stats['supervised_tokens']==12416,stats
    (Path(folder)/'batch_integrity.json').write_text(json.dumps(stats,indent=2)+'\n')
    print('FULL_BATCH_INTEGRITY_PASSED',json.dumps(stats),flush=True)
