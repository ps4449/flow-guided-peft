"""GPU forward-preparation check: verify that every collated sample survives LLaVA.

No training or adapter modification. Run inside an allocated GPU job using the
same environment and LLaVA checkout as the experiment being audited.
"""
import argparse,ast,hashlib,json,sys
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'peft_lora'))
from common import load_model,build_prompt,tokenize,load_image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-file',default='datasets/gqa/train_20000_llava.json')
    parser.add_argument('--image-folder',default=None)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--reproduce-legacy',action='store_true',help='Also omit modalities to audit this installed LLaVA revision')
    args=parser.parse_args()
    assert args.batch_size>0
    images=Path(args.image_folder or str(Path(args.train_file).parent/'images'))
    rows=json.loads(Path(args.train_file).read_text())[:args.batch_size]
    assert len(rows)==args.batch_size
    tokenizer,model,processor=load_model('liuhaotian/llava-v1.5-7b');model.eval()
    source=ast.parse((ROOT/'peft_lora/train_lora.py').read_text())
    function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='collate')
    ns=dict(torch=torch,pad_sequence=pad_sequence,IGNORE_INDEX=-100)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<shared collate>','exec'),ns)
    samples=[]
    for row in rows:
        human,answer=[c['value'] for c in row['conversations']]
        prefix=tokenize(build_prompt(human,None,'vicuna_v1'),tokenizer)
        ids=tokenize(build_prompt(human,answer,'vicuna_v1'),tokenizer)
        assert torch.equal(ids[:len(prefix)],prefix)
        labels=ids.clone();labels[:len(prefix)]=-100
        image,size=load_image(str(images/row['image']),processor,model.config)
        samples.append((ids,labels,image,size))
    batch=ns['collate'](samples,tokenizer.pad_token_id)
    batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
    expected_tokens=int((batch['labels']!=-100).sum());reports={}
    with torch.inference_mode():
        modes=['shared_collator']+(['legacy_without_modalities'] if args.reproduce_legacy else [])
        for mode in modes:
            kw=dict(input_ids=batch['input_ids'],position_ids=None,attention_mask=batch['attention_mask'],past_key_values=None,
                    labels=batch['labels'],images=batch['images'],image_sizes=batch['image_sizes'])
            if mode=='shared_collator':kw['modalities']=batch['modalities']
            result=model.prepare_inputs_labels_for_multimodal(**kw)
            reports[mode]=dict(input_rows=args.batch_size,embedding_rows=int(result[4].shape[0]),
                               label_rows=int(result[5].shape[0]),supervised_tokens=int((result[5]!=-100).sum()))
    actual=reports['shared_collator']
    assert actual['embedding_rows']==actual['label_rows']==args.batch_size,actual
    assert actual['supervised_tokens']==expected_tokens,actual
    import llava.model.llava_arch as arch
    path=Path(arch.__file__)
    print(json.dumps(dict(status='MULTIMODAL_BATCH_CHECK_PASSED',gpu=torch.cuda.get_device_name(),
                          llava_arch_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reports=reports),indent=2))

if __name__=='__main__':main()
