"""Short, inference-only validation of original adapters versus merged weights."""
import json,random,sys,time
from pathlib import Path
import torch
from peft import PeftModel
from common import load_model,load_image,build_prompt,tokenize
from llava.constants import DEFAULT_IMAGE_TOKEN
ROOT=Path(__file__).resolve().parents[2]

def predictions(base,questions,tokenizer,processor):
    base.eval();result=[]
    with torch.inference_mode():
        for q in questions:
            image,size=load_image(str(ROOT/'datasets/scienceqa/images'/q['image']),processor,base.config)
            ids=tokenize(build_prompt(DEFAULT_IMAGE_TOKEN+'\n'+q['text'],None,'vicuna_v1'),tokenizer).unsqueeze(0).cuda()
            answer=base.generate(ids,images=image.unsqueeze(0).cuda(),image_sizes=[size],do_sample=False,max_new_tokens=16,use_cache=True)
            result.append(dict(question_id=q['question_id'],prediction=tokenizer.batch_decode(answer,skip_special_tokens=True)[0].strip(),answer=q['answer']))
    return result

def main():
    questions=[json.loads(s) for s in (ROOT/'datasets/scienceqa/val_questions.jsonl').read_text().splitlines()]
    indices=sorted(random.Random(6130).sample(range(len(questions)),64));questions=[questions[i] for i in indices]
    output=ROOT/'output/diagnostics/scienceqa_inference_integrity';output.mkdir(parents=True,exist_ok=True)
    reports={}
    for method in ['uniform','adalora']:
        tokenizer,base,processor=load_model('liuhaotian/llava-v1.5-7b')
        adapter=ROOT/f'output/peft/{method}/scienceqa_train_llava/seed0/adapter'
        model=PeftModel.from_pretrained(base,str(adapter));model.eval()
        dtypes={str(p.dtype) for n,p in model.named_parameters() if 'lora_' in n}
        unmerged=predictions(model.get_base_model(),questions,tokenizer,processor)
        merged=model.merge_and_unload()
        after=predictions(merged,questions,tokenizer,processor)
        norm=lambda s:s.strip().rstrip('.').lower()
        differences=[dict(question_id=a['question_id'],unmerged=a['prediction'],merged=b['prediction'],answer=a['answer']) for a,b in zip(unmerged,after) if norm(a['prediction'])!=norm(b['prediction'])]
        report=dict(total=64,adapter_dtypes=sorted(dtypes),prediction_differences=len(differences),differences=differences,
                    unmerged_correct=sum(norm(p['prediction'])==norm(p['answer']) for p in unmerged),
                    merged_correct=sum(norm(p['prediction'])==norm(p['answer']) for p in after))
        reports[method]=report
        (output/(method+'_predictions.json')).write_text(json.dumps(dict(unmerged=unmerged,merged=after),indent=2)+'\n')
        print(method,json.dumps(report),flush=True)
        del model,base,merged,tokenizer,processor;torch.cuda.empty_cache()
    result=dict(scope='Seed0,64 fixed validation questions, inference-only. Tests merged evaluation semantics, not retraining or hyperparameter selection.',methods=reports)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('INFERENCE_INTEGRITY_DIAGNOSTIC_COMPLETE',flush=True)
if __name__=='__main__':main()
