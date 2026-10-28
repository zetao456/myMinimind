"""Select preference training configurations using validation accuracy only."""
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from experiments.preference_evaluate import load_policy, evaluate_likelihood


def main():
    torch.set_num_threads(4)
    output=ROOT/'experiments/runs/zh_preference_tuning'
    output.mkdir(parents=True,exist_ok=True)
    data=ROOT/'dataset/prepared/zh_preference'
    config=json.loads((ROOT/'configs/experiments/zh_preference.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    model=load_policy(config,ROOT/config['base_path'],'cuda:0')
    baseline,pairs=evaluate_likelihood(model,tokenizer,data/'validation.jsonl',config,'cuda:0')
    del model;gc.collect();torch.cuda.empty_cache()
    print('Validation baseline:',baseline['preference_accuracy_normalized'],flush=True)
    (output/'baseline_validation.json').write_text(json.dumps(baseline,indent=2))
    trials=[('dpo_stronger',3e-5,.1)]
    history=[]
    best=baseline['preference_accuracy_normalized']
    for name,lr,beta in trials:
        directory=output/name
        directory.mkdir(exist_ok=True)
        for steps in [64,128,256,512]:
            command=[sys.executable,str(ROOT/'trainer/train_dpo.py'),
                '--data_path',str(data/'train.jsonl'),'--device','cuda:0','--dtype','float32',
                '--epochs','4','--batch_size','4','--accumulation_steps','2','--num_workers','0',
                '--learning_rate',str(lr),'--beta',str(beta),'--max_seq_len','768','--max_prompt_len','256',
                '--max_steps',str(steps),'--save_dir',str(directory),'--checkpoint_dir',str(directory),
                '--save_weight',name,'--log_interval','8','--save_interval','64']
            if steps>64:command+=['--from_resume','1']
            (directory/f'command_{steps}.json').write_text(json.dumps(command,indent=2))
            print(f'Training {name} to {steps} updates',flush=True)
            with (directory/'train.log').open('a') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                    env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
            checkpoint=directory/f'{name}_768_resume.pth'
            model=load_policy(config,ROOT/config['base_path'],'cuda:0',checkpoint)
            scores,_=evaluate_likelihood(model,tokenizer,data/'validation.jsonl',config,'cuda:0',pairs)
            del model;gc.collect();torch.cuda.empty_cache()
            accuracy=scores['preference_accuracy_normalized']
            row={'trial':name,'updates':steps,'validation':scores}
            history.append(row)
            print(f'VALIDATION {name} {steps}: {accuracy:.2%} ({accuracy*128:.0f}/128), baseline {baseline["preference_accuracy_normalized"]:.2%}',flush=True)
            if accuracy>best:
                best=accuracy
                shutil.copyfile(checkpoint,output/'best_resume.pth')
                (output/'best.json').write_text(json.dumps(row,indent=2))
            (output/'validation_history.json').write_text(json.dumps(history,indent=2))
    print('Validation-selected best:',best,flush=True)
    if (output/'best_resume.pth').exists():
        model=load_policy(config,ROOT/config['base_path'],'cuda:0',output/'best_resume.pth')
        test,_=evaluate_likelihood(model,tokenizer,data/'test.jsonl',config,'cuda:0',
            [json.loads(x) for x in (ROOT/'experiments/runs/zh_preference_20261029/baseline/pairs.jsonl').read_text().splitlines()])
        (output/'selected_test.json').write_text(json.dumps(test,indent=2))
        print('FINAL TEST:',test,flush=True)


if __name__=='__main__':main()
