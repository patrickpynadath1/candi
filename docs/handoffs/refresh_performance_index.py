import csv,json
from pathlib import Path
from datetime import datetime,timezone
r=Path('/home/patrick/.cache/duo');out=Path(__file__).resolve().parent;rows=[];counts=[]
for plan,result in [('feedback-depth-20k-20261028','feedback-depth-20k-20261028'),('feedback-depth-cdcd-20k-20261030','feedback-depth-cdcd-20k-20261030'),('feedback-geometry-cdcd-20k-20261030','feedback-depth-cdcd-20k-20261030'),('feedback-renoising-128nfe-20261031','feedback-renoising-128nfe-20261031')]:
 cases=json.loads((r/'plans'/plan/'manifest.json').read_text());done=0
 reuse_file=r/'plans'/plan/'reused-results.json'
 reused={x['name']:x['result'] for x in json.loads(reuse_file.read_text())} if reuse_file.exists() else {}
 for c in cases:
  f=r/'nested-tuning'/result/c['name']/'final/result.json'
  is_reused=False
  if not f.exists() and c['name'] in reused:
   f=Path(reused[c['name']]);is_reused=True
  if not f.exists():continue
  d=json.loads(f.read_text());s=d['selection'];cp=s['checkpoint'];done+=1
  rows.append(dict(campaign=plan,case=c['name'],reused=is_reused,final_score=d['score'],validation_score=s['score'],sampler=s['policy']['name'],inference_parameters=json.dumps(s['inference_parameters'],sort_keys=True),final_seed_scores=json.dumps(d['seed_scores']),checkpoint=cp['checkpoint'],checkpoint_config=cp['config'],final_result=str(f),case_directory=str(f.parents[1])))
 counts.append(dict(campaign=plan,completed=done,total=len(cases)))
with (out/'performance-index-2026-11-01.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n");w.writeheader();w.writerows(rows)
(out/'performance-index-2026-11-01.json').write_text(json.dumps({'snapshot_utc':datetime.now(timezone.utc).isoformat(),'counts':counts,'results':rows},indent=2)+'\n')
print(counts)
