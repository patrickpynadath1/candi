import json,sys
from pathlib import Path
sys.path.insert(0,'/tmp/candi-distillation-recipe')
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import OmegaConf
root=Path('/home/patrick/.cache/duo/nested-tuning/feedback-depth-cdcd-20k-20261030')
out=Path('/home/patrick/.codex/visualizations/2026/10/11/01a12d34-6656-78b0-b299-814261a77f1e/simplex-final-comparison')
u=torch.linspace(0,1,1001); fig,axes=plt.subplots(2,2,figsize=(11,7)); info=[]
for j,task in enumerate(['sudoku_hard','text8']):
 curves=[]
 for variant in ['none','expected_embedding']:
  result=json.loads((root/f'{task}-simplex-{variant}-seed-1/final/result.json').read_text()); cp=result['selection']['checkpoint'];cfg=OmegaConf.load(cp['config']);ck=torch.load(cp['checkpoint'],map_location='cpu',weights_only=False)
  clock=instantiate(cfg.time_sampler);clock.load_state_dict({k.removeprefix('algorithm.time_sampler.'):v for k,v in ck['state_dict'].items() if k.startswith('algorithm.time_sampler.')});schedule=instantiate(cfg.model.views.content.schedule)
  with torch.no_grad():
   t=clock(u).clamp(0,1);t=torch.where((u==0)|(u==1),u,t);p=schedule(t)
  label='No SC' if variant=='none' else 'Expected-embedding SC'
  axes[0,j].plot(u,t,label=label);axes[1,j].plot(u,p.alpha,label=label)
  curves.append(t);info.append(dict(task=task,variant=variant,step=ck['global_step'],checkpoint=cp['checkpoint'],points={str(q):{'t':float(t[int(q*1000)]),'alpha':float(p.alpha[int(q*1000)]),'concentration':float(p.concentration[int(q*1000)])} for q in [.1,.25,.5,.75,.9]}))
  del ck
 print(task,'mean_abs_time_diff',float((curves[0]-curves[1]).abs().mean()),'max_abs_time_diff',float((curves[0]-curves[1]).abs().max()))
 axes[0,j].plot(u,u,':',color='gray',label='Unwarped');axes[1,j].plot(u,schedule(u).alpha,':',color='gray',label='Unwarped');axes[0,j].set_title(task)
 for ax in axes[:,j]:ax.set_xlabel('Global time u (clean → noise)');ax.grid(alpha=.2);ax.legend()
 axes[0,j].set_ylabel('Physical corruption time t(u)');axes[1,j].set_ylabel('Simplex clean coefficient α')
fig.suptitle('Simplex: final 20k-step learned schedules (raw checkpoint weights)');fig.tight_layout();fig.savefig(out/'schedules.png',dpi=150);(out/'metadata.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
