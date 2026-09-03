"""Criterion 3, E3 with both swap operations (run from v2/).  Same items, partners and
partner rule as run.py (target answer starts at output rank >= 10).  Writes swap_ops.json.
Output on gpt2-small: coordinate swap 705/999 = 70.6%, subtract-and-add 854/999 = 85.5%."""
import json, sys, torch, time
sys.path.insert(0,'.'); sys.path.insert(0,'c3_reasoning')
from jl.model import Lensed, ranks_of
from jl.interventions import swap_edits
from swaps import coord_swap_edits
torch.set_num_threads(1)
import os
lm=Lensed(device=os.environ.get('DEVICE','cpu')); band=[7,8,9]
items=json.load(open('c3_reasoning/results/prompts.json'))
for it in items:
    it['ids']=lm.encode(it['prompt'].rstrip()); it['clean']=lm.residuals(it['ids'],band); it['lg']=lm.logits(it['ids'])[-1]
    it['s']=lm.tid(' '+it['intermediate']); it['a']=lm.tid(' '+it['answer'])
res={'subadd':[], 'coord':[]}; t0=time.time(); n=0
for it in items:
    for p in items:
        if p['family']!=it['family'] or p['intermediate']==it['intermediate'] or p['answer']==it['answer']: continue
        before=int(ranks_of(it['lg'],[p['a']])[0])
        if before<10: continue
        for name,ed in [('subadd',swap_edits(lm,it['ids'],it['s'],p['s'],band,clean=it['clean'])),('coord',coord_swap_edits(lm,it['ids'],it['s'],p['s'],band,clean=it['clean']))]:
            lg=lm.logits(it['ids'],ed)[-1]; res[name].append((it['family'], int(lg.argmax())==p['a']))
        n+=1
print('trials',n,'time',round(time.time()-t0))
json.dump(res,open('c3_reasoning/results/swap_ops.json','w'))
fams=sorted({f for f,_ in res['subadd']})+['all']
for name in res:
    for fam in fams:
        r=[h for f,h in res[name] if fam=='all' or f==fam]; print(f'{name:7s} {fam:14s} {sum(r)}/{len(r)} = {100*sum(r)/len(r):.1f}%')
