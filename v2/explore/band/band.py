"""Workspace-band statistics for gpt2-small (paper Fig 28 metrics; run from v2/).
48 wikitext-103 validation sequences x 128 tokens.  Per layer: lens-vs-model top-1/top-10
agreement, excess kurtosis of the lens readout, top-1 autocorrelation (lags 1-3) vs a shuffled
null, effective dimensionality of the centered dictionary, and linear CKA between layer
dictionaries (see cka.py for why linear CKA is uninformative here).  Writes band.json."""
import sys, torch, json, random, time
sys.path.insert(0,'.')
from jl.model import Lensed
from datasets import load_dataset
torch.set_num_threads(1); random.seed(0); torch.manual_seed(0)
import os
lm=Lensed(device=os.environ.get('DEVICE','cpu')); LAYERS=list(range(12))
ds=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1',split='validation')
texts=[t for t in ds['text'] if len(t)>600]; random.shuffle(texts); texts=texts[:48]
seqs=[]
for t in texts:
    ids=lm.tok(t,add_special_tokens=False).input_ids[:127]; seqs.append(torch.tensor([[lm.bos]+ids]))
t0=time.time()
stats={L:dict(top1=0,top10=0,n=0,kurt=[],auto=[],null=[]) for L in LAYERS}
for ids in seqs:
    res=lm.residuals(ids,LAYERS); out=lm.logits(ids)
    model_top1=out[:-1].argmax(-1)                        # prediction at each position
    for L in LAYERS:
        lg=lm.lens_logits(res[L],L)                        # [T, vocab]
        lens_top=lg[:-1].topk(10).indices
        stats[L]['top1']+=int((lens_top[:,0]==model_top1).sum()); stats[L]['top10']+=int((lens_top==model_top1[:,None]).any(1).sum()); stats[L]['n']+=len(model_top1)
        z=(lg-lg.mean(-1,keepdim=True))/lg.std(-1,keepdim=True); stats[L]['kurt'].append(float(((z**4).mean(-1)-3).mean()))
        t1=lg.argmax(-1); T=len(t1)
        for d in [1,2,3]:
            stats[L]['auto'].append(float((t1[:-d]==t1[d:]).float().mean()))
            perm=t1[torch.randperm(T)]; stats[L]['null'].append(float((perm[:-d]==perm[d:]).float().mean()))
print('forward stats done',round(time.time()-t0),'s')
# dictionary geometry on a 4000-token vocab subsample: effective dimensionality + CKA between layers
idx=torch.randperm(lm.vocab)[:4000]
D={L: lm.V(L)[idx] for L in LAYERS}
def effdim(V):
    s=torch.linalg.svdvals(V-V.mean(0)); c=(s**2).cumsum(0)/(s**2).sum(); return float((c<0.9).sum()+1)/V.shape[1]
def cka(A,B):
    A=A-A.mean(0); B=B-B.mean(0); K=A@A.T; Lk=B@B.T
    return float((K*Lk).sum()/((K*K).sum().sqrt()*(Lk*Lk).sum().sqrt()))
ed={L:effdim(D[L]) for L in LAYERS}
C=[[cka(D[a],D[b]) for b in LAYERS] for a in LAYERS]
print('\n L  top1  top10  exKurt  autocorr  null   effdim(90%var)')
for L in LAYERS:
    s=stats[L]; print(f'{L:2d}  {s["top1"]/s["n"]:.2f}  {s["top10"]/s["n"]:.2f}   {sum(s["kurt"])/len(s["kurt"]):6.1f}    {sum(s["auto"])/len(s["auto"]):.3f}   {sum(s["null"])/len(s["null"]):.3f}   {ed[L]:.2f}')
print('\nCKA between layer dictionaries (rows/cols = layers 0..11)')
for a in LAYERS: print(f'{a:2d} '+' '.join(f'{C[a][b]:.2f}' for b in LAYERS))
json.dump(dict(stats={L:dict(top1=s['top1']/s['n'],top10=s['top10']/s['n'],kurt=sum(s['kurt'])/len(s['kurt']),auto=sum(s['auto'])/len(s['auto']),null=sum(s['null'])/len(s['null'])) for L,s in stats.items()},effdim=ed,cka=C),open('explore/band/band.json','w'))
