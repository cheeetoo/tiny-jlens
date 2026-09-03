"""Recompute C4 swap rates under each gating class, split by function type. Run from v2/.  Shows the strict-gating 18% is composition (successor functions), not gating."""
import json
from collections import defaultdict
r=json.load(open('c4_generalization/results/results.json'))
sw=[x for x in r['swap'] if x['distinct'] and not x['echo']]
succ={('months','next_month'),('numbers','successor')}
def rate(s): h=sum(x['subadd']['1.0']['hit'] for x in s); return f'{h}/{len(s)} = {100*h/max(len(s),1):.0f}%'
both=lambda x: x['source_gated'] and x['target_gated']; tgt=lambda x: x['target_gated']
for label,sel in [('ungated',lambda x:True),('target-gated',tgt),('both-gated',both)]:
    print(f'{label:13s} all: {rate([x for x in sw if sel(x)]):18s} non-successor fns: {rate([x for x in sw if sel(x) and (x["category"],x["function"]) not in succ]):18s} successor fns: {rate([x for x in sw if sel(x) and (x["category"],x["function"]) in succ])}')
print('\nper function, hits/pairs by gate class [both | target-only | source-only | neither]')
d=defaultdict(lambda: defaultdict(lambda:[0,0]))
for x in sw:
    k=(x['category'],x['function']); cls='both' if both(x) else 'tgt' if tgt(x) else 'src' if x['source_gated'] else 'none'
    d[k][cls][0]+=x['subadd']['1.0']['hit']; d[k][cls][1]+=1
for k in sorted(d): print(f"  {k[0]:9s}/{k[1]:12s}", '  '.join(f"{c}:{d[k][c][0]}/{d[k][c][1]}" for c in ['both','tgt','src','none'] if d[k][c][1]))
