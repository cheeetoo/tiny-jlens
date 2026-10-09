"""Fill template.html with data -> sketches.html (open in a browser; the figures are interactive).

Inputs: results/band/cka.json (GPT-2 CKA, with the final LayerNorm's gain folded into the J-lens
vectors), and in post/fig1/data/: ignition_grid.json (GPT-2
ignition, from ignition_grid.py) and paper_{cka,ignition}.json (Claude Sonnet 4.5: the data behind
the paper's Figs 27 and 29, from transformer-circuits.pub/2026/workspace/data/layer-diagram/cka.json
and .../data/ignition/data.json).
Run from the repo root:  python post/sketches/build.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = ROOT / "post/fig1/data"
ign = json.load(open(D / "ignition_grid.json"))
p_cka = json.load(open(D / "paper_cka.json"))
p_ign = json.load(open(D / "paper_ignition.json"))
g_cka = json.load(open(ROOT / "results/band/cka.json"))
r3 = lambda M: [[round(v, 3) for v in row] for row in M]
data = dict(
    claude=dict(cka=r3(p_cka["sim"]), phases=p_cka["phases"], ws_band=p_ign["ws_band"],
                share_full=r3(p_ign["heatmaps"]["proj"]), share_J=r3(p_ign["heatmaps"]["jspan"])),
    gpt2=dict(cka=r3(g_cka["linear_gain"]),
              share_full=r3(ign["share_full"]), share_J=r3(ign["share_J"])),
)
html = (HERE / "template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
(HERE / "sketches.html").write_text(html)
print("wrote", HERE / "sketches.html")
