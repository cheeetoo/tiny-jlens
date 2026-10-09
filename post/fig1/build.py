"""Figure 1: template.html + data -> fig1.html, rendered to fig1.png with headless Chromium.

Data:
  data/paper_cka.json, data/paper_ignition.json   Claude Sonnet 4.5, the data behind the paper's
      Figs 27 and 29, from transformer-circuits.pub/2026/workspace/data/layer-diagram/cka.json
      and .../data/ignition/data.json
  ../../results/band/cka.json                      GPT-2 dictionary CKA (the final LayerNorm's gain folded in)
  data/ignition_grid.json                          GPT-2 ignition sweep (data/ignition_grid.py)
Run from the repo root:  .venv/bin/python post/fig1/build.py
"""
import json
import pathlib
import subprocess

from PIL import Image, ImageChops

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = HERE / "data"
CHROMIUM = "/Applications/Chromium.app/Contents/MacOS/Chromium"

p_cka = json.load(open(D / "paper_cka.json"))
p_ign = json.load(open(D / "paper_ignition.json"))
g_ign = json.load(open(D / "ignition_grid.json"))
g_cka = json.load(open(ROOT / "results/band/cka.json"))
r3 = lambda M: [[round(v, 3) for v in row] for row in M]
data = dict(
    claude=dict(cka=r3(p_cka["sim"]), ws_band=p_ign["ws_band"], ign=r3(p_ign["heatmaps"]["proj"])),
    gpt2=dict(cka=r3(g_cka["linear_gain"]), ign=r3(g_ign["share_full"])),
)
html = (HERE / "template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
(HERE / "fig1.html").write_text(html)


def render(part, out):
    url = (HERE / "fig1.html").as_uri() + (f"#{part}" if part else "")
    subprocess.run([CHROMIUM, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--window-size=1436,1300", f"--screenshot={out}", url], check=True, capture_output=True)
    # crop the white below the figure, keeping the same 2px (4px at 2x) margin as the other sides
    img = Image.open(out).convert("RGB")
    bottom = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()[3]
    img.crop((0, 0, img.width, bottom + 4)).save(out)


# fig1.png: the five functional tests; fig1_structure.png: the structure panel (e.g. for the thread);
# fig1_full.png: both together
render("tests", HERE / "fig1.png")
render("structure", HERE / "fig1_structure.png")
render("", HERE / "fig1_full.png")
print("wrote", HERE / "fig1.html", "fig1.png, fig1_structure.png, fig1_full.png")
