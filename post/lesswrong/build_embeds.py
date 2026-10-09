"""Split post/sketches/sketches.html into one self-contained HTML file per figure, for embedding.

Writes fig_cka.html (the band of layers: CKA for Claude and GPT-2, with a colour scale below) and
fig_ignition.html (ignition, with the whole-activation / J-lens-only toggle) next to this script.
Each file scales its figure down to fit the width it is shown at (never up).
Run from the repo root after post/sketches/build.py:  python post/lesswrong/build_embeds.py
"""
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
src = (HERE.parent / "sketches/sketches.html").read_text()

FIT_CSS = """
  html, body { margin: 0; overflow-x: hidden; }
  main { width: max-content; padding: 12px 16px; transform-origin: 0 0; }
  figure { margin: 0; }
"""
FIT_JS = """
<script>
// Scale the figure down to the width it is shown at.
(function () {
  const m = document.querySelector("main");
  function fit() {
    m.style.transform = "";
    const s = Math.min(1, document.documentElement.clientWidth / m.offsetWidth);
    m.style.transform = `scale(${s})`;
    document.body.style.height = Math.ceil(m.offsetHeight * s) + "px";
  }
  addEventListener("resize", fit);
  fit();
})();
</script>
"""


def cut(text, start, end):
    """Remove text from the first `start` up to (not including) the next `end`."""
    i = text.index(start)
    j = text.index(end, i + len(start))
    return text[:i] + text[j:]


def make(title, drop_fig, drop_js):
    html = src.replace("<title>Structure sketches</title>", f"<title>{title}</title>")
    html = html.replace("</style>", FIT_CSS + "</style>", 1)
    html = re.sub(r"\n\s*main \{ width: 1500px;[^}]*\}", "", html)  # replaced by FIT_CSS
    html = cut(html, drop_fig[0], drop_fig[1])
    for a, b in drop_js:
        html = cut(html, a, b)
    return html.replace("</body>", FIT_JS + "</body>")


F1 = "<!-- =================================================================== Figure 1: CKA -->"
F2 = "<!-- =================================================================== Figure 2: ignition -->"
JS_SCHEM = "// ------------------------------------------------------------------ Figure 1 schematic"
JS_PANEL = "// ------------------------------------------------------------------ shared heatmap panel"
JS_F1 = "// ------------------------------------------------------------------ Figure 1 heatmaps"
JS_F2 = "// ------------------------------------------------------------------ Figure 2 heatmaps"

cka = make("Band of workspace layers", (F2, "</main>"), [(JS_F2, "</script>")])
ign = make("Ignition", (F1, F2), [(JS_SCHEM, JS_PANEL), (JS_F1, JS_F2)])


def sub(html, old, new):
    assert old in html, old
    return html.replace(old, new, 1)


# Both: white like LessWrong, no colour bar beside the panels and no padding at the edges, so the heatmaps
# get more of the width; a bar spanning the row on top; larger text in the panels.
TIGHT_CSS = """
  :root { --bg: #fff; }
  html, body { overflow: hidden; }
  main { padding: 0; }
  .setup { width: 0; min-width: 100%; margin: 0 0 10px; padding: 10px 14px; flex-wrap: wrap; row-gap: 8px; justify-content: space-between; }
  .setup .ex { font-size: 17px; }
  .seg button { font-size: 14.5px; padding: 5px 11px; }
  svg .tick { font-size: 12.5px; }
  svg .axl { font-size: 13px; }
  svg .model { font-size: 16px; }
  svg .phase { font-size: 14px; }
"""
ROW = """  <div class="row">
    <svg id="{p}-claude" width="452" height="394"></svg>
    <svg id="{p}-gpt2" width="440" height="394" style="margin-left:12px"></svg>
  </div>"""

# Ignition: the bar is the example sentence and the toggle.
ign = sub(ign, "</style>", TIGHT_CSS + "</style>")
ign = sub(ign, """  <div class="row">
    <svg id="f2-claude" width="440" height="414"></svg>
    <svg id="f2-gpt2" width="560" height="414" style="margin-left:40px"></svg>
  </div>""", ROW.format(p="f2"))
ign = sub(ign, '    <div class="note">input embedding = (1−α)·<span class="cb">B</span> + α·<span class="ca">A</span></div>\n', "")
ign = sub(ign, '<div class="ctl" style="margin:0 0 0 12px"><span>measure</span>', '<div class="ctl" style="margin:0">')
ign = sub(ign, '    cbar: [[[0, "pure B"], [.5, "½"], [1, "pure A"]], "position between pure B and pure A"],\n', "")

# CKA: no schematic and no bar on top; a small colour scale under the panels (CKA values matter here).
CKA_CSS = """
  .scale { display: flex; align-items: center; gap: 8px; font-size: 14.5px; color: var(--muted); }
  .scale .grad { width: 170px; height: 12px; border-radius: 2px; background: linear-gradient(90deg, #440154, #482475, #414487, #355f8d, #2a788e, #21918c, #22a884, #44bf70, #7ad151, #bddf26, #fde725); }
"""
cka = sub(cka, "</style>", TIGHT_CSS + CKA_CSS + "</style>")
fig = cka[cka.index('<figure id="f1">'):cka.index("</figure>") + len("</figure>")]
cka = sub(cka, fig, """<figure id="f1">
""" + ROW.format(p="f1") + """
  <div class="scale" style="margin: 2px 0 0 52px"><span>CKA similarity</span><span>0</span><span class="grad"></span><span>1</span></div>
</figure>""")
cka = cut(cka, JS_SCHEM, JS_PANEL)
cka = sub(cka, '  cbar: [[[0, "0.0"], [.2, "0.2"], [.4, "0.4"], [.6, "0.6"], [.8, "0.8"], [1, "1.0"]], "CKA similarity"],\n', "")
(HERE / "fig_cka.html").write_text(cka)
(HERE / "fig_ignition.html").write_text(ign)
print("wrote", HERE / "fig_cka.html", "and", HERE / "fig_ignition.html")
