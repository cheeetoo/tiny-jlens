"""Shared loading and scoring for the directed-modulation figures.

A grid run (`python -m jl.control --model {gpt2,qwen} modulation_grid`) keeps, for every trial, the
best rank of a tracked token at every lens layer and every position of the copied sentence.  The
paper's score for a trial is a hit: a tracked token at lens rank 1 at any layer of the band and any
position.  `best` gives the best rank over a set of layers, so a hit at any rank threshold and any
band is `best(...) <= k`.
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CONDS = ["focus", "mention", "negated-think", "dismissal"]          # the order of the paper's Fig. 65
COND_LABEL = {"baseline": "no instruction", "focus": "think about", "mention": "mention",
              "negated-think": "don't think about", "dismissal": "ignore"}
FAMILIES = ["topic", "math"]
FAMILY_LABEL = {"topic": "Category instance", "math": "Math expression"}
NO_RANK = 10 ** 6

# Layers read for the hit rate.  `ours` is the band each model's other results use.  `paper` is the
# paper's band, 38% to 92% of depth, in each model's layers (layer L of 0..N-1 is at 100 L / (N-1)).
BANDS = {
    "gpt2": {"ours": (7, 9), "paper": (5, 10), "all": (0, 10)},
    "qwen": {"ours": (15, 22), "paper": (9, 21), "all": (0, 22)},
    # Gemma 3: no band of our own, so `ours` is the paper's.  `late` is where the lens starts to
    # agree with the model's output on WikiText (lens top 1 = output top 1 on 7% or more of tokens,
    # results/control/{model}/stats.json).
    "gemma-270m": {"ours": (7, 15), "paper": (7, 15), "late": (11, 16), "all": (0, 16)},
    "gemma-270m-it": {"ours": (7, 15), "paper": (7, 15), "late": (11, 16), "all": (0, 16)},
    "gemma-1b": {"ours": (10, 23), "paper": (10, 23), "all": (0, 24)},
    "gemma-1b-it": {"ours": (10, 23), "paper": (10, 23), "all": (0, 24)},
}
MODEL_LABEL = {"gpt2": "GPT-2 small", "qwen": "Qwen3.5-0.8B", "gemma-270m": "Gemma-3-270m",
               "gemma-270m-it": "Gemma-3-270m-it", "gemma-1b": "Gemma-3-1b", "gemma-1b-it": "Gemma-3-1b-it"}
MAIN_FRAME = {"gpt2": "human", "qwen": "paper", "gemma-270m": "human", "gemma-270m-it": "paper",
              "gemma-1b": "human", "gemma-1b-it": "paper"}


class Grid:
    def __init__(self, model, frame=None):
        frame = frame or MAIN_FRAME[model]
        d = RESULTS / "control" / model
        meta = json.load(open(d / f"modulation_grid_{frame}.json"))
        ranks = np.load(d / f"modulation_grid_{frame}.npz")["ranks"].astype(np.int64)
        ranks[ranks == 0] = NO_RANK                                   # padding past the sentence's end
        self.model, self.frame, self.meta = model, frame, meta
        self.trials, self.ranks = meta["trials"], ranks               # ranks: [trial, layer, position]
        self.layers = meta["layers"]                                  # lens layers, then "output"
        self.family = np.array([t["family"] for t in self.trials])
        self.cond = np.array([t["cond"] for t in self.trials])
        self.phrasing = np.array([t["phrasing"] or "" for t in self.trials])

    def best(self, band="ours", skip_first=False, held=None, skip_last=False):
        """[trial] best rank of a tracked token over the band's layers and every position.
        `band`: a key of BANDS, a (first, last) pair of layers, or "output".
        `skip_first`: leave out the sentence's first token.  `held`: count only positions where no
        tracked token is in the model's own top `held` next tokens (held in mind, not about to be
        said).  `skip_last`: leave out the sentence's last token, where a small model recalls what
        followed the sentence in the prompt (post/DM_LINDSEY.md)."""
        if band == "output":
            return self.ranks[:, -1, :].min(axis=1)
        # a variant run (e.g. `gemma-270m-it_lens-gemma-270m`) uses its model's bands
        lo, hi = BANDS[self.model.split("_")[0]][band] if isinstance(band, str) else band
        r = self.ranks[:, lo:hi + 1, :]
        if held:
            r = np.where((self.ranks[:, -1, :] <= held)[:, None, :], NO_RANK, r)
        if skip_last:
            n = np.array([t["n_pos"] for t in self.trials])
            r = np.where((np.arange(r.shape[2]) == (n - 1)[:, None])[:, None, :], NO_RANK, r)
        if skip_first:
            r = r[:, :, 1:]
        return r.min(axis=(1, 2))

    def rates(self, family, band="ours", k=1, **kw):
        """{cond: (mean over phrasings, {phrasing: hit rate})}, and the no-instruction rate."""
        hit = self.best(band, **kw) <= k
        out = {}
        for c in ["baseline"] + CONDS:
            m = (self.family == family) & (self.cond == c)
            per = {p: float(hit[m & (self.phrasing == p)].mean()) for p in dict.fromkeys(self.phrasing[m])}
            out[c] = (float(np.mean(list(per.values()))), per)
        return out


def claude():
    """The paper's Fig. 65 data: {family: [per model {cond: (mean, {phrasing: rate})}]}, model names."""
    d = json.load(open(ROOT / "ref/paper-data/modulation-prompts.json"))
    keys = [g["key"] for g in d["groups"]]
    out = {f: [None] * len(d["models"]) for f in FAMILIES}
    for p in d["panels"]:
        if p["fi"] < len(FAMILIES):
            r = {k: (g["mean"], {dot["t"]: dot["y"] for dot in g["dots"]}) for k, g in zip(keys, p["groups"])}
            r["baseline"] = (p["none"], {"": p["none"]})
            out[FAMILIES[p["fi"]]][p["mi"]] = r
    return out, d["models"]
