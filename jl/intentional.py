"""Lindsey (2025), "Emergent Introspective Awareness in Large Language Models", section "Intentional
Control of Internal States": the experiment as that paper runs it, on GPT-2 small and on Gemma 3 (270m
and 1b, base and instruction-tuned), read two ways from the same forward passes.

The task is the paper's (ref/introspection/intentional-control.json).  The model is asked to write a
sentence and told to think, or not to think, about a word while it does --
    Write "{sentence}" Think about {word} while you write the sentence. Don't write anything else.
-- and the reply is the sentence, forced.  All of the paper's prompts: the directive pair (think / don't
think) and its three incentive pairs (rewarded / punished, happy / sad, charity / terrorist
organizations), from its appendix, and its four control pairs, which put the word in an affirmative or
a negative phrase that isn't an instruction ("There is [no] such thing as {word}. ...").  Its 50 words
(lowercase) and 50 sentences, every pair (the paper samples pairs at random).  One prompt the paper
doesn't have is added, with no word in it (`no_word`, "Write "{sentence}". Don't write anything else.");
it is the floor for the J-lens read and the centering mean for the cosine.

Read 1, the paper's: concept vectors.  A word's concept vector at layer L is the residual at the final
":" of "Human: Tell me about {word}\\n\\nAssistant:", minus its mean over the paper's 100 baseline
words.  The measure is the cosine of the residual with the word's concept vector at each token of the
reply, averaged over the reply's tokens.  The baseline is the same cosine with each of the paper's 100
control words' vectors.  Kept raw (the paper's) and centered: the residual minus its mean over the reply
tokens of the 50 no-word prompts, our earlier replication's fix for Gemma's massive activations
(github.com/uchicago-xlab/intentional-control).

Read 2, the workspace paper's: the J-lens.  At every lens layer and every token of the reply, the best
lens rank of the word's single-token forms (with or without a leading space, lowercase or capitalized),
and its rank in the model's own output.  Words with no single-token form are left out of this read only.
Scored afterwards (post/figures/dm_lindsey.py) as the workspace paper scores directed modulation: a hit
is a form at rank 1 at any band layer and any token of the reply (its Fig. 10), or in the top 5 (its
Fig. 46, which runs this very task with a named concept).

Frames.  `human`: the paper's transcript as plain text, "\\n\\nHuman: {prompt}\\n\\nAssistant: {sentence}",
for every model (it is GPT-2's only frame).  `chat`: Gemma's turn format (jl.control.Gemma.chat), for the
four Gemma models, so that the base and the instruction-tuned model see the same tokens in each frame.
The concept vectors are read in the trial's frame.  Layer L is the output of block L, as everywhere in jl.

Writes results/intentional/{model}/lindsey_{frame}.json (the trials, the words and their tracked forms,
the control-word baseline per condition) and lindsey_{frame}.npz:
  ranks       [trial, layer, token] uint16: best rank of a tracked form at each lens layer, then (last
              row) in the model's output; 0 past the end of the reply or for an untracked word
  cos, cos_c  [trial, layer]: cosine with the word's concept vector, raw and centered
  cos_nl, cos_c_nl  the same averaged over every token of the reply but its last (where a small model
              that has copied the sentence may be about to add the word)
  ctrl, ctrl_c  [trial, layer]: the same, averaged over the 100 control words
  base, base_c  [condition, layer, control word]: the control-word cosines averaged over trials
  first_ok, rest_ok, n_pos  [trial]: is the reply's first token the model's own top prediction; how many
              of the rest are; the reply's length in tokens
Run:  python -m jl.intentional --model gemma-1b-it          (frames: GPT-2 `human`, Gemma both)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import jl
from jl import control as C

IC = json.load(open(jl.model.ROOT / "ref/introspection/intentional-control.json"))
WORDS, SENTENCES, TEMPLATES = IC["concept_words"], IC["sentences"], IC["templates"]
CONDS = ["none"] + [t["name"] for t in TEMPLATES]


def text(A, frame, user, reply=""):
    """The prompt, with the reply (if any) after it."""
    if frame == "human":
        return f"\n\nHuman: {user}\n\nAssistant:" + (f" {reply}" if reply else "")
    return A.chat([("user", user)], prefill=reply)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def final_residuals(A, frame, words, layers):
    """[word, layer, d]: the residual at the last token of the concept-vector prompt for each word."""
    ids = [A.ids(text(A, frame, IC["concept_vector_prompt"].format(word=w.lower()))) for w in words]
    out = [None] * len(ids)
    for idx, res in C.batched_residuals(A, ids, layers):
        for b, i in enumerate(idx):
            out[i] = torch.stack([res[L][b, ids[i].shape[1] - 1] for L in layers])
    return torch.stack(out)


@torch.no_grad()
def run(A, frame, size=32):
    _, unembed, lens_layers, last = C.lens_parts(A)
    layers = list(range(A.n_layers))                       # every block output; the lens covers all but the last
    assert lens_layers + [last] == layers
    t0 = time.time()

    # concept vectors (the paper's): minus the mean over the 100 baseline words, then unit length
    mean = final_residuals(A, frame, IC["baseline_words"], layers).mean(0)
    cv = unit(final_residuals(A, frame, WORDS, layers) - mean)                    # [W, N, d]
    ctrl = unit(final_residuals(A, frame, IC["control_words"], layers) - mean)    # [C, N, d]
    tracked = [C.forms(A, w) for w in WORDS]

    # the trials: the no-word prompt once per sentence (scored against every word), then every template
    jobs = []                                                                    # (cond, word or -1, sentence)
    ids, pos = [], []
    for cond, users in [("none", [(-1, IC["no_word"])])] + [
            (t["name"], [(wi, t["user"].replace("{word}", w.lower())) for wi, w in enumerate(WORDS)]) for t in TEMPLATES]:
        for wi, user in users:
            for si, s in enumerate(SENTENCES):
                tx = text(A, frame, user.format(sentence=s), s)
                jobs.append((cond, wi, si))
                ids.append(A.ids(tx))
                pos.append(A.span(tx, s, start_at=tx.rindex(s)))

    # the centering mean: every reply token of the 50 no-word prompts (our earlier replication's)
    tot, n = 0, 0
    for idx, res in C.batched_residuals(A, ids[:len(SENTENCES)], layers, size):
        for b, i in enumerate(idx):
            tot = tot + torch.stack([res[L][b, pos[i]].sum(0) for L in layers])
            n += len(pos[i])
    mu = tot / n                                                                 # [N, d]

    # one row per (condition, word, sentence); the no-word forwards fill a row for every word
    rows = [(c, w, s) for c in CONDS for w in range(len(WORDS)) for s in range(len(SENTENCES))]
    row = {r: k for k, r in enumerate(rows)}
    P = max(len(p) for p in pos)
    N, R = len(layers), len(rows)
    ranks = np.zeros((R, len(lens_layers) + 1, P), np.uint16)
    cos, cos_c, cos_nl, cos_c_nl, ct, ct_c = (np.zeros((R, N), np.float32) for _ in range(6))
    first_ok, rest_ok, n_pos = np.zeros(R, bool), np.zeros(R, np.int16), np.zeros(R, np.int16)
    base, base_c = (np.zeros((len(CONDS), N, len(ctrl))) for _ in range(2))
    ids_cpu = [x[0].cpu().numpy() for x in ids]
    n_none = len(SENTENCES)
    # the no-word prompts first, on their own (each is scored against every word), then the rest
    for part in (range(n_none), range(n_none, len(jobs))):
        part = list(part)
        for nb, (idx, res) in enumerate(C.batched_residuals(A, [ids[i] for i in part], layers, size)):
            idx = [part[k] for k in idx]
            P_ = [pos[i] for i in idx]
            starts = np.cumsum([0] + [len(p) for p in P_])
            lens = C.to_device(np.diff(starts).astype(np.float32), cv.device)
            item = C.to_device(np.repeat(np.arange(len(idx)), np.diff(starts)), cv.device)
            H = C.rows_of(res, layers, P_)                                       # [N, M, d]
            Hu, Hcu = unit(H), unit(H - mu[:, None])

            notlast = np.ones(starts[-1], np.float32)
            notlast[starts[1:] - 1] = 0
            notlast = C.to_device(notlast, cv.device)

            def per_item(x, w=None):                                             # [N, M, ...] -> [N, B, ...]
                z = torch.zeros((x.shape[0], len(idx)) + x.shape[2:], device=x.device)
                n = lens if w is None else lens - 1
                x = x if w is None else x * w.view((1, -1) + (1,) * (x.dim() - 2))
                return z.index_add_(1, item, x) / n.view((1, -1) + (1,) * (x.dim() - 2))
            # read 1: cosines with every word's concept vector and every control word's, averaged over
            # each reply's tokens (and, `_nl`, over all but its last token)
            cw_raw, cw_cen = torch.einsum("nmd,wnd->nmw", Hu, cv), torch.einsum("nmd,wnd->nmw", Hcu, cv)
            c_raw, c_cen = per_item(cw_raw), per_item(cw_cen)
            c_raw_nl, c_cen_nl = per_item(cw_raw, notlast), per_item(cw_cen, notlast)
            k_raw, k_cen = per_item(torch.einsum("nmd,cnd->nmc", Hu, ctrl)), per_item(torch.einsum("nmd,cnd->nmc", Hcu, ctrl))
            # the copy check: is each token of the reply the model's own top prediction?
            out = unembed(H[-1]).float().argmax(-1)                              # [M] the next token after each
            prev = unembed(res[last][C.to_device(np.arange(len(idx)), cv.device),
                                     C.to_device([p[0] - 1 for p in P_], cv.device)]).float().argmax(-1)
            # read 2: the J-lens at every lens layer, and the model's output, at every token of each reply
            if part[0] == 0:                                                     # the no-word prompts: every word
                rk = {}
                for s, lg in C.lens_logit_chunks(A, H):
                    for w, tr in enumerate(tracked):
                        if tr:
                            top = lg[:, tr].max(dim=1, keepdim=True).values
                            rk.setdefault(w, []).append((lg > top).sum(dim=1) + 1)
                rk = {w: torch.cat(v).view(len(lens_layers) + 1, -1).clamp(max=C.RANK_CAP).cpu().numpy()
                      for w, v in rk.items()}
            else:
                keep = [k for k, i in enumerate(idx) if tracked[jobs[i][1]]]
                offset = dict(zip(keep, np.cumsum([0] + [len(P_[k]) for k in keep])))
                if keep:
                    rows_k = C.to_device(np.concatenate([np.arange(starts[k], starts[k + 1]) for k in keep]), cv.device)
                    rk_k = C.lens_ranks(A, H[:, rows_k], [tracked[jobs[idx[k]][1]] for k in keep for _ in P_[k]])
                    rk_k = rk_k.clamp(max=C.RANK_CAP).cpu().numpy()
            c_raw, c_cen, c_raw_nl, c_cen_nl, k_raw, k_cen = (x.cpu().numpy() for x in (c_raw, c_cen, c_raw_nl, c_cen_nl,
                                                                                         k_raw, k_cen))
            out, prev = out.cpu().numpy(), prev.cpu().numpy()
            toks = np.concatenate([ids_cpu[i][p] for i, p in zip(idx, P_)])     # the reply's tokens, in row order
            same = out[:-1] == toks[1:]                                          # row m predicts row m+1's token
            first = prev == toks[starts[:-1]]
            for k, i in enumerate(idx):
                cond, wi, si = jobs[i]
                ci = CONDS.index(cond)
                ws = range(len(WORDS)) if wi < 0 else [wi]
                base[ci] += k_raw[:, k] * len(ws)
                base_c[ci] += k_cen[:, k] * len(ws)
                a, e = starts[k], starts[k + 1]
                for w in ws:
                    r = row[(cond, w, si)]
                    cos[r], cos_c[r] = c_raw[:, k, w], c_cen[:, k, w]
                    cos_nl[r], cos_c_nl[r] = c_raw_nl[:, k, w], c_cen_nl[:, k, w]
                    ct[r], ct_c[r] = k_raw[:, k].mean(-1), k_cen[:, k].mean(-1)
                    first_ok[r], rest_ok[r], n_pos[r] = first[k], same[a:e - 1].sum(), e - a
                    if not tracked[w]:
                        continue
                    if wi < 0:
                        ranks[r, :, :e - a] = rk[w][:, a:e]
                    else:
                        ranks[r, :, :e - a] = rk_k[:, offset[k]:offset[k] + e - a]
            if (nb + 1) * size % 2000 < size:
                print(f"  {frame}: {(nb + 1) * size}/{len(part)} prompts ({time.time() - t0:.0f}s)", flush=True)
    per_cond = len(WORDS) * len(SENTENCES)
    d = jl.results_dir(f"intentional/{A.name}")
    np.savez_compressed(d / f"lindsey_{frame}.npz", ranks=ranks, cos=cos, cos_c=cos_c, cos_nl=cos_nl, cos_c_nl=cos_c_nl,
                        ctrl=ct, ctrl_c=ct_c,
                        base=(base / per_cond).astype(np.float32), base_c=(base_c / per_cond).astype(np.float32),
                        first_ok=first_ok, rest_ok=rest_ok, n_pos=n_pos)
    ex = text(A, frame, TEMPLATES[0]["user"].format(sentence=SENTENCES[0], word=WORDS[0].lower()), SENTENCES[0])
    json.dump(dict(
        frame=frame, band=A.band, layers=lens_layers + ["output"], conditions=CONDS, templates=TEMPLATES,
        words=WORDS, sentences=SENTENCES, control_words=IC["control_words"],
        tracked={w: [A.dec(t) for t in tr] for w, tr in zip(WORDS, tracked)},
        trials=[dict(cond=c, word=w, sentence=s) for c, w, s in rows],
        example=ex, example_tokens=[A.dec(int(t)) for t in A.ids(ex)[0]],
        example_positions=A.span(ex, SENTENCES[0], start_at=ex.rindex(SENTENCES[0])),
        concept_vector_example=text(A, frame, IC["concept_vector_prompt"].format(word=WORDS[0].lower()))),
        open(d / f"lindsey_{frame}.json", "w"), indent=1)
    print(f"== lindsey {A.name} frame={frame}: {len(jobs)} prompts, {R} rows ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gpt2", *C.GEMMA])
    ap.add_argument("--frames", default=None, help="comma-separated: human, chat (default: GPT-2 human, Gemma both)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    A = C.load(args.model)
    print(f"model {getattr(A, 'model_id', 'gpt2')}  band {A.band}  device {A.device}", flush=True)
    for f in args.frames.split(",") if args.frames else (["human"] if args.model == "gpt2" else ["chat", "human"]):
        run(A, f, args.batch)
