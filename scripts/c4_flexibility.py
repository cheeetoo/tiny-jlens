"""C4 — Flexible generalization / broadcast (paper §3.4, BRIEF C4).

The same argument swap (e.g. France->China), applied identically across
prompts that each apply a different function, should redirect each function's
answer. Capability filter first: keep (category, function) cells where the
model answers >=3/4 args correctly; swap only between args whose cells are
correct. Workspace loading (cos sim of residual with source lens vector)
should predict swap success.

Usage: python scripts/c4_flexibility.py --lens runs/smollm2-135m-it/lens.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import transformers

import jlens

from tinyjlens.lensops import LensKit
from tinyjlens.prompts import build_raw, variant_token_ids
from tinyjlens.interventions import clamped_swap_edits, logits_with_edits

REF = os.path.join(os.path.dirname(__file__), "..", "ref", "jacobian-lens", "data")

CUSTOM_CATEGORIES = [
    {"name": "months2", "args": ["February", "April", "July", "October"], "funcs": [
        {"name": "season", "template": "In the northern hemisphere, the season during {arg} is",
         "answers": {"February": "winter", "April": "spring", "July": "summer", "October": "fall|autumn"}},
        {"name": "before", "template": "The month that comes right after {arg} is",
         "answers": {"February": "March", "April": "May", "July": "August", "October": "November"}},
        {"name": "holiday", "template": "A famous holiday celebrated in {arg} is",
         "answers": {"February": "Valentine", "April": "Easter", "July": "Independence", "October": "Halloween"}},
        {"name": "half", "template": "Of the two halves of the year, {arg} falls in the",
         "answers": {"February": "first", "April": "first", "July": "second", "October": "second"}},
    ]},
    {"name": "animals2", "args": ["lion", "eagle", "spider", "fish"], "funcs": [
        {"name": "legs", "template": "The number of legs on a {arg} is",
         "answers": {"lion": "four|4", "eagle": "two|2", "spider": "eight|8", "fish": "zero|0|no"}},
        {"name": "covered", "template": "The body of a {arg} is covered in",
         "answers": {"lion": "fur", "eagle": "feathers", "spider": "hair", "fish": "scales"}},
        {"name": "class", "template": "In biology, a {arg} is classified as a",
         "answers": {"lion": "mammal", "eagle": "bird", "spider": "arachnid|spider", "fish": "fish"}},
        {"name": "home", "template": "The place where a {arg} lives is called its",
         "answers": {"lion": "den|pride", "eagle": "nest|aer", "spider": "web", "fish": "water|tank|aquarium"}},
    ]},
    {"name": "numbers2", "args": ["three", "four", "seven", "eight"], "funcs": [
        {"name": "first_letter", "template": "The first letter of the word {arg} is",
         "answers": {"three": "t", "four": "f", "seven": "s", "eight": "e"}},
        {"name": "evenodd", "template": "The number {arg} is either even or odd; it is",
         "answers": {"three": "odd", "four": "even", "seven": "odd", "eight": "even"}},
        {"name": "bigger5", "template": "Is the number {arg} bigger than five, yes or no? The answer is",
         "answers": {"three": "no", "four": "no", "seven": "yes", "eight": "yes"}},
        {"name": "double", "template": "Two times {arg} equals",
         "answers": {"three": "six", "four": "eight", "seven": "fourteen", "eight": "sixteen"}},
    ]},
]

EXTRA_FUNCS = {
    "months": [
        {"name": "days", "template": "The number of days in the month of {arg} is usually",
         "answers": {"February": "28", "April": "30", "July": "31", "October": "31"}},
        {"name": "half", "template": "Of the two halves of the year, {arg} falls in the",
         "answers": {"February": "first", "April": "first", "July": "second", "October": "second"}},
    ],
    "animals": [
        {"name": "legs2", "template": "The number of legs on a {arg} is",
         "answers": {"lion": "four", "eagle": "two", "shark": "zero", "spider": "eight"}},
        {"name": "baby", "template": "A baby {arg} is called a",
         "answers": {"lion": "cub", "eagle": "chick", "shark": "pup", "spider": "spider"}},
    ],
    "numbers": [
        {"name": "evenodd", "template": "The number {arg} is either even or odd; it is",
         "answers": {"three": "odd", "five": "odd", "seven": "odd", "nine": "odd"}},
        {"name": "bigger", "template": "Of the numbers four and {arg}, the bigger one is",
         "answers": {"three": "four", "five": "five", "seven": "seven", "nine": "nine"}},
    ],
}


def first_token_id(tok, word):
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    for i in ids:
        if tok.decode([i]).strip():
            return i
    return ids[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--lens", default="runs/smollm2-135m-it/lens.pt")
    ap.add_argument("--band", default=None)
    ap.add_argument("--alphas", default="1,2")
    ap.add_argument("--chat", action="store_true", help="render cells as chat questions with prefill anchors")
    ap.add_argument("--union", action="store_true", help="per cell, accept whichever format (raw/chat) passes")
    ap.add_argument("--out", default="runs/c4_flexibility.json")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(args.lens)
    kit = LensKit.build(model, lens)
    n_layers = model.n_layers
    if args.band:
        lo, hi = map(int, args.band.split(":"))
    else:
        lo, hi = int(n_layers * 0.4), int(n_layers * 0.87)
    band = [l for l in range(lo, hi + 1) if l in lens.source_layers]

    with open(os.path.join(REF, "experiments", "flexible-generalization.json")) as f:
        cats = json.load(f)["categories"]
    for c in cats:
        c["funcs"] = c["funcs"] + EXTRA_FUNCS.get(c["name"], [])
    cats = cats + CUSTOM_CATEGORIES

    results = {"band": [band[0], band[-1]], "cells": {}, "prompt_ext": {}, "trials": []}

    WORDNUM = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
               "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}

    def accept_set(answer_spec: str) -> set[int]:
        ids = []
        for alt in answer_spec.split("|"):
            ids += variant_token_ids(tok, alt)
            if alt in WORDNUM:
                ids += variant_token_ids(tok, WORDNUM[alt])
            if len(alt) == 1:
                ids += [t for t in variant_token_ids(tok, alt.upper())]
        return set(ids)

    ARTICLES = {" the", " a", " an", " called", " its", " in"}

    def graded_prompt(prompt: str, accept: set[int]) -> tuple[str, bool]:
        """Follow up to 2 article tokens; return (extended_prompt, correct)."""
        for _ in range(3):
            bp = build_raw(tok, prompt)
            logits = logits_with_edits(kit, bp.input_ids, [])[-1]
            top = int(logits.argmax())
            if top in accept:
                return prompt, True
            if tok.decode([top]) in ARTICLES:
                prompt = prompt + tok.decode([top])
                continue
            return prompt, False
        return prompt, False

    # ---------- capability filter ----------
    for c in cats:
        for fn in c["funcs"]:
            ok_args = []
            for arg in c["args"]:
                if args.union:
                    q = fn["template"].format(arg=arg)
                    chatp = tok.apply_chat_template(
                        [{"role": "user", "content": f"{q} ... Complete the sentence with one word."}],
                        tokenize=False, add_generation_prompt=True) + " " + q
                    rawp = "Fact: " + q
                    acc = accept_set(fn["answers"][arg])
                    for cand in (rawp, chatp):
                        ext, ok = graded_prompt(cand, acc)
                        if ok:
                            ok_args.append(arg)
                            results["prompt_ext"][f"{c['name']}/{fn['name']}/{arg}"] = ext
                            break
                    continue
                if args.chat:
                    q = fn["template"].format(arg=arg)
                    prompt = tok.apply_chat_template(
                        [{"role": "user", "content": f"{q} ... Complete the sentence with one word."}],
                        tokenize=False, add_generation_prompt=True) + " " + q
                else:
                    prompt = "Fact: " + fn["template"].format(arg=arg)
                ext, ok = graded_prompt(prompt, accept_set(fn["answers"][arg]))
                if ok:
                    ok_args.append(arg)
                    results["prompt_ext"][f"{c['name']}/{fn['name']}/{arg}"] = ext
            results["cells"][f"{c['name']}/{fn['name']}"] = ok_args
            print(f"{c['name']}/{fn['name']}: {len(ok_args)}/4 {ok_args}")

    # ---------- swap trials ----------
    def paired_forms(a, b):
        pairs = []
        for pre in (" ", ""):
            ia = tok(pre + a, add_special_tokens=False)["input_ids"]
            ib = tok(pre + b, add_special_tokens=False)["input_ids"]
            if len(ia) == 1 and len(ib) == 1:
                pairs.append((ia[0], ib[0]))
        return pairs

    for c in cats:
        kept_funcs = [fn for fn in c["funcs"]
                      if len(results["cells"][f"{c['name']}/{fn['name']}"]) >= 3]
        if len(kept_funcs) < 2:
            continue
        for fn in kept_funcs:
            ok_args = results["cells"][f"{c['name']}/{fn['name']}"]
            for src in ok_args:
                for tgt in ok_args:
                    if src == tgt or fn["answers"][src].split("|")[0] == fn["answers"][tgt].split("|")[0]:
                        continue
                    pairs = paired_forms(src, tgt)
                    if not pairs:
                        continue
                    prompt = results["prompt_ext"][f"{c['name']}/{fn['name']}/{src}"]
                    # (chat mode: prompt_ext already contains the rendered form)
                    bp = build_raw(tok, prompt)
                    clean = logits_with_edits(kit, bp.input_ids, [])[-1]
                    want_set = accept_set(fn["answers"][tgt])
                    # workspace loading: cos(resid, src lens vec), argument+final pos
                    arg_span = bp.find_span(src)
                    resid = kit.residuals(bp.input_ids, band)
                    loads = []
                    for l in band:
                        v = kit.jlens_vector(l, pairs[0][0])
                        v = v / v.norm()
                        for p in list(range(*arg_span)) + [bp.n_tokens - 1]:
                            h = resid[l][p]
                            loads.append(float((h / h.norm()) @ v))
                    def best_rank(lg):
                        return min(int((lg > lg[w]).sum()) for w in want_set)
                    row = {"cat": c["name"], "func": fn["name"], "src": src, "tgt": tgt,
                           "want": fn["answers"][tgt],
                           "pre_rank": best_rank(clean),
                           "loading": sum(loads) / len(loads)}
                    for a in alphas:
                        edits = clamped_swap_edits(kit, bp.input_ids, band,
                                                   [p[0] for p in pairs], [p[1] for p in pairs], a)
                        lg = logits_with_edits(kit, bp.input_ids, edits)[-1]
                        row[f"post_rank_a{a:g}"] = best_rank(lg)
                    results["trials"].append(row)

    n = len(results["trials"])
    for a in alphas:
        k = sum(r[f"post_rank_a{a:g}"] == 0 for r in results["trials"])
        print(f"alpha={a:g}: {k}/{n} swaps hit top-1")
        results[f"top1_a{a:g}"] = k
    # loading-success relationship
    if n:
        import statistics
        succ = [r for r in results["trials"] if r[f"post_rank_a{alphas[0]:g}"] == 0]
        fail = [r for r in results["trials"] if r[f"post_rank_a{alphas[0]:g}"] != 0]
        if succ and fail:
            print("mean loading success/fail:",
                  round(statistics.mean(r["loading"] for r in succ), 4),
                  round(statistics.mean(r["loading"] for r in fail), 4))
    by_cat = {}
    for r in results["trials"]:
        s = by_cat.setdefault(r["cat"], [0, 0])
        s[1] += 1
        s[0] += r[f"post_rank_a{alphas[-1]:g}"] == 0
    print("by category (last alpha):", by_cat)
    results["by_cat"] = by_cat

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
