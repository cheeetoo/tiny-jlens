"""Capability survey of a tiny chat model, to establish the task repertoire
for the C1-C5 experiments (BRIEF §3, capability filter).

Everything here is *behavioral* (no lens involved), so it can be run before
lens fitting finishes on another model. Results go to runs/<tag>-capability.json.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import transformers

REF = os.path.join(os.path.dirname(__file__), "..", "ref", "jacobian-lens", "data")


def load(name):
    with open(os.path.join(REF, "experiments", name)) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--tag", default="smollm2-135m-it")
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    hf.eval()

    def gen(text_or_msgs, max_new=16, prefill=""):
        if isinstance(text_or_msgs, str):
            full = text_or_msgs + prefill
        else:
            full = tok.apply_chat_template(text_or_msgs, tokenize=False, add_generation_prompt=True) + prefill
        ids = tok(full, return_tensors="pt")["input_ids"].cuda()
        with torch.no_grad():
            out = hf.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def chat(user, **kw):
        return gen([{"role": "user", "content": user}], **kw)

    results = {}

    # --- template inspection ---
    rendered = tok.apply_chat_template([{"role": "user", "content": "XYZ"}], tokenize=False, add_generation_prompt=True)
    results["chat_template_example"] = rendered

    # --- A: category report ---
    vr = load("verbal-report.json")["candidates"]
    cat_results = {}
    for cat, members in vr.items():
        out = chat(f"Think of a {cat}. Answer with one word.", prefill=f'The {cat} I am thinking of is "', max_new=6)
        word = out.split('"')[0].strip().strip(".").lower()
        valid = word in [m.lower() for m in members]
        cat_results[cat] = {"raw": out[:40], "word": word, "in_candidate_list": valid}
    results["A_category_report"] = cat_results

    # --- B: flexible-generalization function cells (raw completion) ---
    fg = load("flexible-generalization.json")["categories"]
    fg_results = {}
    for c in fg:
        for fn in c["funcs"]:
            cell = []
            for arg in c["args"]:
                prompt = "Fact: " + fn["template"].format(arg=arg)
                out = gen(prompt, max_new=4)
                ans = fn["answers"][arg]
                ok = out.strip().lower().startswith(ans.lower()[: max(3, len(ans))]) or ans.lower() in out.lower()
                cell.append({"arg": arg, "out": out[:24], "want": ans, "ok": ok})
            fg_results[f"{c['name']}/{fn['name']}"] = {
                "n_ok": sum(x["ok"] for x in cell), "cells": cell}
    results["B_flexgen"] = fg_results

    # --- C: probe-swap two-hop (raw) ---
    ps = load("probe-swap.json")["items"]
    ps_results = []
    for it in ps:
        out = gen(it["prompt"], max_new=4)
        ok = out.strip().lower().startswith(it["answer"].lower()[:4]) or it["answer"].lower() in out.lower()
        ps_results.append({"name": it["name"], "out": out[:24], "want": it["answer"], "ok": ok})
    results["C_twohop"] = {"n_ok": sum(x["ok"] for x in ps_results), "n": len(ps_results), "items": ps_results}

    # --- D: copying, with and without a side instruction ---
    dm = load("directed-modulation.json")
    carriers = dm["carrier_sentences"][:8]
    copy_plain, copy_inst = [], []
    for s in carriers:
        o1 = chat(f'Copy this sentence exactly: "{s}"', prefill=' "', max_new=24)
        copy_plain.append({"carrier": s, "out": o1[:70], "exact": o1.strip().startswith(s)})
        o2 = chat(f'Think about citrus fruits while you write. Copy this sentence exactly: "{s}"', prefill=' "', max_new=24)
        copy_inst.append({"carrier": s, "out": o2[:70], "exact": o2.strip().startswith(s)})
    results["D_copy"] = {
        "plain_exact": sum(x["exact"] for x in copy_plain), "inst_exact": sum(x["exact"] for x in copy_inst),
        "n": len(carriers), "plain": copy_plain, "inst": copy_inst}

    # --- E: language tasks ---
    sl = load("selectivity-language.json")
    lang_results = {}
    for p in sl["passages"]:
        name_out = chat(f"Here is a passage of text:\n\n{p['text']}\n\nWhat language is this passage written in? Answer with one word.",
                        prefill=' The language is', max_new=4)
        cont_out = chat(f"Here is a passage of text:\n\n{p['text']}\n\nContinue it by writing the next sentence. Output only the continuation.", max_new=20)
        author_out = chat(sl["task"]["explicit_q"].format(text=p["text"]), prefill=" The author is", max_new=6)
        hello = {"French": "bonjour", "German": "hallo", "Spanish": "hola", "Italian": "ciao"}[p["category"]]
        hello_out = chat(f"Here is a passage of text:\n\n{p['text']}\n\nHow do people say \"hello\" in the language of this passage? Answer with one word.",
                         prefill=' They say "', max_new=4)
        lang_results[p["key"]] = {
            "want": p["category"], "name": name_out[:20],
            "name_ok": p["category"].lower() in name_out.lower(),
            "cont": cont_out[:60],
            "author": author_out[:24],
            "hello_want": hello, "hello": hello_out[:16], "hello_ok": hello.lower() in hello_out.lower()}
    results["E_language"] = lang_results

    # --- F: tier-1 math ---
    math_results = []
    for m in [x for x in dm["math_problems"] if x["tier"] == 1][:10]:
        out = chat(f"What is {m['expr']}? Answer with just the number.", prefill=" The answer is", max_new=4)
        math_results.append({"expr": m["expr"], "out": out[:12], "want": m["answer"], "ok": m["answer"] in out})
    results["F_math1"] = {"n_ok": sum(x["ok"] for x in math_results), "n": len(math_results), "items": math_results}

    # --- G: shallow battery candidates ---
    sentiments = [
        ("I loved this movie, it was wonderful and heartwarming.", "positive"),
        ("This was a terrible waste of time, I hated every minute.", "negative"),
        ("An absolute delight from start to finish.", "positive"),
        ("Boring, predictable, and badly acted.", "negative"),
        ("The best meal I have had in years.", "positive"),
        ("The service was awful and the food was cold.", "negative"),
    ]
    sent_results = []
    for text, want in sentiments:
        out = chat(f'Review: "{text}"\nIs this review positive or negative? Answer with one word.',
                   prefill=" The review is", max_new=3)
        sent_results.append({"want": want, "out": out[:16], "ok": want in out.lower()})
    results["G_sentiment"] = {"n_ok": sum(x["ok"] for x in sent_results), "n": len(sent_results), "items": sent_results}

    odd_items = [
        ("apple banana cherry hammer", "hammer"),
        ("dog cat horse chair", "chair"),
        ("red green blue seven", "seven"),
        ("car truck bus apple", "apple"),
    ]
    odd_results = []
    for words, want in odd_items:
        out = chat(f"Which word does not belong: {words}? Answer with one word.", prefill=' The word that does not belong is "', max_new=4)
        odd_results.append({"want": want, "out": out[:16], "ok": want in out.lower()})
    results["G_oddone"] = {"n_ok": sum(x["ok"] for x in odd_results), "n": len(odd_results), "items": odd_results}

    out_path = f"runs/{args.tag}-capability.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)

    # summary
    print("=== SUMMARY ===")
    print("A category report in-list:", sum(v["in_candidate_list"] for v in cat_results.values()), "/", len(cat_results))
    for k, v in results["B_flexgen"].items():
        print(f"B {k}: {v['n_ok']}/4")
    print("C two-hop:", results["C_twohop"]["n_ok"], "/", results["C_twohop"]["n"])
    print("D copy plain:", results["D_copy"]["plain_exact"], "inst:", results["D_copy"]["inst_exact"], "/", results["D_copy"]["n"])
    print("E language name_ok:", sum(v["name_ok"] for v in lang_results.values()), "/", len(lang_results),
          "| hello_ok:", sum(v["hello_ok"] for v in lang_results.values()))
    print("F math1:", results["F_math1"]["n_ok"], "/", results["F_math1"]["n"])
    print("G sentiment:", results["G_sentiment"]["n_ok"], "/", results["G_sentiment"]["n"],
          "| oddone:", results["G_oddone"]["n_ok"], "/", results["G_oddone"]["n"])
    print("saved", out_path)


if __name__ == "__main__":
    main()
