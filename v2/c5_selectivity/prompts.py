"""Prompt material for criterion 5 (selectivity): the paper's §3.5.

Selectivity says the J-space is required for *flexible* cognition (report, multi-step inference)
but not for *automatic* processing (parsing, copying, one-step recall).  We test this three ways.

S2a  J-space ablation battery (§3.5.2, Fig 22/24).  We ablate the top-k J-space directions and
     watch a battery of tasks split by how much they depend on assembling inferred content:
       - two-hop reasoning  (flexible)  -- criterion 3's task, the paper's own multihop control
       - one-hop recall     (recall)    -- the same country facts, one hop
       - induction / copy   (automatic) -- copy a token seen earlier in the context
       - next-token match   (automatic) -- agreement with the clean model on wikitext prose
     The two-hop items and their country table ARE criterion 3's, imported verbatim so the
     positive control is exactly the task c3 validated (48/53 answered).

S2b  same-latent language dissociation (§3.5.1, Fig 20).  The paper's released passages
     (ref/.../selectivity-language.json): a passage whose language is evident but unnamed, under
     a deliberate task (name the language) vs an automatic task (continue the passage).  The same
     language-label swap redirects the deliberate report but not the continuation.

floor  line-length counting (§3.5.1, Fig 21) is a base-model capability floor -- see run.py.
"""
from __future__ import annotations

import json
import random

# The two-hop task is criterion 3's.  Its country table and relation families are copied here
# verbatim (a static world-knowledge table) so c5 is self-contained and its positive control is
# byte-identical to the multihop eval c3 validated.  Kept in sync by hand; see c3_reasoning/prompts.py.
COUNTRIES: dict[str, dict] = {
    "France":      dict(language="French",     capital="Paris",     continent="Europe",  language_unique=True),
    "Germany":     dict(language="German",     capital="Berlin",    continent="Europe",  language_unique=True),
    "Spain":       dict(language="Spanish",    capital="Madrid",    continent="Europe",  language_unique=True),
    "Italy":       dict(language="Italian",    capital="Rome",      continent="Europe",  language_unique=True),
    "Poland":      dict(language="Polish",     capital="Warsaw",    continent="Europe",  language_unique=True),
    "Russia":      dict(language="Russian",    capital="Moscow",    continent="Europe",  language_unique=True),
    "China":       dict(language="Chinese",    capital="Beijing",   continent="Asia",    language_unique=True),
    "Portugal":    dict(language="Portuguese", capital="Lisbon",    continent="Europe",  language_unique=True),
    "Greece":      dict(language="Greek",      capital="Athens",    continent="Europe",  language_unique=True),
    "India":       dict(language="Hindi",      capital="Delhi",     continent="Asia",    language_unique=True),
    "Turkey":      dict(language="Turkish",    capital="Ankara",    continent="Asia",    language_unique=True),
    "Sweden":      dict(language="Swedish",    capital="Stockholm", continent="Europe",  language_unique=True),
    "Norway":      dict(language="Norwegian",  capital="Oslo",      continent="Europe",  language_unique=True),
    "Finland":     dict(language="Finnish",    capital="Helsinki",  continent="Europe",  language_unique=True),
    "Hungary":     dict(language="Hungarian",  capital="Budapest",  continent="Europe",  language_unique=True),
    "Iran":        dict(language="Persian",    capital="Tehran",    continent="Asia",    language_unique=True),
    "Korea":       dict(language="Korean",     capital="Seoul",     continent="Asia",    language_unique=True),
    "Vietnam":     dict(language="Vietnamese", capital="Hanoi",     continent="Asia",    language_unique=True),
    "Thailand":    dict(language="Thai",       capital="Bangkok",   continent="Asia",    language_unique=True),
    "Netherlands": dict(language="Dutch",      capital="Amsterdam", continent="Europe",  language_unique=True),
    "Ukraine":     dict(language="Ukrainian",  capital="Kiev",      continent="Europe",  language_unique=True),
    "Denmark":     dict(language="Danish",     capital="Copenhagen", continent="Europe", language_unique=True),
    "Romania":     dict(language="Romanian",   capital="Bucharest", continent="Europe",  language_unique=True),
    "Bulgaria":    dict(language="Bulgarian",  capital="Sofia",     continent="Europe",  language_unique=True),
    "Serbia":      dict(language="Serbian",    capital="Belgrade",  continent="Europe",  language_unique=True),
    "Indonesia":   dict(language="Indonesian", capital="Jakarta",   continent="Asia",    language_unique=True),
    "Pakistan":    dict(language="Urdu",       capital="Islamabad", continent="Asia",    language_unique=True),
    "Iceland":     dict(language="Icelandic",  capital="Reykjavik", continent="Europe",  language_unique=True),
    "Mongolia":    dict(language="Mongolian",  capital="Ulaanbaatar", continent="Asia",  language_unique=True),
    "Kenya":       dict(language="Swahili",    capital="Nairobi",   continent="Africa",  language_unique=True),
    "Nigeria":     dict(language="Yoruba",     capital="Abuja",     continent="Africa",  language_unique=True),
    "Peru":        dict(language="Quechua",    capital="Lima",      continent="America", language_unique=True),
    "Cuba":        dict(language="Spanish",    capital="Havana",    continent="America", language_unique=False),
    "Argentina":   dict(language="Spanish",    capital="Buenos",    continent="America", language_unique=False),
}
FAMILIES = [
    ("lang_capital", "language", "capital",
     "In the country where people speak Arabic, the capital city is called Cairo. "
     "In the country where people speak Hebrew, the capital city is called Jerusalem. "
     "In the country where people speak {arg}, the capital city is called"),
    ("cap_language", "capital", "language",
     "The country governed from Cairo has one main language, namely Arabic. "
     "The country governed from Tokyo has one main language, namely Japanese. "
     "The country governed from {arg} has one main language, namely"),
]

REF = "/tiny-jlens/ref/jacobian-lens/data/experiments"
LANGUAGE_DATA = f"{REF}/selectivity-language.json"
LINECOUNT_DATA = f"{REF}/selectivity-linecount.json"

# --- S2a battery -----------------------------------------------------------------------------

# one-hop recall, few-shot (the same "capital of {country}" fact, one hop instead of two).
ONE_HOP = {
    "capital": ("The capital of Egypt is Cairo. The capital of Japan is Tokyo. "
                "The capital of {country} is", "capital"),
    "language": ("The main language of Egypt is Arabic. The main language of Japan is Japanese. "
                 "The main language of {country} is", "language"),
}

# induction / copying: a novel token pair, then filler, then the cue token -> copy its partner.
INDUCTION_POOL = [" apple", " river", " tiger", " velvet", " harbor", " maple", " copper",
                  " cat", " dog", " house", " water", " fire", " gold", " king", " book",
                  " tree", " star", " moon", " road", " door", " ship", " lake", " wolf",
                  " bear", " iron", " glass", " stone", " cloud", " forest", " garden"]
INDUCTION_FILLER = " The quick brown fox jumps over the lazy dog."


def two_hop_items(lm):
    """Criterion 3's two-hop items: few-shot frame + query, answer gated to greedy-correct.
    Reproduces c3's construction (shot-collision guard, single-token answers)."""
    items = []
    for fam, argf, ansf, tmpl in FAMILIES:
        fixed = tmpl.lower().replace("{arg}", "")
        for c, f in COUNTRIES.items():
            if argf == "language" and not f["language_unique"]:
                continue
            arg, ans = f[argf], f[ansf]
            if not lm.is_single(" " + c) or not lm.is_single(" " + ans):
                continue
            if c.lower() in fixed or ans.lower() in fixed:  # shot-collision guard
                continue
            text = tmpl.format(arg=arg)
            ids = lm.encode(text)
            if int(lm.logits(ids)[-1].argmax()) == lm.tid(" " + ans):
                items.append(dict(text=text, ids=ids, answer=ans, ans_id=lm.tid(" " + ans)))
    return items


def one_hop_items(lm):
    """One-hop recall over the same country facts, few-shot; gated to greedy-correct."""
    items = []
    for key, (tmpl, ansf) in ONE_HOP.items():
        for c, f in COUNTRIES.items():
            ans = f[ansf]
            if not lm.is_single(" " + ans) or c in ("Egypt", "Japan"):
                continue
            ids = lm.encode(tmpl.format(country=c))
            if int(lm.logits(ids)[-1].argmax()) == lm.tid(" " + ans):
                items.append(dict(text=tmpl.format(country=c), ids=ids, answer=ans,
                                  ans_id=lm.tid(" " + ans)))
    return items


def induction_items(lm, n=40, seed=1):
    """Copy the partner of a repeated token; gated to greedy-correct (pure induction)."""
    rng = random.Random(seed)
    pool = [w for w in INDUCTION_POOL if lm.is_single(w)]  # the lens/copy target must be one token
    items, tries = [], 0
    while len(items) < n and tries < 400:
        tries += 1
        a, b = rng.sample(pool, 2)
        text = f"{a}{b}{INDUCTION_FILLER}{a}"
        ids = lm.encode(text)
        if int(lm.logits(ids)[-1].argmax()) == lm.tid(b):
            items.append(dict(text=text, ids=ids, answer=b, ans_id=lm.tid(b)))
        if len(items) >= n:
            break
    return items


def pretraining_paragraphs(n=16, min_len=300, max_tokens=96):
    """Natural prose for the next-token top-1 match (paper: 'pretraining-like documents').
    wikitext-2-raw test -- the corpus the released gpt2-small lens was fit on."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    out = []
    for t in ds["text"]:
        s = t.strip()
        if len(s) >= min_len and not s.startswith("="):
            out.append(s)
        if len(out) >= n:
            break
    return out


# --- S2b language dissociation ---------------------------------------------------------------

LANG_DEMOS = {  # other-language demos for the few-shot language-ID cloze (none is fr/de/es/it)
    "Portuguese": "O rato comeu o queijo na cozinha ontem",
    "Dutch": "De kat zit op de mat bij het raam",
    "Swedish": "Katten sover pa stolen bredvid fonstret",
}
LANGS = ["French", "German", "Spanish", "Italian"]

# Short, unambiguous continuation phrases, one per language, used to grade the *automatic* task:
# after the (swapped) passage, does the model still prefer to continue in the passage's own
# language, or has the swap pushed it toward the alternative?  (A next-token language classifier
# is unreliable for the closely-related Romance languages; scoring a whole diagnostic phrase is
# robust.)  "the sun was shining", in each language.
LANG_CONT = {
    "French": " et le soleil brillait",
    "German": " und die Sonne schien",
    "Spanish": " y el sol brillaba",
    "Italian": " e il sole splendeva",
}


def language_data():
    return json.load(open(LANGUAGE_DATA))


def report_prompt(lm, text):
    """Few-shot language-ID cloze: '... Passage: {text}\\nLanguage:' -> the language name.
    Built from explicit token pieces so the passage span is exact (avoids the GPT-2
    leading-space tokenization drift)."""
    import torch
    prefix = "".join(f"Passage: {d}\nLanguage: {k}\n" for k, d in LANG_DEMOS.items()) + "Passage:"
    pid = lm.tok(prefix, add_special_tokens=False).input_ids
    tid = lm.tok(" " + text, add_special_tokens=False).input_ids
    sid = lm.tok("\nLanguage:", add_special_tokens=False).input_ids
    ids = torch.tensor([[lm.bos] + pid + tid + sid], device=lm.device)
    passage_pos = list(range(1 + len(pid), 1 + len(pid) + len(tid)))  # +1 for BOS
    return ids, passage_pos


def continuation_prompt(lm, text):
    """Automatic task: the passage alone; the model continues it in-language natively."""
    import torch
    tid = lm.tok(text, add_special_tokens=False).input_ids
    ids = torch.tensor([[lm.bos] + tid], device=lm.device)
    passage_pos = list(range(1, 1 + len(tid)))
    return ids, passage_pos
