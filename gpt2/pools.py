"""Task pools for the GPT-2 experiments.

Everything is data + light tokenizer filtering; running/grading lives in the
experiment scripts. Two sources:
  - the paper's own shipped items (ref/jacobian-lens/data/...), imported
    verbatim and capability-filtered like everything else;
  - GPT-2-targeted families built around COUNTRIES, one fact table that
    supplies two-hop items (language→capital etc.), the cross-function
    anti-smuggling control, and the C4 function grid — all sharing the same
    country intermediates.

Single-token constraints are enforced at build time with the (shared) GPT-2
tokenizer: an item is only emitted if its intermediate and answer have a
single-token leading-space form.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

REF = "/tiny-jlens/ref/jacobian-lens"

# --------------------------------------------------------------------------
# Country facts: country -> (language, capital, continent)
# language_unique: language -> country is unambiguous enough for a two-hop
# --------------------------------------------------------------------------

COUNTRIES: dict[str, dict] = {
    "France":   dict(language="French",     capital="Paris",      continent="Europe", language_unique=True),
    "Germany":  dict(language="German",     capital="Berlin",     continent="Europe", language_unique=True),
    "Spain":    dict(language="Spanish",    capital="Madrid",     continent="Europe", language_unique=True),
    "Italy":    dict(language="Italian",    capital="Rome",       continent="Europe", language_unique=True),
    "Poland":   dict(language="Polish",     capital="Warsaw",     continent="Europe", language_unique=True),
    "Russia":   dict(language="Russian",    capital="Moscow",     continent="Europe", language_unique=True),
    "China":    dict(language="Chinese",    capital="Beijing",    continent="Asia",   language_unique=True),
    "Japan":    dict(language="Japanese",   capital="Tokyo",      continent="Asia",   language_unique=True),
    "Portugal": dict(language="Portuguese", capital="Lisbon",     continent="Europe", language_unique=True),
    "Greece":   dict(language="Greek",      capital="Athens",     continent="Europe", language_unique=True),
    "Egypt":    dict(language="Arabic",     capital="Cairo",      continent="Africa", language_unique=False),
    "India":    dict(language="Hindi",      capital="Delhi",      continent="Asia",   language_unique=True),
    "Turkey":   dict(language="Turkish",    capital="Ankara",     continent="Asia",   language_unique=True),
    "Sweden":   dict(language="Swedish",    capital="Stockholm",  continent="Europe", language_unique=True),
    "Norway":   dict(language="Norwegian",  capital="Oslo",       continent="Europe", language_unique=True),
    "Denmark":  dict(language="Danish",     capital="Copenhagen", continent="Europe", language_unique=True),
    "Finland":  dict(language="Finnish",    capital="Helsinki",   continent="Europe", language_unique=True),
    "Hungary":  dict(language="Hungarian",  capital="Budapest",   continent="Europe", language_unique=True),
    "Austria":  dict(language="German",     capital="Vienna",     continent="Europe", language_unique=False),
    "Ireland":  dict(language="English",    capital="Dublin",     continent="Europe", language_unique=False),
    "Israel":   dict(language="Hebrew",     capital="Jerusalem",  continent="Asia",   language_unique=True),
    "Iran":     dict(language="Persian",    capital="Tehran",     continent="Asia",   language_unique=True),
    "Korea":    dict(language="Korean",     capital="Seoul",      continent="Asia",   language_unique=True),
    "Vietnam":  dict(language="Vietnamese", capital="Hanoi",      continent="Asia",   language_unique=True),
    "Thailand": dict(language="Thai",       capital="Bangkok",    continent="Asia",   language_unique=True),
    "Canada":   dict(language="English",    capital="Ottawa",     continent="America", language_unique=False),
    "Mexico":   dict(language="Spanish",    capital="Mexico",     continent="America", language_unique=False),
    "Brazil":   dict(language="Portuguese", capital="Bras",       continent="America", language_unique=False),
}

# Two-hop families over the table. Each: (name, arg field, answer field,
# template over the arg). The intermediate is always the country.
TWOHOP_FAMILIES = [
    # templates chosen by a format bake-off on gpt2-small (see LOG.md):
    # base GPT-2 needs quiz-register phrasings; "is spoken is the city of"
    # nearly doubles the pass rate over "primary language ... is".
    ("lang_capital",   "language", "capital",
     "The capital of the country where {arg} is spoken is the city of"),
    ("lang_continent", "language", "continent",
     "The country where {arg} is spoken is a country on the continent of"),
    ("city_language",  "capital", "language",
     "In the country whose capital city is {arg}, the primary language is"),
    ("city_continent", "capital", "continent",
     "The country whose capital is {arg} is located on the continent of"),
]

# Riddle-style two-hops (paper's "animal that spins webs" style), intermediate
# is the unspoken concept.
RIDDLES = [
    dict(prompt="The number of legs on the animal that spins webs is",
         onehop="The number of legs on a spider is",
         intermediate="spider", answer="eight",
         alts=dict(intermediate="ant", answer="six")),
    dict(prompt="The number of legs on the animal that says moo is",
         onehop="The number of legs on a cow is",
         intermediate="cow", answer="four",
         alts=dict(intermediate="spider", answer="eight")),
    dict(prompt="The color of the fruit that monkeys famously eat is",
         onehop="The color of a banana is",
         intermediate="banana", answer="yellow",
         alts=dict(intermediate="cherry", answer="red")),
    dict(prompt="The color of the vegetable that rabbits famously eat is",
         onehop="The color of a carrot is",
         intermediate="carrot", answer="orange",
         alts=dict(intermediate="lettuce", answer="green")),
    dict(prompt="The color of the planet fourth from the Sun is",
         onehop="The color of the planet Mars is",
         intermediate="Mars", answer="red",
         alts=dict(intermediate="Neptune", answer="blue")),
]

# One-hop (second hop given the intermediate) and first-hop (cue ->
# intermediate) templates for the country families, used as capability
# diagnostics alongside the two-hop item itself.
ONEHOP_TEMPLATES = {
    # quiz-register formats (bake-off: "The capital of Poland is" -> " now
    # home to..." in prose register, 0/26; the analogy shot gets 22/26)
    "lang_capital":   "Kenya's capital is Nairobi. {country}'s capital is",
    "lang_continent": "{country} is a country on the continent of",
    "city_language":  "The primary language spoken in {country} is",
    "city_continent": "{country} is a country on the continent of",
}
FIRSTHOP_TEMPLATES = {
    "lang_capital":   "The country where {arg} is spoken is called",
    "lang_continent": "The country where {arg} is spoken is called",
    "city_language":  "The country whose capital city is {arg} is called",
    "city_continent": "The country whose capital city is {arg} is called",
}


@dataclass
class TwoHop:
    family: str
    prompt: str
    arg: str            # the surface cue (language/capital/riddle key)
    intermediate: str   # unspoken concept (country etc.)
    answer: str
    source: str = "ours"
    swap_pool: list = field(default_factory=list)  # (intermediate', answer') partners
    onehop: str | None = None    # second hop given the intermediate
    firsthop: str | None = None  # cue -> intermediate


def _single(tokenizer, word: str) -> int | None:
    ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def twohop_items(tokenizer) -> list[TwoHop]:
    items: list[TwoHop] = []
    for fam, argf, ansf, tmpl in TWOHOP_FAMILIES:
        pool = []
        for country, f in COUNTRIES.items():
            if argf == "language" and not f["language_unique"]:
                continue
            arg, ans = f[argf], f[ansf]
            if None in (_single(tokenizer, country), _single(tokenizer, ans)):
                continue
            pool.append((country, arg, ans))
        for country, arg, ans in pool:
            partners = [(c2, a2) for (c2, _, a2) in pool if c2 != country and a2 != ans]
            items.append(TwoHop(fam, tmpl.format(arg=arg), arg, country, ans,
                                swap_pool=partners,
                                onehop=ONEHOP_TEMPLATES[fam].format(country=country),
                                firsthop=FIRSTHOP_TEMPLATES[fam].format(arg=arg)))
    for r in RIDDLES:
        if None in (_single(tokenizer, r["intermediate"]), _single(tokenizer, r["answer"]),
                    _single(tokenizer, r["alts"]["intermediate"]), _single(tokenizer, r["alts"]["answer"])):
            continue
        items.append(TwoHop("riddle", r["prompt"], r["prompt"].split()[-3],
                            r["intermediate"], r["answer"],
                            swap_pool=[(r["alts"]["intermediate"], r["alts"]["answer"])],
                            onehop=r["onehop"]))
    # the paper's own two-hop sets
    with open(f"{REF}/data/experiments/probe-swap.json") as f:
        for it in json.load(f)["items"]:
            if None in (_single(tokenizer, it["intermediate"]), _single(tokenizer, it["answer"]),
                        _single(tokenizer, it["swap_to"]), _single(tokenizer, it["swap_answer"])):
                continue
            items.append(TwoHop(f"paper_{it['category']}", it["prompt"].rstrip(), it["name"],
                                it["intermediate"], it["answer"], source="paper",
                                swap_pool=[(it["swap_to"], it["swap_answer"])]))
    return items


# --------------------------------------------------------------------------
# C1: category report (few-shot, base-model form)
# --------------------------------------------------------------------------

# Format chosen by bake-off (LOG.md): colon-list with 5 shots, all shot
# categories/answers disjoint from every eval category (v1's QA shots leaked
# answers; QA formats also induce category echo). The colon is the readout
# anchor — the position immediately before the report, as in the paper.
REPORT_FEWSHOT = (
    "tool: hammer\nflower: rose\nfurniture: chair\n"
    "vehicle: truck\nclothing: shirt\n{cat}:"
)

EXTRA_CATEGORIES = {
    "tree": ["oak", "pine", "maple", "birch", "willow", "cedar", "elm", "ash",
             "spruce", "palm", "fir", "beech"],
    "metal": ["iron", "gold", "silver", "copper", "steel", "tin", "lead",
              "zinc", "nickel", "brass", "aluminum", "bronze"],
    "language": ["English", "French", "German", "Spanish", "Italian", "Russian",
                 "Chinese", "Japanese", "Arabic", "Greek", "Polish", "Korean"],
    "day of the week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                        "Saturday", "Sunday"],
}

# legitimate members the paper's 14-item lists omit (grading additions only;
# targets for swaps are still drawn from the curated lists)
MEMBER_ADDITIONS = {
    "country": ["USA", "America", "Russia", "England", "Canada", "Australia", "Brazil"],
    "fruit": ["pear", "peach", "plum", "melon", "lemon"],
    "sport": ["rugby", "boxing", "cricket", "golf"],
    "profession": ["farmer", "artist", "writer", "singer", "gardener"],
    "beverage": ["tea", "milk", "wine", "beer", "juice", "soda", "water"],
    "city": ["London", "Paris", "York", "Chicago", "Boston"],
}


def report_categories(tokenizer, with_additions: bool = False) -> dict[str, dict[str, int]]:
    """{category: {member: leading-space token id}} with >=6 members.
    with_additions=True widens the member lists for *grading* validity
    (swap targets should still come from the curated lists)."""
    with open(f"{REF}/data/experiments/verbal-report.json") as f:
        cats = json.load(f)["candidates"]
    cats = {**cats, **EXTRA_CATEGORIES}
    if with_additions:
        cats = {c: list(ms) + MEMBER_ADDITIONS.get(c, []) for c, ms in cats.items()}
    out = {}
    for cat, members in cats.items():
        toks = {}
        for m in members:
            t = _single(tokenizer, m) or _single(tokenizer, m.capitalize())
            if t is not None:
                toks[m] = t
        if len(toks) >= 6:
            out[cat] = toks
    return out


# --------------------------------------------------------------------------
# C4: function grid (countries functions share COUNTRIES; others from paper)
# --------------------------------------------------------------------------

C4_COUNTRY_FUNCS = [
    ("capital",   "The capital of {arg} is the city of"),
    ("language",  "The primary language spoken in {arg} is"),
    ("continent", "{arg} is located on the continent of"),
]

def c4_grid(tokenizer):
    """[(category, func_name, template, {arg: answer})]"""
    grid = []
    args = [c for c in ["France", "Germany", "Spain", "Italy", "Poland", "China",
                        "Japan", "Egypt", "India", "Sweden"]
            if _single(tokenizer, c) is not None]
    for fname, tmpl in C4_COUNTRY_FUNCS:
        answers = {}
        for c in args:
            key = {"capital": "capital", "language": "language", "continent": "continent"}[fname]
            ans = COUNTRIES[c][key]
            if _single(tokenizer, ans) is not None:
                answers[c] = ans
        grid.append(("countries", fname, tmpl, answers))
    with open(f"{REF}/data/experiments/flexible-generalization.json") as f:
        for cat in json.load(f)["categories"]:
            if cat["name"] == "countries":
                continue  # ours above, richer
            for fn in cat["funcs"]:
                answers = {a: ans for a, ans in fn["answers"].items()
                           if _single(tokenizer, a) is not None and _single(tokenizer, ans) is not None}
                if len(answers) >= 3:
                    grid.append((cat["name"], fn["name"], fn["template"], answers))
    return grid


# --------------------------------------------------------------------------
# C5a: language selectivity (passages + few-shot report templates)
# --------------------------------------------------------------------------

def c5_passages():
    with open(f"{REF}/data/experiments/selectivity-language.json") as f:
        d = json.load(f)
    return d["passages"]  # [{category, key, text}]

EXTRA_PASSAGES = [
    dict(category="French",  key="fx1", text="Il pleuvait depuis le matin, et les rues de la petite ville restaient vides. Jean regardait par la fenêtre en buvant son café,"),
    dict(category="German",  key="gx1", text="Der alte Mann ging jeden Morgen durch den Wald. Die Vögel sangen in den Bäumen, und die Sonne schien durch die Blätter,"),
    dict(category="Spanish", key="sx1", text="La ciudad despertaba lentamente bajo el sol de la mañana. María caminaba por la plaza con su cesta de pan,"),
    dict(category="Italian", key="ix1", text="Il mare era calmo quella mattina, e le barche dei pescatori tornavano al porto. Marco guardava l'orizzonte,"),
]

# two shots, one English one Dutch (Dutch is never an eval language), so the
# format demonstrably admits non-English answers
LANG_REPORT_FEWSHOT = (
    'Text: "The rain fell softly on the quiet village roofs."\nLanguage: English\n'
    'Text: "De zon scheen fel boven de stille straten van de stad."\nLanguage: Dutch\n'
    'Text: "{passage}"\nLanguage:'
)
LANG_COUNTRY_FEWSHOT = (
    'Text: "The rain fell softly on the quiet village roofs."\nCountry where this language is spoken: England\n'
    'Text: "De zon scheen fel boven de stille straten van de stad."\nCountry where this language is spoken: Holland\n'
    'Text: "{passage}"\nCountry where this language is spoken:'
)

# language classification of generated continuations: langid restricted to
# the five relevant languages (hand-rolled stopword sets could not separate
# Romance languages on 16-token snippets — see LOG.md)
_LANGID_NAMES = {"fr": "French", "de": "German", "es": "Spanish",
                 "it": "Italian", "en": "English"}


def classify_language(text: str) -> str:
    import langid

    langid.set_languages(list(_LANGID_NAMES))
    if len(text.strip()) < 8:
        return "unknown"
    code, _ = langid.classify(text)
    return _LANGID_NAMES[code]

# --------------------------------------------------------------------------
# C2: think/don't-think materials (exploration set; held-out reserved later)
# --------------------------------------------------------------------------

C2_WORDS = ["ocean", "guitar", "castle", "tiger", "lemon", "mirror",
            "engine", "forest", "candle", "rocket", "violin", "desert"]
C2_SENTENCES = [
    "The old painting hung crookedly on the wall.",
    "A cold wind blew across the empty station platform.",
    "The children played quietly in the garden after lunch.",
]
C2_TEMPLATES = {
    "think":  'Write "{s}" Think about {w} while you write the sentence. "{s_open}',
    "dont":   'Write "{s}" Don\'t think about {w} while you write the sentence. "{s_open}',
    "base":   'Write "{s}" "{s_open}',
}
