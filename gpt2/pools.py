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
    # sample-size expansion (2026-08-19): rows authored from world knowledge
    # + tokenizer viability only, before any lens measurement; the build-time
    # single-token filter and the capability filter decide what survives.
    # language_unique stays one-canonical-country-per-language.
    "Netherlands": dict(language="Dutch",      capital="Amsterdam", continent="Europe", language_unique=True),
    "Ukraine":     dict(language="Ukrainian",  capital="Kiev",      continent="Europe", language_unique=True),
    "Romania":     dict(language="Romanian",   capital="Bucharest", continent="Europe", language_unique=True),
    "Bulgaria":    dict(language="Bulgarian",  capital="Sofia",     continent="Europe", language_unique=True),
    "Serbia":      dict(language="Serbian",    capital="Belgrade",  continent="Europe", language_unique=True),
    "Croatia":     dict(language="Croatian",   capital="Zagreb",    continent="Europe", language_unique=True),
    "Iceland":     dict(language="Icelandic",  capital="Reykjavik", continent="Europe", language_unique=True),
    "Albania":     dict(language="Albanian",   capital="Tirana",    continent="Europe", language_unique=True),
    "Belarus":     dict(language="Belarusian", capital="Minsk",     continent="Europe", language_unique=True),
    "Indonesia":   dict(language="Indonesian", capital="Jakarta",   continent="Asia",   language_unique=True),
    "Philippines": dict(language="Tagalog",    capital="Manila",    continent="Asia",   language_unique=True),
    "Pakistan":    dict(language="Urdu",       capital="Islamabad", continent="Asia",   language_unique=True),
    "Bangladesh":  dict(language="Bengali",    capital="Dhaka",     continent="Asia",   language_unique=True),
    "Nepal":       dict(language="Nepali",     capital="Kathmandu", continent="Asia",   language_unique=True),
    "Laos":        dict(language="Lao",        capital="Vientiane", continent="Asia",   language_unique=True),
    "Myanmar":     dict(language="Burmese",    capital="Naypyidaw", continent="Asia",   language_unique=True),
    "Cambodia":    dict(language="Khmer",      capital="Phnom Penh", continent="Asia",  language_unique=True),
    "Mongolia":    dict(language="Mongolian",  capital="Ulaanbaatar", continent="Asia", language_unique=True),
    "Malaysia":    dict(language="Malay",      capital="Kuala Lumpur", continent="Asia", language_unique=True),
    "Armenia":     dict(language="Armenian",   capital="Yerevan",   continent="Asia",   language_unique=True),
    "Georgia":     dict(language="Georgian",   capital="Tbilisi",   continent="Asia",   language_unique=True),
    "Iraq":        dict(language="Arabic",     capital="Baghdad",   continent="Asia",   language_unique=False),
    "Syria":       dict(language="Arabic",     capital="Damascus",  continent="Asia",   language_unique=False),
    "Lebanon":     dict(language="Arabic",     capital="Beirut",    continent="Asia",   language_unique=False),
    "Jordan":      dict(language="Arabic",     capital="Amman",     continent="Asia",   language_unique=False),
    "Libya":       dict(language="Arabic",     capital="Tripoli",   continent="Africa", language_unique=False),
    "Morocco":     dict(language="Arabic",     capital="Rabat",     continent="Africa", language_unique=False),
    "Algeria":     dict(language="Arabic",     capital="Algiers",   continent="Africa", language_unique=False),
    "Sudan":       dict(language="Arabic",     capital="Khartoum",  continent="Africa", language_unique=False),
    "Nigeria":     dict(language="English",    capital="Abuja",     continent="Africa", language_unique=False),
    "Kenya":       dict(language="Swahili",    capital="Nairobi",   continent="Africa", language_unique=False),
    "Ethiopia":    dict(language="Amharic",    capital="Addis Ababa", continent="Africa", language_unique=True),
    "Somalia":     dict(language="Somali",     capital="Mogadishu", continent="Africa", language_unique=True),
    "Cuba":        dict(language="Spanish",    capital="Havana",    continent="America", language_unique=False),
    "Chile":       dict(language="Spanish",    capital="Santiago",  continent="America", language_unique=False),
    "Peru":        dict(language="Spanish",    capital="Lima",      continent="America", language_unique=False),
    "Argentina":   dict(language="Spanish",    capital="Buenos Aires", continent="America", language_unique=False),
    "Colombia":    dict(language="Spanish",    capital="Bogota",    continent="America", language_unique=False),
    "Venezuela":   dict(language="Spanish",    capital="Caracas",   continent="America", language_unique=False),
}

# Two-hop families over the table. Each: (name, arg field, answer field,
# template over the arg). The intermediate is always the country. The
# lang_capital and city_language templates carry in-context shots built from
# Egypt/Israel/Kenya rows; the shot-collision guard in twohop_items excludes
# any item whose country or answer appears verbatim in a template's fixed
# text (the answer could be echoed from the prompt). Continent families
# stay shot-free: no shot phrasing beat them (capability wall at 124M).
TWOHOP_FAMILIES = [
    ("lang_capital",   "language", "capital",
     "In the country where people speak Arabic, the capital city is called "
     "Cairo. In the country where people speak Hebrew, the capital city is "
     "called Jerusalem. In the country where people speak {arg}, the capital "
     "city is called"),
    ("lang_continent", "language", "continent",
     "The country where {arg} is spoken is a country on the continent of"),
    ("city_language",  "capital", "language",
     "The country governed from Cairo has an official language, namely "
     "Arabic. The country governed from Nairobi has an official language, "
     "namely Swahili. The country governed from Jerusalem has an official "
     "language, namely Hebrew. The country governed from {arg} has an "
     "official language, namely"),
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
    # expansion riddles (same style; capability filter decides survival)
    dict(prompt="The number of legs on the animal that barks is",
         onehop="The number of legs on a dog is",
         intermediate="dog", answer="four",
         alts=dict(intermediate="spider", answer="eight")),
    dict(prompt="The number of legs on the animal that says meow is",
         onehop="The number of legs on a cat is",
         intermediate="cat", answer="four",
         alts=dict(intermediate="ant", answer="six")),
    dict(prompt="The number of legs on the insect that makes honey is",
         onehop="The number of legs on a bee is",
         intermediate="bee", answer="six",
         alts=dict(intermediate="spider", answer="eight")),
    dict(prompt="The color of the fruit that famously keeps the doctor away is",
         onehop="The color of an apple is",
         intermediate="apple", answer="red",
         alts=dict(intermediate="banana", answer="yellow")),
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
# the *2 paraphrase families share the per-country diagnostics
for _fam in ("lang_capital", "lang_continent", "city_language", "city_continent"):
    ONEHOP_TEMPLATES[_fam + "2"] = ONEHOP_TEMPLATES[_fam]
    FIRSTHOP_TEMPLATES[_fam + "2"] = FIRSTHOP_TEMPLATES[_fam]


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
    table = COUNTRIES
    items: list[TwoHop] = []
    for fam, argf, ansf, tmpl in TWOHOP_FAMILIES:
        pool = []
        fixed = tmpl.replace("{arg}", "").lower()
        for country, f in table.items():
            if argf == "language" and not f["language_unique"]:
                continue
            arg, ans = f[argf], f[ansf]
            if None in (_single(tokenizer, country), _single(tokenizer, ans)):
                continue
            # shot-collision guard: an item whose answer or intermediate
            # appears verbatim in the template's fixed text (the in-context
            # shots) is not a valid two-hop test — the answer could be
            # echoed from the prompt.
            if country.lower() in fixed or ans.lower() in fixed:
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

# The canonical format is the "Name a {cat}:" instruction register (the
# strongest of ~30 formats tried on gpt2-small; the colon is the readout
# anchor — the position immediately before the report, as in the paper).
# Shot categories/answers must stay disjoint from the eval categories a
# format is graded on; 20_capability enforces this per (format, category).
REPORT_FEWSHOT = (
    "Name a shape: circle\nName a fabric: wool\nName a building: church\n"
    "Name a utensil: spoon\nName a toy: doll\nName a {cat}:"
)

# Format variants (index 0 = canonical): the arrow variant of the same
# register (best for the report correlation), and two colon-list formats
# (the original bake-off register), so no result rides on one phrasing.
REPORT_FEWSHOTS = [
    REPORT_FEWSHOT,
    ("Name a tool -> hammer\nName a flower -> rose\nName a furniture -> chair\n"
     "Name a vehicle -> truck\nName a clothing -> shirt\nName a {cat} ->"),
    ("tool: hammer\nflower: rose\nfurniture: chair\n"
     "vehicle: truck\nclothing: shirt\n{cat}:"),
    ("appliance: oven\nfootwear: boot\ncontainer: jar\n"
     "weapon: spear\ndessert: cake\n{cat}:"),
]

EXTRA_CATEGORIES = {
    "tree": ["oak", "pine", "maple", "birch", "willow", "cedar", "elm", "ash",
             "spruce", "palm", "fir", "beech"],
    "metal": ["iron", "gold", "silver", "copper", "steel", "tin", "lead",
              "zinc", "nickel", "brass", "aluminum", "bronze"],
    "language": ["English", "French", "German", "Spanish", "Italian", "Russian",
                 "Chinese", "Japanese", "Arabic", "Greek", "Polish", "Korean"],
    "day of the week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                        "Saturday", "Sunday"],
    # expansion categories (2026-08-19); "May" deliberately omitted from month
    # (the lowercase modal " may" contaminates grading and target eligibility)
    "month": ["January", "February", "March", "April", "June", "July",
              "August", "September", "October", "November", "December"],
    "animal": ["dog", "cat", "horse", "lion", "tiger", "bear", "wolf", "fox",
               "deer", "rabbit", "mouse", "elephant", "monkey", "sheep",
               "goat", "pig"],
    "body part": ["arm", "leg", "hand", "foot", "head", "eye", "ear", "nose",
                  "mouth", "finger", "knee", "shoulder", "elbow", "ankle"],
    "vegetable": ["carrot", "potato", "onion", "pepper", "cabbage", "lettuce",
                  "spinach", "tomato", "pumpkin", "corn", "peas"],
    "insect": ["ant", "bee", "fly", "moth", "beetle", "cricket", "butterfly",
               "mosquito"],
    "gem": ["diamond", "ruby", "pearl", "amber", "quartz", "crystal"],
    # 2026-08-19 additions (single-token everyday categories). Several of
    # these double as shot categories of the colon-list formats; the
    # per-(format, category) shot-collision guard in 20_capability keeps a
    # category out of any format whose fixed text mentions it or a member.
    "furniture": ["chair", "table", "sofa", "couch", "bed", "desk", "shelf",
                  "bench", "stool", "cabinet", "lamp", "mattress"],
    "vehicle": ["car", "truck", "bus", "train", "boat", "ship", "plane",
                "van", "tractor", "jeep", "taxi", "ambulance"],
    "clothing": ["shirt", "coat", "jacket", "dress", "hat", "sweater",
                 "skirt", "glove", "scarf", "sock", "belt", "gown"],
    "flower": ["rose", "tulip", "daisy", "lily", "orchid", "sunflower",
               "poppy", "violet", "iris", "lavender", "jasmine", "daffodil"],
    "tool": ["hammer", "saw", "drill", "wrench", "shovel", "knife", "axe",
             "chisel", "rake", "pliers", "screwdriver", "file"],
}

# expansion members for the paper's curated categories: canonical members
# only (valid both as swap targets and for grading), merged into the curated
# lists in report_categories; multi-token forms are dropped by the token
# filter as usual.
CURATED_EXTENSIONS = {
    "country": ["Russia", "England", "Poland", "Turkey", "Sweden", "Greece",
                "Portugal", "Argentina", "Cuba", "Iran", "Israel", "Korea",
                "Vietnam", "Thailand", "Ireland", "Austria", "Hungary",
                "Finland", "Denmark", "Norway", "Ukraine", "Chile", "Peru",
                "Colombia", "Indonesia", "Netherlands", "Nigeria", "Morocco",
                "Iceland", "Croatia"],
    "language": ["Portuguese", "Dutch", "Turkish", "Hebrew", "Hindi",
                 "Swedish", "Danish", "Finnish", "Persian", "Thai",
                 "Vietnamese", "Ukrainian", "Romanian", "Norwegian",
                 "Hungarian", "Indonesian", "Croatian", "Serbian",
                 "Bulgarian", "Icelandic"],
    "fruit": ["pear", "peach", "plum", "lemon", "lime", "cherry", "grape",
              "mango", "coconut"],
    "beverage": ["coffee", "tea", "milk", "wine", "beer", "juice", "soda",
                 "water", "cider", "whiskey", "vodka"],
    "instrument": ["piano", "guitar", "violin", "drums", "trumpet"],
    "color": ["red", "blue", "green", "yellow", "orange", "purple", "pink",
              "brown", "black", "white", "gray", "violet", "crimson"],
    "sport": ["rugby", "boxing", "cricket", "golf", "tennis", "soccer",
              "baseball", "basketball", "hockey", "swimming", "cycling",
              "wrestling", "volleyball", "skiing"],
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
    cats = {c: list(dict.fromkeys(list(ms) + CURATED_EXTENSIONS.get(c, [])))
            for c, ms in cats.items()}
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
    # frame-diversity paraphrases; trailing digit marks the same base
    # function (broadcast stats must not count capital+capital2 as two
    # downstream operations)
    ("capital2",   "{arg}'s capital city is called"),
    ("language2",  "The people of {arg} mostly speak"),
    ("continent2", "{arg} is part of the continent called"),
]

def c4_grid(tokenizer):
    """[(category, func_name, template, {arg: answer})]"""
    grid = []
    arg_list = ["France", "Germany", "Spain", "Italy", "Poland", "China",
                "Japan", "Egypt", "India", "Sweden",
                # expansion args; capability filter per cell decides
                "Russia", "Greece", "Portugal", "Turkey", "Norway", "Denmark",
                "Finland", "Hungary", "Austria", "Ireland", "Israel", "Iran",
                "Korea", "Vietnam", "Thailand", "Canada", "Mexico", "Brazil",
                "Netherlands", "Ukraine", "Indonesia", "Argentina", "Chile",
                "Peru", "Cuba", "Nigeria", "Kenya", "Morocco", "Iraq"]
    args = [c for c in arg_list if _single(tokenizer, c) is not None]
    for fname, tmpl in C4_COUNTRY_FUNCS:
        answers = {}
        for c in args:
            key = fname.rstrip("0123456789")  # capital2 -> capital, etc.
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
    # expansion passages (2026-08-19), authored in the same register (short
    # everyday prose, ends mid-sentence on a comma); capability-filtered like
    # the rest.
    dict(category="French",  key="fx2", text="Le marché ouvrait à sept heures, et les marchands installaient leurs étals sous les arcades. Une odeur de pain chaud flottait dans la rue,"),
    dict(category="French",  key="fx3", text="Après le dîner, les enfants jouaient dans la cour pendant que leur grand-mère préparait le café. La nuit tombait doucement sur le village,"),
    dict(category="French",  key="fx4", text="Le train traversait la campagne sous un ciel gris. Claire regardait les champs défiler derrière la vitre, en pensant à sa sœur restée à Lyon,"),
    dict(category="French",  key="fx5", text="Chaque dimanche, le vieux professeur marchait jusqu'au bord du fleuve avec son chien. Il saluait les pêcheurs, achetait son journal,"),
    dict(category="French",  key="fx6", text="La bibliothèque était presque vide à cette heure. Une étudiante rangeait ses livres près de la fenêtre, et la pluie commençait à tomber,"),
    dict(category="French",  key="fx7", text="L'été dernier, nous avons loué une petite maison près de la mer. Le matin, on entendait les mouettes et le bruit des vagues,"),
    dict(category="French",  key="fx8", text="Le boulanger ferma la porte de sa boutique et compta la recette de la journée. Dehors, les lampadaires s'allumaient un à un,"),
    dict(category="German",  key="gx2", text="Am Bahnhof warteten nur wenige Reisende auf den letzten Zug. Der Wind trieb Zeitungen über den Bahnsteig, und die Uhr zeigte kurz vor Mitternacht,"),
    dict(category="German",  key="gx3", text="Jeden Sonntag backte die Großmutter einen Kuchen, und die ganze Familie versammelte sich in der kleinen Küche. Draußen läuteten die Glocken der Kirche,"),
    dict(category="German",  key="gx4", text="Der Lehrer öffnete das Fenster und ließ die frische Luft herein. Die Schüler schrieben still in ihre Hefte, während der Regen gegen das Dach trommelte,"),
    dict(category="German",  key="gx5", text="Im Herbst wurden die Tage kürzer, und der Nebel lag bis zum Mittag über dem Fluss. Die Bauern brachten die letzte Ernte in die Scheune,"),
    dict(category="German",  key="gx6", text="Nach der Arbeit ging Thomas noch durch den Park nach Hause. Die Blätter raschelten unter seinen Schuhen, und irgendwo bellte ein Hund,"),
    dict(category="German",  key="gx7", text="Die kleine Buchhandlung am Markt gehörte seit dreißig Jahren derselben Familie. Jeden Morgen stellte der alte Herr die Kisten mit Büchern vor die Tür,"),
    dict(category="German",  key="gx8", text="Das Dorf lag still zwischen den Hügeln, und aus den Schornsteinen stieg Rauch. Im Gasthaus saßen die Männer beim Bier und sprachen über das Wetter,"),
    dict(category="Spanish", key="sx2", text="El tren salió de la estación con veinte minutos de retraso. Ana miraba por la ventanilla los campos secos del verano, pensando en su pueblo,"),
    dict(category="Spanish", key="sx3", text="Cada domingo, la abuela preparaba arroz para toda la familia. Los niños jugaban en el patio mientras los mayores conversaban en la sombra,"),
    dict(category="Spanish", key="sx4", text="El pescador salió del puerto antes del amanecer. El mar estaba tranquilo y las luces del pueblo se reflejaban en el agua oscura,"),
    dict(category="Spanish", key="sx5", text="La tienda de la esquina cerraba a las nueve. Don Manuel contaba las monedas detrás del mostrador mientras su hijo barría el suelo,"),
    dict(category="Spanish", key="sx6", text="En invierno, las calles del barrio quedaban vacías después de las ocho. Solo el panadero seguía trabajando, y el olor a pan llenaba la noche,"),
    dict(category="Spanish", key="sx7", text="Marta encontró las cartas de su madre en una caja del armario. Se sentó junto a la ventana y empezó a leerlas una por una,"),
    dict(category="Spanish", key="sx8", text="El autobús subía despacio por la carretera de la montaña. Abajo, el valle se llenaba de niebla y se encendían las primeras luces del pueblo,"),
    dict(category="Italian", key="ix2", text="La domenica mattina il paese si svegliava tardi. Le campane suonavano a distesa e le donne uscivano dalla chiesa parlando del più e del meno,"),
    dict(category="Italian", key="ix3", text="Il treno per Milano partiva alle sei. Paolo aspettava sulla banchina con la valigia, guardando le luci della città che si accendevano,"),
    dict(category="Italian", key="ix4", text="Ogni estate andavamo dalla nonna in campagna. Il pomeriggio raccoglievamo i pomodori nell'orto e la sera mangiavamo tutti insieme sotto il pergolato,"),
    dict(category="Italian", key="ix5", text="Il negozio di alimentari chiudeva a mezzogiorno. La signora Rosa sistemava la frutta sugli scaffali mentre il gatto dormiva vicino alla porta,"),
    dict(category="Italian", key="ix6", text="La pioggia cadeva da tre giorni sulla città. Luca guardava dalla finestra dell'ufficio i passanti con gli ombrelli, aspettando la fine del turno,"),
    dict(category="Italian", key="ix7", text="Il vecchio pescatore riparava le reti seduto sul molo. Le barche rientravano una dopo l'altra, e il sole scendeva dietro le colline,"),
    dict(category="Italian", key="ix8", text="In autunno il bosco dietro casa si riempiva di funghi. Mio padre si alzava presto, prendeva il cesto e il bastone,"),
    # third batch (2026-08-19 late), same register
    dict(category="French",  key="fx9",  text="Le facteur passait toujours vers dix heures. Ce matin-là, il apportait un colis pour la voisine du deuxième étage, qui l'attendait depuis une semaine,"),
    dict(category="French",  key="fx10", text="Au printemps, les étudiants révisaient leurs examens dans le parc. On entendait les pages tourner et les conversations à voix basse,"),
    dict(category="French",  key="fx11", text="Le garagiste ferma le capot de la vieille voiture et s'essuya les mains. Le moteur tournait enfin sans faire de bruit,"),
    dict(category="German",  key="gx9",  text="Die Kinder bauten den ganzen Nachmittag eine Burg aus Sand. Als die Sonne unterging, packte die Mutter die Taschen zusammen,"),
    dict(category="German",  key="gx10", text="Der Bäcker stand jeden Morgen um vier Uhr auf. In der Backstube roch es nach frischem Brot und warmem Zucker,"),
    dict(category="German",  key="gx11", text="Im Winter fuhren nur wenige Autos über die alte Brücke. Der Fluss darunter war zugefroren, und die Enten saßen am Ufer,"),
    dict(category="Spanish", key="sx9",  text="El jardinero regaba las plantas antes de que saliera el sol. Las calles todavía estaban vacías y frescas,"),
    dict(category="Spanish", key="sx10", text="Mi tío arreglaba relojes en un taller pequeño del centro. Cada tarde, los clientes traían piezas antiguas,"),
    dict(category="Spanish", key="sx11", text="La biblioteca del barrio abría a las diez. Los estudiantes esperaban en la puerta con sus cuadernos bajo el brazo,"),
    dict(category="Italian", key="ix9",  text="Ogni mattina il fornaio apriva la bottega alle sei. Il profumo del pane fresco riempiva tutta la via,"),
    dict(category="Italian", key="ix10", text="D'estate i bambini giocavano in piazza fino a tardi. Le madri li chiamavano dalle finestre quando calava la sera,"),
    dict(category="Italian", key="ix11", text="Il giardiniere tagliava l'erba del parco ogni venerdì. I passanti si fermavano a guardare le aiuole fiorite,"),
]

# QA register with NON-ENGLISH demos only (Dutch and Portuguese; neither is
# an eval language). An English demo re-installs the model's English-default
# prior, which absorbs the report whenever the passage's own language
# coordinate is suppressed — the failed swaps under English-demo formats all
# output " English", and removing the demo removes the failure.
LANG_REPORT_FEWSHOT = (
    'Q: What language is this text written in? "De zon scheen fel boven de stille straten van de stad."\n'
    'A: Dutch\n'
    'Q: What language is this text written in? "Os barcos voltaram ao porto ao anoitecer."\n'
    'A: Portuguese\n'
    'Q: What language is this text written in? "{passage}"\nA:'
)
LANG_COUNTRY_FEWSHOT = (
    'Text: "The rain fell softly on the quiet village roofs."\nCountry where this language is spoken: England\n'
    'Text: "De zon scheen fel boven de stille straten van de stad."\nCountry where this language is spoken: Holland\n'
    'Text: "{passage}"\nCountry where this language is spoken:'
)

# Frame-diversity variants (2026-08-19): same two-shot structure, different
# wording and a different non-eval demonstration language (Swedish instead of
# Dutch). Tests, among other things, whether the French->'English' collapse
# under report swaps is format-specific.
LANG_REPORT_FEWSHOT2 = (
    'Sentence: "The boats returned to the harbor at sunset."\nThis sentence is written in: English\n'
    'Sentence: "Solen gick ner bakom husen vid det gamla torget."\nThis sentence is written in: Swedish\n'
    'Sentence: "{passage}"\nThis sentence is written in:'
)
LANG_COUNTRY_FEWSHOT2 = (
    'Sentence: "The boats returned to the harbor at sunset."\nThis language is spoken in the country of: England\n'
    'Sentence: "Solen gick ner bakom husen vid det gamla torget."\nThis language is spoken in the country of: Sweden\n'
    'Sentence: "{passage}"\nThis language is spoken in the country of:'
)

# language classification of generated continuations: langid restricted to
# the five relevant languages (hand-rolled stopword sets could not separate
# Romance languages on 16-token snippets)
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
            "engine", "forest", "candle", "rocket", "violin", "desert",
            # expansion words (single-token concrete nouns, disjoint from
            # every C2 sentence; asserted below)
            "bridge", "piano", "river", "valley", "temple", "dragon",
            "crystal", "anchor", "blanket", "bottle", "camera", "carpet",
            "compass", "curtain", "feather", "fountain", "helmet", "island",
            "jungle", "ladder", "lantern", "needle", "palace", "pillow",
            # third batch (2026-08-19 late)
            "arrow", "basket", "canoe", "cannon", "cottage", "glove",
            "kettle", "parrot", "quilt", "raft", "wagon", "whale"]
C2_SENTENCES = [
    "The old painting hung crookedly on the wall.",
    "A cold wind blew across the empty station platform.",
    "The children played quietly in the garden after lunch.",
    # expansion sentences (same register: mundane declaratives)
    "The meeting started late because the projector refused to work.",
    "She wrote the address carefully on the back of the envelope.",
    "The bakery on the corner sells fresh bread every morning.",
    "He parked the car outside the library and waited patiently.",
    "They painted the kitchen a pale shade of blue last spring.",
    "The teacher asked everyone to open their books to page forty.",
    "The shop closed early because of the heavy afternoon traffic.",
    # third batch (2026-08-19 late)
    "The committee approved the budget after a short discussion.",
    "Her office was on the third floor of the old building.",
    "Lunch was served in the small dining room at noon.",
    "The two clerks counted the votes again before midnight.",
]

# materials invariant: no held word may appear in any transcription sentence
# (a mention would contaminate the think/dont/base contrast)
for _w in C2_WORDS:
    for _s in C2_SENTENCES:
        assert _w.lower() not in _s.lower(), f"C2 word {_w!r} occurs in {_s!r}"
# Template roles: think = attend instruction, dont = suppress instruction.
# Paper taxonomy note: "Think about X" matches the paper's attend/focus
# type, but "Don't think about X" is the paper's PROHIBITIVE variant, which
# the paper reports suppresses less than its ignore phrasings ("X is
# irrelevant to this task") — the latter is covered by C2_PHRASINGS below.
# The earlier bake-off's separate focus/ignore text pair was removed with
# the confirm plumbing and its receipt no longer exists; the checked-in
# receipt for this pair gives think<dont 24/36.
C2_TEMPLATES = {
    "think":  'Write "{s}" Think about {w} while you write the sentence. "{s_open}',
    "dont":   'Write "{s}" Don\'t think about {w} while you write the sentence. "{s_open}',
    "base":   'Write "{s}" "{s_open}',
}

# Frame diversity (2026-08-19): multiple phrasings per condition (mirrors
# the paper's phrasing-sensitivity appendix: focus/concentrate variants for
# attend, prohibit/irrelevant variants for suppress). Index 0 of each list is
# the canonical phrasing above; the base condition has no instruction to vary.
C2_PHRASINGS = {
    "think": [
        'Write "{s}" Think about {w} while you write the sentence. "{s_open}',
        'Write "{s}" Concentrate on {w} while you write the sentence. "{s_open}',
        'Write "{s}" Focus on {w} while you write the sentence. "{s_open}',
        'Write "{s}" Keep {w} in mind while you write the sentence. "{s_open}',
        'Write "{s}" While you write, hold a single thought: {w}. "{s_open}',
    ],
    "dont": [
        'Write "{s}" Don\'t think about {w} while you write the sentence. "{s_open}',
        'Write "{s}" Ignore {w} while you write the sentence. "{s_open}',
        'Write "{s}" {w} is irrelevant to this task; just write the sentence. "{s_open}',
        'Write "{s}" Try not to think about {w} while you write the sentence. "{s_open}',
        # pre-positioned suppress: the instruction comes BEFORE the word's
        # only mention, so the word is not re-primed right at the
        # transcription span. Instruction position is a large lever at 124M:
        # this phrasing suppresses to near-base ranks while the terminal
        # phrasings above sit far lower.
        '{w} is irrelevant to this task. Write "{s}" "{s_open}',
    ],
    "base": ['Write "{s}" "{s_open}'],
}
INJECT_CONCEPTS = ["ocean", "fire", "music", "winter", "horses", "coffee",
                   "gold", "rain", "chess", "bread", "anger", "ships",
                   "glass", "honey", "wolves", "silk", "thunder", "cotton",
                   "iron", "roses", "salt", "dreams", "mirrors", "wheat",
                   # expansion concepts
                   "storm", "dust", "snow", "smoke", "copper", "velvet",
                   "garlic", "pepper", "maple", "amber", "ivory", "marble",
                   "canyon", "harbor", "comet", "planet", "tide", "frost",
                   "ash", "clay", "pearl", "flame",
                   # third batch (2026-08-19 late); multi-token forms are
                   # dropped at runtime by the single-token filter
                   "butter", "candy", "cheese", "cloud", "coral", "fog",
                   "gravel", "lava", "moss", "oil", "opera", "paint",
                   "poison", "rust", "sand", "shell", "sponge", "steam",
                   "sugar", "swamp", "wax", "wool", "cave", "cliff",
                   "whistle", "wagon", "ferry", "vinegar", "leather",
                   "chalk", "hay", "straw", "fur", "dew"]

# C1d frames. Index 0 of each list is the canonical pair: the first-person
# report frame with the strongest selective window (s ~ 0.15-0.25), and the
# other-mind control — identical syntax, same open noun slot, but the noun
# is attributed to someone ELSE's mind, so a self-report reading is blocked.
# The remaining controls are NOUN-expecting (a verb-expecting control would
# earn a low blurt rate for free); none asks what the speaker is thinking.
INJECT_REPORT_FRAMES = [
    "All day my thoughts kept returning to one thing, and that thing was",
    "The word that best describes what I am thinking about right now is",
    "If I had to name one thing on my mind right now, it would be",
    "Right now, the main thing I keep thinking about is",
    "My mind keeps drifting back to one particular topic, namely",
    "Someone just asked me what I am thinking about, and my honest answer is",
]
INJECT_CONTROL_FRAMES = [
    "The word she wrote at the top of the page was",
    "The weather report for tomorrow morning said it would be",
    "She opened the box and found a small",
    "The first prize in the village raffle was a brand new",
    "At the bottom of the old suitcase there was a",
    "The shop on the corner mostly sells things like",
]
# no frame may contain an inject concept (a mention would contaminate ranks)
for _f in INJECT_REPORT_FRAMES + INJECT_CONTROL_FRAMES:
    for _w in INJECT_CONCEPTS:
        assert _w.lower() not in _f.lower(), f"concept {_w!r} occurs in frame {_f!r}"

# C2 demand-loading (43) frame diversity: three shot-set variants with the
# same structure (remember word -> copy -> recall / no recall). Remember-words
# are disjoint from C2_WORDS (asserted).
DEMAND_VARIANTS = [
    dict(words=("lamp", "kite", "moss"),
         sents=("The road was wet.", "Dinner is at six.", "He lost his keys."),
         done="Done."),
    dict(words=("rope", "coin", "leaf"),
         sents=("The bus was late again.", "Her coat was warm.", "The gate swung open."),
         done="Finished."),
    dict(words=("stone", "cloth", "torch"),
         sents=("The door creaked loudly.", "It snowed all afternoon.", "The soup was too hot."),
         done="All done."),
]
for _v in DEMAND_VARIANTS:
    for _w in _v["words"]:
        assert _w not in C2_WORDS, f"demand shot word {_w!r} collides with C2_WORDS"


def demand_shots(variant: dict) -> tuple[str, str]:
    """(demand shots, no-demand shots) for one variant."""
    d_lines, n_lines = [], []
    for w, s in zip(variant["words"], variant["sents"]):
        head = f'Remember the word "{w}". Copy: "{s}" -> "{s}"'
        d_lines.append(f'{head} The word was "{w}".')
        n_lines.append(f'{head} {variant["done"]}')
    return "\n".join(d_lines) + "\n", "\n".join(n_lines) + "\n"


# --------------------------------------------------------------------------
# C5b: shallow (automatic) task suite for the ablation battery.
#
# Every task here is automatic — a memorized sequence, idiom, collocation,
# or grammatical reflex, no inference required — but the tasks differ in the
# CLASS of their answer token (content word / function word / punctuation).
# That split is the point: shallow-vs-flexible and content-vs-function
# output class are confounded in a wikitext-style battery, and the ablation
# damage tracks output class, not task depth. Items are (prompt, expected
# single token); each experiment applies its own clean-top-1 filter.
# --------------------------------------------------------------------------

SHALLOW_SUITE = {
    "seq_formulaic": [   # answers: content words (memorized sequences)
        ("Wednesday, Thursday, Friday,", " Saturday"), ("April, May, June,", " July"),
        ("May, June, July,", " August"), ("three, four, five,", " six"),
        ("seven, eight, nine,", " ten"), ("4, 5, 6, 7,", " 8"),
        ("10, 11, 12, 13,", " 14"), ("h, i, j,", " k"), ("m, n, o,", " p"),
        ("second, third,", " fourth"), ("October, November,", " December"),
        ("five, six, seven,", " eight"),
    ],
    "idiom": [           # answers: content words
        ("Out of sight, out of", " mind"), ("A picture is worth a thousand", " words"),
        ("The early bird catches the", " worm"), ("Don't judge a book by its", " cover"),
        ("Two birds with one", " stone"), ("The grass is always greener on the other", " side"),
        ("All that glitters is not", " gold"), ("Rome wasn't built in a", " day"),
        ("The apple doesn't fall far from the", " tree"), ("Curiosity killed the", " cat"),
        ("Barking up the wrong", " tree"), ("Absence makes the heart grow", " fonder"),
    ],
    "collocation": [     # answers: content words
        ("black and", " white"), ("up and", " down"), ("day and", " night"),
        ("safe and", " sound"), ("loud and", " clear"), ("short and", " sweet"),
        ("far and", " wide"), ("first and", " foremost"), ("odds and", " ends"),
        ("give and", " take"), ("trial and", " error"), ("thick and", " thin"),
    ],
    "func_cloze": [      # answers: function words
        ("She looked forward", " to"), ("It consists", " of"), ("They insisted", " on"),
        ("He is capable", " of"), ("This results", " in"), ("He apologized", " for"),
        ("It reminds me", " of"), ("She is afraid", " of"), ("instead", " of"),
        ("because", " of"), ("They believe", " in"), ("on behalf", " of"),
    ],
    "punct": [           # answers: punctuation-class tokens
        ("She didn", "'t"), ("They wouldn", "'t"), ("It isn", "'t"),
        ("We couldn", "'t"), ("He hasn", "'t"), ("I shouldn", "'t"),
        ("You mustn", "'t"), ("It wasn", "'t"),
    ],
    "agreement": [       # answers: function words (verb agreement)
        ("One of the boxes", " is"), ("The books on the shelf", " are"),
        ("The color of the walls", " is"), ("Both of the brothers", " are"),
        ("The list of names", " is"), ("People in the village", " are"),
        ("Each of the players", " is"), ("The keys to the cabinet", " are"),
    ],
}
