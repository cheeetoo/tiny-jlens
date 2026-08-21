"""Prompt material for criterion 2 (directed modulation), from the paper's released data
(ref/jacobian-lens/data/experiments/{directed-modulation,top-down-summoning}.json) plus the
imagine-property materials for the privileging test (paper §A "Directed modulation affects the
J-space more than other representations").

Base-model frame (deviation; see PROTOCOL.md).  The paper runs a chat: the user gives an
instruction about a target X and asks the model to copy an unrelated carrier sentence; the lens
is read over the model's copy, where the surface text is the carrier (unrelated to X).  GPT-2
has no chat turns, so we use a copy frame

    Write "{carrier}" {instruction} "{carrier}

and read the lens over the SECOND (copied) occurrence of the carrier -- positions whose surface
token is the carrier, not X.  The instruction sits between the shown sentence and the copy.
Baseline = no instruction (and no mention of X): `Write "{carrier}" "{carrier}`.

Conditions (the paper's, via the reference data's `phrasings` / `group_kind`):
    baseline   no instruction, X absent from the prompt        the ~0 anchor (Fig 10)
    mention    X named, no instruction (control)               isolates priming (App Fig 65)
    focus      "Think about X while you write" (focus)         instructed activation
    dismissal  "Ignore X" / "X is irrelevant" (suppress)       downward control ("ignore")
    negated    "Don't think about X" (suppress)                white-bear (App Fig 65)

Primary target = the concept word itself, named and tracked (App Fig 65 framing).  The main-text
Fig 10 form -- name the *category*, track its unnamed *members* -- is near-null at 124M (the
members never appear in the prompt); we quantify that separately.  See PROTOCOL.md.
"""
from __future__ import annotations

import json

DM = json.load(open("/tiny-jlens/ref/jacobian-lens/data/experiments/directed-modulation.json"))
TD = json.load(open("/tiny-jlens/ref/jacobian-lens/data/experiments/top-down-summoning.json"))

CARRIERS: list[str] = DM["carrier_sentences"]          # 20 unrelated sentences to copy
TOPICS: list[dict] = DM["topic_categories"]            # {name, members}
MATH: list[dict] = DM["math_problems"]                 # {expr, answer, tier}
GROUP_KIND: dict = DM["group_kind"]                    # group -> {focus, control, suppress}

# instruction phrasings, regrouped into the five report conditions.  `mention` is group_kind
# 'control'; `dismissal` and `negated` are the two 'suppress' groups the paper separates in App
# Fig 65 (dismissal = "ignore/irrelevant", negated-think = "don't think about").
COND_GROUPS = {
    "focus": ["focus"],
    "mention": ["mention"],
    "dismissal": ["dismissal"],
    "negated": ["negated-think"],
}
PHRASINGS: dict[str, list[str]] = {
    cond: [p["text"] for p in DM["phrasings"] if p["group"] in groups]
    for cond, groups in COND_GROUPS.items()
}
CONDITIONS = ["baseline", "mention", "focus", "dismissal", "negated"]
KIND = {"baseline": "baseline", "mention": "control", "focus": "focus",
        "dismissal": "suppress", "negated": "suppress"}


# --------------------------------------------------------------------- concept pool (2a)
def concept_words(lm) -> list[str]:
    """Single-token lowercase concept words drawn from the reference topic members.
    A word is kept if ` word` is one GPT-2 token (the lens has one direction per token)."""
    seen, out = set(), []
    for c in TOPICS:
        for w in c["members"]:
            if w.isalpha() and w[0].islower() and w not in seen and lm.is_single(" " + w):
                seen.add(w)
                out.append(w)
    return out


def word_targets(lm, w: str) -> list[int]:
    """Token ids to track for concept `w`: the ` word` and ` Word` single-token forms."""
    out = [lm.tid(" " + w)]
    if lm.is_single(" " + w.capitalize()):
        out.append(lm.tid(" " + w.capitalize()))
    return sorted(set(out))


# --------------------------------------------------------------------- the copy frame
def copy_frame(lm, carrier: str, instruction: str | None, x: str):
    """(ids [1,T], copy_span list[int]) for `Write "{carrier}" {instruction} "{carrier}`.
    `instruction` is a phrasing template with an {x} slot, or None for the baseline."""
    pre = f'Write "{carrier}" ' + (f"{instruction.format(x=x)} " if instruction else "") + '"'
    import torch
    pre_ids = lm.tok(pre, add_special_tokens=False).input_ids
    copy_ids = lm.tok(carrier, add_special_tokens=False).input_ids
    ids = torch.tensor([[lm.bos] + pre_ids + copy_ids], device=lm.device)
    span = list(range(1 + len(pre_ids), 1 + len(pre_ids) + len(copy_ids)))
    return ids, span


# --------------------------------------------------------------------- imagine materials (2d)
# Paired English/French sentences (the claim asserts the English one is French) and held-out
# passages for the French-vs-English property probe (kept separate from the test sentences).
IMAGINE = {
    "en": [
        "The kitchen window was open all afternoon.",
        "A gray cat slept on the warm stone steps.",
        "The letters were stacked neatly on the desk.",
        "Rain kept falling on the empty market square.",
        "The last train left the station before dark.",
        "Her brother fixed the fence behind the barn.",
        "The garden gate stayed open through the night.",
        "Her sister bought fresh bread at the corner shop.",
        "The children walked to school along the canal.",
        "A small lamp burned on the kitchen table.",
        "Snow covered the rooftops early in December.",
        "The old clock in the hallway stopped at noon.",
    ],
    "fr": [
        "La fenetre de la cuisine est restee ouverte tout l'apres-midi.",
        "Un chat gris dormait sur les marches de pierre chaude.",
        "Les lettres etaient posees sur le bureau pres de la porte.",
        "La pluie tombait sur la place vide du marche.",
        "Le dernier train a quitte la gare avant la nuit.",
        "Son frere a repare la cloture derriere la grange.",
        "La porte du jardin est restee ouverte toute la nuit.",
        "Sa soeur a achete du pain frais a l'epicerie du coin.",
        "Les enfants allaient a l'ecole le long du canal.",
        "Une petite lampe brulait sur la table de la cuisine.",
        "La neige couvrait les toits des le debut de decembre.",
        "La vieille horloge du couloir s'est arretee a midi.",
    ],
    # held-out probe passages (longer, comma-ended; distinct from the test sentences above)
    "fr_probe": [
        "Le marche ouvrait a sept heures, et les marchands installaient leurs etals sous les arches,",
        "Apres le diner les enfants jouaient dans la cour pendant que leur grand-mere preparait le cafe,",
        "Le train traversait la campagne sous un ciel gris pendant que Claire regardait les champs,",
        "Chaque dimanche le vieux professeur marchait au bord de la riviere avec son chien,",
        "La bibliotheque etait presque vide a cette heure, et une etudiante rangeait ses livres,",
        "Le boulanger allumait les fours, sortait les plateaux, et saluait les premiers clients,",
    ],
    "en_probe": [
        "The market opened at seven, and the vendors set up their stalls under the arches,",
        "After dinner the children played in the courtyard while their grandmother made coffee,",
        "The train crossed the countryside under a gray sky while Claire watched the fields,",
        "Every Sunday the old professor walked to the riverbank with his dog and his newspaper,",
        "The library was nearly empty at that hour, and a student was packing up her books,",
        "The baker lit the ovens, set out the trays, and greeted the first customers,",
    ],
}
IMAGINE_HEADERS = {
    "neutral": ["Here is a sentence: {s}", "Sentence: {s}", "Read the following sentence: {s}"],
    "claim": ["Imagine that the following sentence is written in French: {s}",
              "The following sentence is written in French: {s}",
              "This next sentence is in French: {s}"],
}
IMAGINE_LABEL = "French"     # tracked lens token (` French` / ` french`)
