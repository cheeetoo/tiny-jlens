"""C2 privilege, property form (the paper's own design, ported): a header
CLAIMS an English sentence is French. The claim should raise French in the
J-lens at the sentence tokens, while leaving a J-orthogonalized
French-vs-English property probe unmoved; a real French sentence moves the
probe. (Word-level materials cannot make this dissociation — a mentioned
word contaminates every channel; see 41.) A second, structurally different
property category (past tense) runs the same design at the end.

Run:  python experiments/42_imagine.py [model]
"""

import json
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import torch

import core
import pools

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
kit = core.Kit(MODEL)
tok = kit.tokenizer
LAYS = [l for l in kit.layers if 0.55 * kit.n_layers <= l <= 1.0 * kit.n_layers]

ENGLISH = [
    "The kitchen window was open all afternoon.",
    "A gray cat slept on the warm stone steps.",
    "The letters were stacked neatly on the desk.",
    "Rain kept falling on the empty market square.",
    "The last train left the station before dark.",
    "Her brother fixed the fence behind the barn.",
    # expansion pairs (2026-08-19)
    "The garden gate stayed open through the night.",
    "Her sister bought fresh bread at the corner shop.",
    "The children walked to school along the canal.",
    "The doctor's office was closed on Thursday afternoon.",
    "A small lamp burned on the kitchen table.",
    "The neighbors painted their fence green last spring.",
    "Snow covered the rooftops early in December.",
    "The old clock in the hallway stopped at noon.",
    "He left his umbrella on the morning train.",
    "The museum stays open late on Friday evenings.",
    "Their cat slept in the sun by the window.",
    "The baker starts work long before sunrise.",
    # third batch (2026-08-19 late)
    "The keys were hanging by the front door.",
    "The market closes early on Sundays.",
    "A cold wind swept across the empty beach.",
    "The students handed in their essays on Monday.",
    "The bridge was closed for repairs all summer.",
    "She poured the tea into two small cups.",
    "The train conductor checked every ticket.",
    "The bakery smells of warm bread in the morning.",
    "The dog waited patiently outside the shop.",
    "The mountains were still covered in snow in May.",
    "He repaired the roof before the storm arrived.",
    "The lamp on the desk flickered twice and went out.",
]
FRENCH = [
    "La fenêtre de la cuisine est restée ouverte tout l'après-midi.",
    "Un chat gris dormait sur les marches de pierre chaude.",
    "Les lettres étaient posées sur le bureau près de la porte.",
    "La pluie tombait sur la place vide du marché.",
    "Le dernier train a quitté la gare avant la nuit.",
    "Son frère a réparé la clôture derrière la grange.",
    # expansion pairs (2026-08-19)
    "La porte du jardin est restée ouverte toute la nuit.",
    "Sa sœur a acheté du pain frais à l'épicerie du coin.",
    "Les enfants allaient à l'école le long du canal.",
    "Le cabinet du médecin était fermé le jeudi après-midi.",
    "Une petite lampe brûlait sur la table de la cuisine.",
    "Les voisins ont peint leur clôture en vert au printemps dernier.",
    "La neige couvrait les toits dès le début de décembre.",
    "La vieille horloge du couloir s'est arrêtée à midi.",
    "Il a oublié son parapluie dans le train du matin.",
    "Le musée reste ouvert tard le vendredi soir.",
    "Leur chat dormait au soleil près de la fenêtre.",
    "Le boulanger commence à travailler bien avant le lever du soleil.",
    # third batch (2026-08-19 late)
    "Les clés étaient accrochées près de la porte d'entrée.",
    "Le marché ferme tôt le dimanche.",
    "Un vent froid balayait la plage déserte.",
    "Les étudiants ont rendu leurs dissertations lundi.",
    "Le pont était fermé pour travaux tout l'été.",
    "Elle a versé le thé dans deux petites tasses.",
    "Le contrôleur a vérifié tous les billets.",
    "La boulangerie sent le pain chaud le matin.",
    "Le chien attendait patiemment devant la boutique.",
    "Les montagnes étaient encore couvertes de neige en mai.",
    "Il a réparé le toit avant l'arrivée de la tempête.",
    "La lampe du bureau a clignoté deux fois puis s'est éteinte.",
]
# Headers: deltas are computed against the mean over neutral headers. The
# canonical claim (index 0) is the bare noun-phrase header — it loads
# 'French' in the lens harder than genuinely French text does while leaving
# the orthogonal property probe flat.
NEUTRALS = ["Here is a sentence: {s}",
            "Sentence: {s}",
            "Read the following sentence: {s}"]
CLAIMS = ["French sentence: {s}",
          "The following sentence is written in French: {s}",
          "This next sentence is in French: {s}",
          "Imagine that the following sentence is written in French: {s}"]

# ---- property probe: French vs English, from held-out C5a passages ----
fr_pass = [p["text"] for p in pools.c5_passages() + pools.EXTRA_PASSAGES
           if p["category"] == "French"]
en_pass = ["The old man walked through the forest every morning while the birds sang in the trees,",
           "The city woke slowly under the morning sun as people crossed the square with baskets of bread,",
           "The sea was calm that morning, and the fishing boats returned to the harbor one by one,",
           # expansion (balance against the widened French passage pool)
           "The market opened at seven, and the vendors set up their stalls under the arches while the smell of warm bread drifted down the street,",
           "After dinner the children played in the courtyard while their grandmother made coffee, and night fell slowly over the village,",
           "The train crossed the countryside under a gray sky while Claire watched the fields slide past the window,",
           "Every Sunday the old professor walked to the riverbank with his dog, greeted the fishermen, and bought his newspaper,",
           "The library was nearly empty at that hour, and a student was packing up her books by the window as the rain began to fall,"]


def mean_final(texts):
    out = {l: [] for l in LAYS}
    for t in texts:
        r = kit.residuals(kit.encode(t), LAYS)
        for l in LAYS:
            out[l].append(r[l][-1])
    return {l: torch.stack(v).mean(0) for l, v in out.items()}


mf, me = mean_final(fr_pass), mean_final(en_pass)
probe, probe_orth = {}, {}
for l in LAYS:
    p = mf[l] - me[l]
    _, _, recon = core.gradient_pursuit(kit, p, l, 16, centered=True)
    probe[l] = p / p.norm()
    q = p - recon
    probe_orth[l] = q / q.norm()
    print(f"L{l}: French-probe J-share {recon.norm()**2 / p.norm()**2:.1%}")

fr_ids = [kit.tok_id(" French"), kit.tok_id("French")]


def measures(text: str, sentence: str):
    start = text.index(sentence)
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"].cuda()
    span = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
            if a >= start and b > a]
    resid = kit.residuals(ids, LAYS)
    lens_lp = max(
        torch.logsumexp(kit.lens_logits(resid[l][span], l).log_softmax(-1)[:, fr_ids],
                        dim=-1).mean().item() for l in LAYS)
    orth = sum(float((resid[l][span] @ probe_orth[l]).mean()) for l in LAYS) / len(LAYS)
    full = sum(float((resid[l][span] @ probe[l]).mean()) for l in LAYS) / len(LAYS)
    return lens_lp, orth, full


rows = []
for s_en, s_fr in zip(ENGLISH, FRENCH):
    rows.append(dict(
        neutral=[measures(t.format(s=s_en), s_en) for t in NEUTRALS],
        claim=[measures(t.format(s=s_en), s_en) for t in CLAIMS],
        real=[measures(t.format(s=s_fr), s_fr) for t in NEUTRALS]))
    print(".", end="", flush=True)
print()


def base(r, i):
    return sum(m[i] for m in r["neutral"]) / len(r["neutral"])


def paired_z(cond, vi, i):
    d = torch.tensor([r[cond][vi][i] - base(r, i) for r in rows])
    return (d.mean() / (d.std() / len(rows) ** 0.5)).item()


print(f"\nn={len(rows)} sentences; paired t vs mean-of-neutral-headers "
      f"(mean/(sd/sqrt n) — NOT a baseline-SD z; mean-shift ratios are the "
      f"honest magnitude readout, see analyze.py) "
      f"(lens 'French' | J-orth property probe | full probe):")
for cond, n_var in (("claim", len(CLAIMS)), ("real", len(NEUTRALS))):
    for vi in range(n_var):
        print(f"  {cond}[{vi}] lens t {paired_z(cond, vi, 0):+6.1f}   "
              f"orth-probe t {paired_z(cond, vi, 1):+6.1f}   "
              f"full-probe t {paired_z(cond, vi, 2):+6.1f}")
json.dump(rows, open(f"/tiny-jlens/gpt2/results/c2_imagine_{MODEL}.json", "w"))

# ---- second property category: past tense ----
# Same design, structurally different property: base = present-tense
# sentence, real = its past-tense form, claim = a header asserting the
# past tense over the present-tense sentence. Caveat: the real lens delta
# for ' past' is small in absolute terms (the label is a weaker verbal
# habit than a language name), so raw deltas are saved alongside ratios.

PRESENT = [
    "The kitchen window stays open all afternoon.",
    "A gray cat sleeps on the warm stone steps.",
    "The letters sit neatly stacked on the desk.",
    "Rain keeps falling on the empty market square.",
    "The last train leaves the station before dark.",
    "Her brother fixes the fence behind the barn.",
    "The garden gate stays open through the night.",
    "Her sister buys fresh bread at the corner shop.",
    "The children walk to school along the canal.",
    "The doctor's office closes on Thursday afternoons.",
    "A small lamp burns on the kitchen table.",
    "The neighbors paint their fence green every spring.",
]
PAST = [
    "The kitchen window stayed open all afternoon.",
    "A gray cat slept on the warm stone steps.",
    "The letters sat neatly stacked on the desk.",
    "Rain kept falling on the empty market square.",
    "The last train left the station before dark.",
    "Her brother fixed the fence behind the barn.",
    "The garden gate stayed open through the night.",
    "Her sister bought fresh bread at the corner shop.",
    "The children walked to school along the canal.",
    "The doctor's office closed on Thursday afternoons.",
    "A small lamp burned on the kitchen table.",
    "The neighbors painted their fence green every spring.",
]
# probe passages: past vs present narrations of the same scenes
PRES_PASS = [
    "Every morning the baker lights the ovens, sets out the trays, and greets the first customers as the street slowly fills,",
    "The ferry crosses the bay twice a day, and the gulls follow it while passengers watch the waves from the deck,",
    "On Sundays the family walks to the orchard, picks a basket of apples, and eats lunch under the old tree,",
    "The night guard checks every door, notes the time in his ledger, and listens to the rain on the roof,",
    "Each spring the gardeners plant new rows of tulips while the fountain runs and children chase each other on the grass,",
    "The librarian sorts the returned books, stamps the cards, and waters the plants by the tall windows,",
]
PAST_PASS = [
    "Every morning the baker lit the ovens, set out the trays, and greeted the first customers as the street slowly filled,",
    "The ferry crossed the bay twice a day, and the gulls followed it while passengers watched the waves from the deck,",
    "On Sundays the family walked to the orchard, picked a basket of apples, and ate lunch under the old tree,",
    "The night guard checked every door, noted the time in his ledger, and listened to the rain on the roof,",
    "Each spring the gardeners planted new rows of tulips while the fountain ran and children chased each other on the grass,",
    "The librarian sorted the returned books, stamped the cards, and watered the plants by the tall windows,",
]
TENSE_CLAIMS = ["The following sentence is written in the past tense: {s}",
                "The next sentence is in the past tense: {s}",
                "Note: the sentence below is written in the past tense. {s}"]

mp, mn = mean_final(PAST_PASS), mean_final(PRES_PASS)
t_probe, t_probe_orth = {}, {}
for l in LAYS:
    p = mp[l] - mn[l]
    _, _, recon = core.gradient_pursuit(kit, p, l, 16, centered=True)
    t_probe[l] = p / p.norm()
    q = p - recon
    t_probe_orth[l] = q / q.norm()
past_ids = [kit.tok_id(" past"), kit.tok_id("past")]


def t_measures(text: str, sentence: str):
    start = text.index(sentence)
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"].cuda()
    span = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
            if a >= start and b > a]
    resid = kit.residuals(ids, LAYS)
    lens_lp = max(
        torch.logsumexp(kit.lens_logits(resid[l][span], l).log_softmax(-1)[:, past_ids],
                        dim=-1).mean().item() for l in LAYS)
    orth = sum(float((resid[l][span] @ t_probe_orth[l]).mean()) for l in LAYS) / len(LAYS)
    full = sum(float((resid[l][span] @ t_probe[l]).mean()) for l in LAYS) / len(LAYS)
    return lens_lp, orth, full


t_rows = []
for s_pres, s_past in zip(PRESENT, PAST):
    t_rows.append(dict(
        neutral=[t_measures(t.format(s=s_pres), s_pres) for t in NEUTRALS],
        claim=[t_measures(t.format(s=s_pres), s_pres) for t in TENSE_CLAIMS],
        real=[t_measures(t.format(s=s_past), s_past) for t in NEUTRALS]))
    print(".", end="", flush=True)
print()

n = len(t_rows)
t_base = lambda r, i: sum(m[i] for m in r["neutral"]) / len(r["neutral"])
real_d = [sum(sum(m[i] for m in r["real"]) / len(r["real"]) - t_base(r, i)
              for r in t_rows) / n for i in range(2)]
print(f"\npast tense (n={n}; deltas vs mean-neutral, % of real-past delta; "
      f"real lens delta raw {real_d[0]:+.2f}):")
for cond, headers in (("claim", TENSE_CLAIMS), ("real", NEUTRALS)):
    for vi in range(len(headers)):
        parts = []
        for i, name in [(0, "lens'past'"), (1, "J-orth probe")]:
            dv = sum(r[cond][vi][i] - t_base(r, i) for r in t_rows) / n
            w = sum(r[cond][vi][i] > t_base(r, i) for r in t_rows)
            parts.append(f"{name} {100*dv/real_d[i]:+.0f}% (up {w}/{n})")
        print(f"  {cond}[{vi}] " + "   ".join(parts))
json.dump(t_rows, open(f"/tiny-jlens/gpt2/results/c2_imagine_tense_{MODEL}.json", "w"))
