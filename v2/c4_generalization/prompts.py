"""Prompt material for criterion 4 (flexible generalization): one argument, many
functions, one fixed swap.

The paper's §3.4 constructs, for each *category* of argument (countries, months,
animals, number words), a set of *functions* that each apply a different operation
to the same argument ("the capital of France is", "most people in France speak",
"France is on the continent of", ...).  It then swaps the J-lens vector for the
argument (France -> China) identically across every function and asks whether each
downstream function reads the swapped-in argument correctly.  Systematically:
4 categories x 4 functions x 4 arguments; within a category the four arguments give
12 ordered source->target swap pairs per function, hence 16 functions x 12 = 192
swap trials (paper: 76/192 at alpha=1, 101/192 at alpha=2).

We use the paper's OWN released material for this experiment verbatim ---
`ref/jacobian-lens/data/experiments/flexible-generalization.json` (the categories,
arguments, function templates, and answers).  `load_categories()` returns it.

Deviation (capability floor).  GPT-2-small answers only 9/64 of these bare
templates correctly (see PROTOCOL.md / the `floor` block in run.py), so --- exactly
as criterion 1 wrapped "Name a {cat}:" in a few-shot list and criterion 3 wrapped
its two-hop query in a few-shot frame --- each function's query is preceded by a
2-shot frame that teaches the *function* with two DEMO arguments disjoint from the
four test arguments.  The frame lifts capability to 20/64; the swap and loading
analyses run on top of it.  `FRAMES[(category, function)]` is the demo prefix; the
query is `FRAMES[...] + template.format(arg=test_arg)`.  Prepending the frame also
means the test argument is never sentence-initial, so it always tokenizes with a
leading space (the paper ignores the capitalization-token positions; here they do
not arise).

The demo arguments and their answers are chosen disjoint from the test arguments;
any residual echo (a target answer appearing in the frame) is dropped at scoring
time by the echo guard in run.py.  The paper's bare templates are run verbatim,
with no frame, as the capability `floor` in run.py.
"""
from __future__ import annotations

import json

REF_DATA = "/tiny-jlens/ref/jacobian-lens/data/experiments/flexible-generalization.json"


def load_categories() -> list[dict]:
    """The paper's §3.4 data, verbatim: 4 categories, each with `args` (4) and
    `funcs` (4), each func a dict(name, template, answers={arg: answer})."""
    return json.load(open(REF_DATA))["categories"]


# 2-shot demo prefixes, one per (category, function).  Demo arguments are disjoint
# from the four test arguments of each category; each shot uses the function's own
# template so the frame teaches exactly the function under test.  A trailing space
# joins the frame to the query, keeping the test argument non-initial (leading-space
# token).  These are the ONLY authored text in this criterion; everything else
# (arguments, templates, answers) is the paper's.
FRAMES: dict[tuple[str, str], str] = {
    # countries — demos: Japan, Italy, Australia, Brazil, Sweden (not France/Canada/China/Egypt)
    ("countries", "capital"):   "The capital of Japan is the city of Tokyo. The capital of Italy is the city of Rome. ",
    ("countries", "language"):  "Most people in Japan speak Japanese. Most people in Italy speak Italian. ",
    ("countries", "continent"): "Australia is a country on the continent of Oceania. Brazil is a country on the continent of South. ",
    ("countries", "currency"):  "The single-word name for the currency now used in Japan is the Yen. The single-word name for the currency now used in Sweden is the Krona. ",
    # months — demos: January, June, December, November, March (not February/April/July/October)
    ("months", "season"):       "In the northern hemisphere, January is in the season of winter. In the northern hemisphere, June is in the season of summer. ",
    ("months", "number"):       "Counting from January, March is month number three. Counting from January, June is month number six. ",
    ("months", "holiday"):      "The biggest holiday in December is Christmas. The biggest holiday in November is Thanksgiving. ",
    ("months", "next_month"):   "The month right after January is February. The month right after June is July. ",
    # animals — demos: camel, frog, ant, crab, dog, snake, wolf, fish (not lion/eagle/shark/spider)
    ("animals", "habitat"):     "The natural habitat of a camel is the desert. The natural habitat of a frog is the pond. ",
    ("animals", "legs"):        "How many legs does an ant have? In one word: six. How many legs does a crab have? In one word: ten. ",
    ("animals", "class"):       "Biologically, a dog is a type of mammal. Biologically, a snake is a type of reptile. ",
    ("animals", "group"):       "A group of wolves is called a pack. A group of fish is called a school. ",
    # numbers — demos: one, four, two, six, twelve, zero, eight (not three/five/seven/nine)
    ("numbers", "double"):      "Two times one equals two. Two times four equals eight. ",
    ("numbers", "square"):      "Two squared equals four. Six squared equals thirty. ",
    ("numbers", "successor"):   "The number that comes right after one is two. The number that comes right after twelve is thirteen. ",
    ("numbers", "first_letter"):"The word 'zero' begins with the letter z. The word 'eight' begins with the letter e. ",
}

# Fig 18 case study: one argument (France) read by every country function, under a
# single fixed swap France -> China.
CASE_STUDY = dict(category="countries", source="France", target="China")
