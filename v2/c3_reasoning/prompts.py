"""Prompt material for criterion 3 (internal reasoning): two-hop factual chains
with a known *unspoken* intermediate.

The paper's §3.3 uses two-hop prompts whose answer depends on first inferring an
unspoken bridge entity ("the animal that spins webs" -> spider -> eight legs).
GPT-2-small cannot resolve the paper's riddle phrasings at all (0/6 on the
spider/Mars riddles; 7/90 on the released probe-swap.json), so — exactly as
criterion 1 wrapped "Name a {cat}:" in a few-shot list — we present each two-hop
query in a few-shot frame that teaches the *relation* while leaving the specific
bridge entity to be computed.

Two relation families, inverses of each other through the country:
  lang_capital :  language -> (country) -> capital     "...speak {L}, the capital city is called ___"
  cap_language :  capital  -> (country) -> language     "...governed from {C} ... language, namely ___"
The intermediate is always the COUNTRY, which never appears in the prompt or the
answer.  The few-shot shots use a fixed set of demo countries; any test item whose
country or answer collides with the shot text is dropped (the answer must not be
echoable from the prompt).

For the privileging experiment (§3.3, Fig 16) we also build a probe for each
country from several cue prompts that *imply* the country through different
surface cues and ask about different attributes, with the country name never the
next token (paper: "prompts that imply the same intermediate through different
surface cues and ask different questions about it").

The paper's own two-hop set (ref/.../probe-swap.json) is loaded verbatim for the
capability-floor check in run.py.
"""
from __future__ import annotations

# country -> (language, capital, continent, language_unique)
# world knowledge only; single-token capitals/answers are enforced at build time
# in run.py, and capability + the shot-collision guard decide what survives.
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

# Two-hop families: (name, arg field, answer field, few-shot template).
# Shots use Egypt/Israel (lang_capital) and Egypt/Japan (cap_language); the
# shot-collision guard in run.py drops any item colliding with the shot text.
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

# §3.3 privileging: cues that IMPLY the country through varied surface cues and
# ask about a DIFFERENT attribute, so the country name is never the next token.
# Filled per country from its capital / language / continent.
PROBE_CUES = [
    "People in the country whose capital is {capital} mostly speak the language of",
    "A traveler flying into {capital} has landed on the continent of",
    "The homeland of the {language} language lies on the continent of",
    "Newspapers printed in {capital} are usually written in the language of",
    "The country whose largest city is {capital} is located on the continent of",
    "Someone who grew up speaking {language} was most likely born in the city of",
]

CASE_STUDY = dict(family="lang_capital", source="France", target="China")  # Paris -> Beijing
