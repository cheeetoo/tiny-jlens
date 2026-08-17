"""Custom two-hop item pool for tiny models (C3), extending the paper's
probe-swap families with more items of the same logical form: a prompt whose
correct completion requires an unspoken intermediate entity.

Each item: prompt (raw "Fact:" completion), intermediate (unspoken entity),
answer, plus category. Swap pairs are formed within-category at runtime.
`explicit_template` lets us also measure the model's knowledge of the second
hop directly (diagnostic only, BRIEF C3 note).
"""

LANG_CAPITAL = [
    # (language, country=intermediate, capital=answer)
    ("French", "France", "Paris"),
    ("German", "Germany", "Berlin"),
    ("Spanish", "Spain", "Madrid"),
    ("Italian", "Italy", "Rome"),
    ("Russian", "Russia", "Moscow"),
    ("Japanese", "Japan", "Tokyo"),
    ("Chinese", "China", "Beijing"),
    ("Polish", "Poland", "Warsaw"),
    ("Swedish", "Sweden", "Stockholm"),
    ("Greek", "Greece", "Athens"),
    ("Hungarian", "Hungary", "Budapest"),
    ("Dutch", "Netherlands", "Amsterdam"),
    ("Portuguese", "Portugal", "Lisbon"),
    ("Turkish", "Turkey", "Ankara"),
    ("Korean", "Korea", "Seoul"),
    ("Thai", "Thailand", "Bangkok"),
    ("Vietnamese", "Vietnam", "Hanoi"),
    ("Finnish", "Finland", "Helsinki"),
    ("Norwegian", "Norway", "Oslo"),
    ("Danish", "Denmark", "Copenhagen"),
    ("Hindi", "India", "Delhi"),
    ("English", "England", "London"),
]

CITY_LANGUAGE = [
    # (city, country=intermediate, language=answer)
    ("Moscow", "Russia", "Russian"),
    ("Lyon", "France", "French"),
    ("Cairo", "Egypt", "Arabic"),
    ("Tokyo", "Japan", "Japanese"),
    ("Berlin", "Germany", "German"),
    ("Madrid", "Spain", "Spanish"),
    ("Rome", "Italy", "Italian"),
    ("Warsaw", "Poland", "Polish"),
    ("Athens", "Greece", "Greek"),
    ("Beijing", "China", "Chinese"),
    ("Amsterdam", "Netherlands", "Dutch"),
    ("Stockholm", "Sweden", "Swedish"),
    ("Seoul", "Korea", "Korean"),
    ("Bangkok", "Thailand", "Thai"),
    ("Hanoi", "Vietnam", "Vietnamese"),
    ("Lisbon", "Portugal", "Portuguese"),
]

LANDMARK_COUNTRY = [
    # (landmark, city=intermediate, country=answer)
    ("the Eiffel Tower", "Paris", "France"),
    ("the Colosseum", "Rome", "Italy"),
    ("Big Ben", "London", "England"),
    ("the Kremlin", "Moscow", "Russia"),
    ("the Statue of Liberty", "New York", "America"),
    ("the Acropolis", "Athens", "Greece"),
    ("the Sagrada Familia", "Barcelona", "Spain"),
    ("Christ the Redeemer", "Rio", "Brazil"),
]

FOOD_ANIMAL = [
    # (food, animal=answer) -- intermediate IS the answer's source; here the
    # unspoken intermediate is the animal category; single-hop-ish, kept as an
    # easy family: prompt implies the animal via its product.
    ("butter", "cow"),
    ("honey", "bee"),
    ("wool", "sheep"),
    ("milk", "cow"),
    ("bacon", "pig"),
]


def build_items():
    items = []
    for lang, country, capital in LANG_CAPITAL:
        items.append(dict(
            family="lang-capital",
            prompt=f"Fact: The capital of the country where {lang} is the primary language is",
            intermediate=country, answer=capital,
            explicit_template="Fact: The capital of {x} is the city of",
        ))
    for city, country, language in CITY_LANGUAGE:
        items.append(dict(
            family="city-language",
            prompt=f"Fact: The language spoken in the country where {city} is located is",
            intermediate=country, answer=language,
            explicit_template="Fact: The language spoken in {x} is",
        ))
    for landmark, city, country in LANDMARK_COUNTRY:
        items.append(dict(
            family="landmark-country",
            prompt=f"Fact: The country you are in when you visit {landmark} is",
            intermediate=city, answer=country,
            explicit_template="Fact: The country containing the city of {x} is",
        ))
    return items
