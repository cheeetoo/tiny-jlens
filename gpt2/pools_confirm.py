"""HELD-OUT materials for confirmatory runs. Nothing here was used in any
exploration run (day 2's held-out sets were burned in old/ confirmatory runs
and are excluded too). The capability filter is applied to these items
before and independently of any lens measurement, per BRIEF §3.

Frozen together with CONFIRMED.md before the confirmatory suite runs.
"""

# ---- C3/C4: fresh countries (none appear in pools.COUNTRIES) ----
NEW_COUNTRIES = {
    "Ukraine":   dict(language="Ukrainian",  capital="Kiev",      continent="Europe", language_unique=True),
    "Romania":   dict(language="Romanian",   capital="Bucharest", continent="Europe", language_unique=True),
    "Bulgaria":  dict(language="Bulgarian",  capital="Sofia",     continent="Europe", language_unique=True),
    "Serbia":    dict(language="Serbian",    capital="Belgrade",  continent="Europe", language_unique=True),
    "Croatia":   dict(language="Croatian",   capital="Zagreb",    continent="Europe", language_unique=True),
    "Indonesia": dict(language="Indonesian", capital="Jakarta",   continent="Asia",   language_unique=True),
    "Malaysia":  dict(language="Malay",      capital="Kuala",     continent="Asia",   language_unique=True),
    "Pakistan":  dict(language="Urdu",       capital="Islamabad", continent="Asia",   language_unique=True),
    "Nigeria":   dict(language="Yoruba",     capital="Abuja",     continent="Africa", language_unique=True),
    "Ethiopia":  dict(language="Amharic",    capital="Addis",     continent="Africa", language_unique=True),
    "Morocco":   dict(language="Arabic",     capital="Rabat",     continent="Africa", language_unique=False),
    "Cuba":      dict(language="Spanish",    capital="Havana",    continent="America", language_unique=False),
    "Chile":     dict(language="Spanish",    capital="Santiago",  continent="America", language_unique=False),
    "Peru":      dict(language="Spanish",    capital="Lima",      continent="America", language_unique=False),
    "Colombia":  dict(language="Spanish",    capital="Bogota",    continent="America", language_unique=False),
    "Iraq":      dict(language="Arabic",     capital="Baghdad",   continent="Asia",   language_unique=False),
    "Jordan":    dict(language="Arabic",     capital="Amman",     continent="Asia",   language_unique=False),
    "Lebanon":   dict(language="Arabic",     capital="Beirut",    continent="Asia",   language_unique=False),
    "Afghanistan": dict(language="Pashto",   capital="Kabul",     continent="Asia",   language_unique=True),
    "Cambodia":  dict(language="Khmer",      capital="Phnom",     continent="Asia",   language_unique=True),
    "Mongolia":  dict(language="Mongolian",  capital="Ulan",      continent="Asia",   language_unique=True),
    "Armenia":   dict(language="Armenian",   capital="Yerevan",   continent="Asia",   language_unique=True),
    "Georgia":   dict(language="Georgian",   capital="Tbilisi",   continent="Asia",   language_unique=True),
    "Albania":   dict(language="Albanian",   capital="Tirana",    continent="Europe", language_unique=True),
    "Slovakia":  dict(language="Slovak",     capital="Bratislava", continent="Europe", language_unique=True),
    "Iceland":   dict(language="Icelandic",  capital="Reykjavik", continent="Europe", language_unique=True),
}

# grading additions: for continent answers, " Southeast" (Asia) is correct
# for these countries (pre-declared after the exploration near-miss)
SOUTHEAST_OK = {"Thailand", "Vietnam", "Cambodia", "Malaysia", "Indonesia"}

# ---- C1: fresh categories (no overlap with eval or shot categories) ----
NEW_CATEGORIES = {
    "vegetable": ["carrot", "potato", "onion", "cabbage", "spinach", "pea",
                  "corn", "bean", "pumpkin", "celery", "lettuce", "broccoli"],
    "insect": ["ant", "bee", "beetle", "butterfly", "moth", "fly", "wasp",
               "cricket", "spider", "mosquito"],
    "gem": ["diamond", "ruby", "emerald", "pearl", "opal", "jade", "amber",
            "sapphire", "topaz", "quartz"],
    "weather condition": ["rain", "snow", "wind", "fog", "storm", "hail",
                          "sunshine", "thunder", "frost", "drizzle"],
    "body part": ["arm", "leg", "hand", "eye", "ear", "nose", "knee",
                  "shoulder", "elbow", "ankle", "wrist", "chin"],
    "fish": ["salmon", "trout", "shark", "cod", "tuna", "bass", "pike",
             "carp", "herring", "eel"],
    "dance": ["waltz", "tango", "salsa", "ballet", "polka", "samba",
              "swing", "jazz"],
    "fabric": ["silk", "cotton", "wool", "linen", "velvet", "denim",
               "leather", "satin", "nylon"],
}

# ---- C2: fresh words and sentences ----
C2_CONFIRM_WORDS = ["anchor", "meadow", "trumpet", "ladder", "pepper",
                    "marble", "falcon", "blanket", "lantern", "ribbon",
                    "barrel", "compass"]
C2_CONFIRM_SENTENCES = [
    "The museum opened its doors at nine every morning.",
    "Two workers painted the fence behind the school.",
    "The letter arrived three days after the storm.",
]

# ---- C1 inject: fresh concepts ----
INJECT_CONFIRM = ["snow", "tigers", "copper", "whales", "candles", "spices",
                  "arrows", "pearls", "smoke", "feathers", "ropes", "cliffs",
                  "engines", "petals", "shadows", "ashes", "banners", "chains",
                  "meadows", "sparks", "lanterns", "rivers", "crowns", "sails"]

# ---- C5a: fresh passages (3 per language) ----
C5_CONFIRM_PASSAGES = [
    dict(category="French", key="cf1", text="Le marché du village ouvrait à huit heures, et les marchands installaient leurs étals de fruits et de fleurs. Camille choisissait toujours les pommes les plus rouges,"),
    dict(category="French", key="cf2", text="La bibliothèque était silencieuse ce soir-là. Antoine tournait les pages d'un vieux livre relié en cuir, cherchant une carte oubliée,"),
    dict(category="French", key="cf3", text="Le train traversait la campagne sous un ciel gris. Par la fenêtre, Sophie regardait défiler les champs et les petits villages,"),
    dict(category="German", key="cg1", text="Am Bahnhof warteten viele Menschen auf den letzten Zug des Tages. Thomas stand mit seinem Koffer am Ende des Bahnsteigs und dachte an die lange Reise,"),
    dict(category="German", key="cg2", text="Die Bäckerei an der Ecke öffnete früh am Morgen, und der Duft von frischem Brot zog durch die Straße. Anna kaufte jeden Tag zwei Brötchen,"),
    dict(category="German", key="cg3", text="Im Garten hinter dem Haus wuchsen Äpfel und Birnen. Der alte Nachbar pflegte die Bäume seit vielen Jahren mit großer Sorgfalt,"),
    dict(category="Spanish", key="cs1", text="El mercado del pueblo abría temprano, y los vendedores colocaban sus frutas bajo el sol de la mañana. Rosa compraba naranjas y pan recién hecho,"),
    dict(category="Spanish", key="cs2", text="La lluvia caía suavemente sobre los tejados de la ciudad vieja. Miguel caminaba despacio bajo su paraguas negro, pensando en la carta,"),
    dict(category="Spanish", key="cs3", text="El barco salía del puerto cada mañana antes del amanecer. Los pescadores preparaban sus redes mientras las gaviotas volaban sobre el agua,"),
    dict(category="Italian", key="ci1", text="La piazza del paese si riempiva di gente ogni domenica mattina. Giovanni beveva il suo caffè al bar guardando i bambini giocare vicino alla fontana,"),
    dict(category="Italian", key="ci2", text="Il treno per Firenze partiva alle sette. Lucia guardava dal finestrino le colline coperte di vigneti e ulivi, pensando alla casa della nonna,"),
    dict(category="Italian", key="ci3", text="Nella cucina della trattoria, il cuoco preparava la pasta fresca come ogni sera. Il profumo di basilico e pomodoro riempiva la sala,"),
]

# ---- C4: held-out argument set (countries not in the exploration grid args) ----
C4_CONFIRM_ARGS = ["Portugal", "Greece", "Turkey", "Norway", "Denmark",
                   "Finland", "Hungary", "Korea", "Thailand", "Russia"]

# ---- 42_imagine: fresh English/French sentence pairs ----
ENGLISH_CONFIRM = [
    "The bakery on the corner sold out of bread by noon.",
    "A small boat drifted slowly across the quiet lake.",
    "The teacher wrote the date in the corner of the board.",
    "Snow covered the roofs of the houses along the river.",
    "The clock in the hallway stopped just after midnight.",
    "Her uncle kept his tools in a box under the stairs.",
]
FRENCH_CONFIRM = [
    "La boulangerie du coin avait vendu tout son pain avant midi.",
    "Un petit bateau dérivait lentement sur le lac tranquille.",
    "Le professeur a écrit la date dans le coin du tableau.",
    "La neige couvrait les toits des maisons le long de la rivière.",
    "L'horloge du couloir s'est arrêtée juste après minuit.",
    "Son oncle gardait ses outils dans une boîte sous l'escalier.",
]
