"""Held-out item pools for CONFIRMATORY runs (BRIEF R3).

These items must not be used during exploration. They extend each experiment's
item family without changing its logical form.
"""

# --- C5a: fresh language passages (same 4 languages as exploration set) ---
CONFIRM_PASSAGES = [
    {"category": "French", "key": "fr_c1", "text": "La pluie tombait doucement sur les toits de la ville. Dans la cuisine, grand-mère préparait une soupe aux légumes pendant que"},
    {"category": "French", "key": "fr_c2", "text": "Chaque été, nous allions chez mon oncle à la campagne. Il avait un grand jardin plein de tomates et de fraises, et"},
    {"category": "French", "key": "fr_c3", "text": "Le train partait à huit heures du matin. Sur le quai, les voyageurs attendaient avec leurs valises et leurs journaux, et"},
    {"category": "German", "key": "de_c1", "text": "Der Winter kam früh in diesem Jahr. Schnee bedeckte die Dächer des kleinen Dorfes, und die Kinder spielten"},
    {"category": "German", "key": "de_c2", "text": "Jeden Morgen fuhr Herr Müller mit dem Fahrrad zur Arbeit. Auf dem Weg kaufte er eine Zeitung und ein frisches Brötchen, denn"},
    {"category": "German", "key": "de_c3", "text": "Die Bibliothek war still und warm. Anna saß am Fenster und las ein altes Buch über die Geschichte der Stadt, während"},
    {"category": "Spanish", "key": "es_c1", "text": "El mercado estaba lleno de colores y de voces. Las señoras compraban naranjas y tomates mientras los niños corrían entre"},
    {"category": "Spanish", "key": "es_c2", "text": "Cada domingo, la familia se reunía en la casa de la abuela. Ella preparaba paella y todos hablaban de"},
    {"category": "Spanish", "key": "es_c3", "text": "El sol brillaba sobre el pueblo blanco. Los pescadores volvían del mar con sus barcas llenas, y en la plaza"},
    {"category": "Italian", "key": "it_c1", "text": "La mattina era fresca e chiara. Il fornaio apriva la sua bottega e il profumo del pane riempiva la strada, mentre"},
    {"category": "Italian", "key": "it_c2", "text": "Ogni estate andavamo al mare con i nonni. La spiaggia era piena di ombrelloni colorati e i bambini costruivano"},
    {"category": "Italian", "key": "it_c3", "text": "Il treno attraversava le colline verdi della Toscana. Dal finestrino si vedevano i cipressi e le vecchie case di pietra, e"},
]

# --- C5b / battery: fresh sentiment items ---
CONFIRM_SENTIMENT = [
    ("What a wonderful surprise, I enjoyed every second of it.", "positive"),
    ("Utterly disappointing, nothing worked as promised.", "negative"),
    ("The staff were friendly and the room was spotless.", "positive"),
    ("I regret buying this, it broke after one day.", "negative"),
    ("A masterpiece, moving and unforgettable.", "positive"),
    ("The plot made no sense and the ending was insulting.", "negative"),
    ("Delicious food and a lovely atmosphere.", "positive"),
    ("Cold coffee, rude waiter, never again.", "negative"),
    ("This book changed how I see the world, truly brilliant.", "positive"),
    ("A dull, lifeless slog from beginning to end.", "negative"),
]

# --- C5b / battery: fresh analogies ---
CONFIRM_ANALOGIES = [
    ("hot is to cold as wet is to", "dry"),
    ("teacher is to school as doctor is to", "hospital"),
    ("bird is to nest as bee is to", "hive"),
    ("young is to old as new is to", "old"),
    ("eye is to see as ear is to", "hear"),
    ("start is to finish as begin is to", "end"),
    ("north is to south as east is to", "west"),
    ("rich is to poor as strong is to", "weak"),
]

# --- C3: fresh two-hop items (same families as twohop_pool) ---
CONFIRM_TWOHOP = [
    dict(family="lang-capital", prompt="Fact: The capital of the country where Czech is the primary language is",
         intermediate="Czechia", answer="Prague"),
    dict(family="lang-capital", prompt="Fact: The capital of the country where Romanian is the primary language is",
         intermediate="Romania", answer="Bucharest"),
    dict(family="lang-capital", prompt="Fact: The capital of the country where Ukrainian is the primary language is",
         intermediate="Ukraine", answer="Kyiv"),
    dict(family="lang-capital", prompt="Fact: The capital of the country where Irish is the primary language is",
         intermediate="Ireland", answer="Dublin"),
    dict(family="lang-capital", prompt="Fact: The capital of the country where Indonesian is the primary language is",
         intermediate="Indonesia", answer="Jakarta"),
    dict(family="city-language", prompt="Fact: The language spoken in the country where Prague is located is",
         intermediate="Czechia", answer="Czech"),
    dict(family="city-language", prompt="Fact: The language spoken in the country where Helsinki is located is",
         intermediate="Finland", answer="Finnish"),
    dict(family="city-language", prompt="Fact: The language spoken in the country where Oslo is located is",
         intermediate="Norway", answer="Norwegian"),
    dict(family="city-language", prompt="Fact: The language spoken in the country where Copenhagen is located is",
         intermediate="Denmark", answer="Danish"),
    dict(family="city-language", prompt="Fact: The language spoken in the country where Ankara is located is",
         intermediate="Turkey", answer="Turkish"),
]

# --- C1: confirmation uses a fresh random seed over the same category lists
# (targets are sampled, so freshness comes from the seed) plus these extra
# categories with common single-token instances ---
CONFIRM_CATEGORIES = {
    "vegetable": ["carrot", "potato", "onion", "tomato", "pepper", "cabbage",
                  "spinach", "broccoli", "corn", "bean", "pea", "lettuce"],
    "animal": ["dog", "cat", "horse", "lion", "tiger", "bear", "wolf", "fox",
               "rabbit", "mouse", "elephant", "monkey"],
    "metal": ["gold", "silver", "iron", "copper", "steel", "tin", "zinc",
              "lead", "aluminum", "brass", "nickel", "platinum"],
    "flower": ["rose", "tulip", "daisy", "lily", "orchid", "sunflower",
               "violet", "poppy", "iris", "daffodil"],
}

# --- C2: held-out carriers (confirmation) ---
CONFIRM_CARRIERS = [
    "The little boat drifted slowly across the quiet lake.",
    "A gentle breeze moved the curtains in the empty room.",
    "The clock on the tower struck nine as people hurried past.",
    "Fresh snow covered the path leading up to the cabin.",
]
