"""Relevance safety rules that work on narration text and clip metadata only.

* ``content_words``   - the narration words a clip should reflect (proper names are left out: stock footage never shows them).
* ``ENTITY_STANDINS`` - named brands/products that are shown through a generic visual stand-in, never a logo or a person.
* ``SENSITIVE``       - sad or serious news. Such scenes get sombre queries and reject clips whose metadata is upbeat.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")

ENTITY_STANDINS: Dict[str, Tuple[str, ...]] = {
    "tesla": ("electric car charging", "electric", "car"),
    "iphone": ("person using smartphone", "smartphone", "phone"),
    "ipad": ("person using tablet", "tablet"),
    "nike": ("athlete running shoes", "running", "shoes"),
    "adidas": ("athlete running shoes", "running", "shoes"),
    "spacex": ("rocket launch", "rocket", "launch"),
    "amazon": ("delivery packages warehouse", "packages", "delivery"),
    "netflix": ("watching tv at home", "television", "watching"),
    "microsoft": ("office laptop work", "laptop", "office"),
    "google": ("search on laptop", "laptop", "search"),
}

# trigger regex -> (sombre queries, topic words that count as subject evidence)
SENSITIVE: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    (r"\bflood\w*\b", ("flooded street aftermath", "heavy rain storm clouds"), ("flood", "flooded", "flooding", "aftermath", "rain", "storm")),
    (r"\b(rescue|rescuers|survivors?)\b", ("rescue team emergency responders", "emergency rescue boat"), ("rescue", "emergency", "responders", "firefighters", "ambulance")),
    (r"\b(earthquake|collapsed|rubble)\b", ("collapsed building rubble",), ("rubble", "collapsed", "earthquake", "ruins", "destruction")),
    (r"\b(wildfire|blaze)\b", ("firefighters wildfire smoke",), ("fire", "wildfire", "smoke", "firefighters", "burning")),
    (r"\b(lost their homes?|homeless|devastat\w*|disaster)\b", ("damaged houses aftermath", "disaster aftermath"), ("damaged", "destroyed", "disaster", "aftermath", "ruins", "destruction")),
    (r"\b(died|death|deadly|killed|funeral|victims?|mourn\w*|tragedy|tragic|passed away)\b", ("candle memorial", "rain on window"), ("candle", "memorial", "rain", "mourning", "grief", "flowers")),
    (r"\b(war|bombing|refugees?)\b", ("empty street at dusk", "refugee camp"), ("refugee", "camp", "empty", "dusk", "ruins", "conflict")),
    (r"\b(still missing|missing (people|persons|children)|grief|grieving|sorrow)\b", ("rain on window", "candle flame dark"), ("rain", "window", "candle", "dark", "grief")),
)

# metadata words that make a clip wrong for a sad or serious scene
UPBEAT_TERMS = frozenset({
    "happy", "smile", "smiling", "laugh", "laughing", "fun", "party", "celebration", "celebrate", "cheerful", "joy",
    "joyful", "relax", "relaxing", "relaxed", "vacation", "holiday", "summer", "picnic", "playing", "dance", "dancing",
    "wedding", "birthday", "festival", "carnival", "cartoon", "cute", "funny", "sunny", "sunshine", "lifestyle",
    "students", "student", "laptop", "freelancer", "friends", "friendship", "success", "winning",
})


def content_words(text: str, stop_words: Iterable[str]) -> List[str]:
    stop = set(stop_words)
    out: List[str] = []
    for i, tok in enumerate(_WORD.findall(text or "")):
        w = re.sub(r"['’]s$", "", tok.lower())
        if len(w) < 4 or w in stop:
            continue
        if i > 0 and tok[0].isupper() and not tok.isupper():
            continue        # mid-sentence proper noun
        out.append(w)
    return list(dict.fromkeys(out))


def entity_standins(text: str) -> Tuple[List[str], List[str]]:
    """-> (queries, words) for brands/products named in the narration."""
    low = set(w.lower() for w in _WORD.findall(text or ""))
    queries: List[str] = []
    words: List[str] = []
    for name, (query, *terms) in ENTITY_STANDINS.items():
        if name in low:
            queries.append(query)
            words += terms
    return queries, list(dict.fromkeys(words))


def sensitive_profile(text: str) -> Tuple[bool, List[str], List[str]]:
    """-> (is_sensitive, sombre queries, topic words)."""
    low = (text or "").lower()
    queries: List[str] = []
    words: List[str] = []
    hit = False
    for pattern, qs, ws in SENSITIVE:
        if re.search(pattern, low):
            hit = True
            queries += qs
            words += ws
    return hit, list(dict.fromkeys(queries)), list(dict.fromkeys(words))


def term_matches(word: str, stem, cterms) -> bool:
    """True if a scene word appears in clip metadata terms (exact stem, or a shared 5-letter prefix such as coast/coastal)."""
    s = stem(word)
    if s in cterms:
        return True
    return len(s) >= 5 and any(len(t) >= 5 and (t.startswith(s[:5]) or s.startswith(t[:5])) for t in cterms)
