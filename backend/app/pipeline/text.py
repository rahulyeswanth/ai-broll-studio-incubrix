"""Script -> scene text splitting. Never rewrites or drops narration content."""
from __future__ import annotations

import re
from typing import List

_ABBREVIATIONS = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "Inc.", "Ltd.", "vs.", "etc.", "e.g.", "i.e.", "approx."]
_PLACEHOLDER = "\u0001"
_SENT_RE = re.compile(r"(?<=[.!?\u3002\u0964])[\"'\u201d\u2019)\]]*\s+(?=[\"'\u201c\u2018(\[]*[^\W_])", re.UNICODE)
_CLAUSE_RE = re.compile(r"(?<=[,;:\u2014])\s+")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_sentences(text: str) -> List[str]:
    """Split on blank lines, single newlines and sentence punctuation. Concatenation (space-joined)
    of the result equals the whitespace-normalised input."""
    out: List[str] = []
    for line in re.split(r"\n+", text or ""):
        line = normalize_whitespace(line)
        if not line:
            continue
        protected = line
        for ab in _ABBREVIATIONS:
            protected = protected.replace(ab, ab.replace(".", _PLACEHOLDER))
        # decimal numbers like 3.5 must not split
        protected = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + _PLACEHOLDER + m.group(2), protected)
        for part in _SENT_RE.split(protected):
            part = part.replace(_PLACEHOLDER, ".").strip()
            if part:
                out.append(part)
    return out


def _wc(s: str) -> int:
    return len(s.split())


def split_long(sentence: str, max_words: int) -> List[str]:
    if _wc(sentence) <= max_words:
        return [sentence]
    clauses = [c for c in _CLAUSE_RE.split(sentence) if c.strip()]
    chunks: List[str] = []
    cur = ""
    for c in clauses:
        if cur and _wc(cur) + _wc(c) > max_words:
            chunks.append(cur)
            cur = c
        else:
            cur = f"{cur} {c}".strip()
    if cur:
        chunks.append(cur)
    final: List[str] = []
    for ch in chunks:  # hard split clauses that are still too long
        words = ch.split()
        for i in range(0, len(words), max_words):
            final.append(" ".join(words[i:i + max_words]))
    return final


def build_scene_texts(script: str, *, max_words: int = 32, min_words: int = 5) -> List[str]:
    """Meaningful scene texts: one per sentence, tiny sentences merged forward, huge ones split at clauses."""
    units: List[str] = []
    for s in split_sentences(script):
        units.extend(split_long(s, max_words))
    merged: List[str] = []
    carry = ""
    for u in units:
        cand = f"{carry} {u}".strip() if carry else u
        if _wc(cand) < min_words:
            carry = cand
            continue
        merged.append(cand)
        carry = ""
    if carry:
        if merged and _wc(merged[-1]) + _wc(carry) <= max_words + min_words:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)
    return merged
