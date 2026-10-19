"""Word/phrase timing utilities and Mode-B scene building. Every function states how its timings were obtained
so the manifest never overclaims alignment precision."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import PipelineSettings
from .models import Scene, WordTiming
from .text import split_long, split_sentences

_NON_ALNUM = re.compile(r"[^\w]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t for t in (text or "").split() if t]


def norm(tok: str) -> str:
    return _NON_ALNUM.sub("", tok.lower())


def distribute_words(tokens: Sequence[str], start: float, end: float) -> List[WordTiming]:
    """Spread tokens across [start,end] proportionally to character length (ESTIMATED timing)."""
    if not tokens:
        return []
    span = max(end - start, 1e-3)
    weights = [max(1, len(norm(t))) + (2 if re.search(r"[.!?,;:]$", t) else 0) for t in tokens]
    total = float(sum(weights))
    out, cur = [], start
    for t, w in zip(tokens, weights):
        nxt = cur + span * w / total
        out.append(WordTiming(t, round(cur, 3), round(nxt, 3)))
        cur = nxt
    out[-1].end = round(end, 3)
    return out


def explode_words(words: Sequence[WordTiming]) -> List[WordTiming]:
    """Ensure each WordTiming holds exactly one whitespace-free token."""
    out: List[WordTiming] = []
    for w in words:
        parts = tokenize(w.word)
        if not parts:
            continue
        if len(parts) == 1:
            out.append(WordTiming(parts[0], w.start, w.end))
        else:
            out.extend(distribute_words(parts, w.start, w.end))
    return out


def map_script_to_timed_words(script_tokens: Sequence[str], timed: Sequence[WordTiming], start: float, end: float
                              ) -> Tuple[List[WordTiming], float]:
    """Give each *script* token a time using the timed (ASR/TTS) words as anchors.

    Returns (timed script tokens, fraction of script tokens that were matched to an anchor)."""
    if not script_tokens:
        return [], 1.0
    a = [norm(t) for t in script_tokens]
    b = [norm(w.word) for w in timed]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    anchors: Dict[int, WordTiming] = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            anchors[blk.a + k] = timed[blk.b + k]
    out: List[Optional[WordTiming]] = [None] * len(script_tokens)
    for i, w in anchors.items():
        out[i] = WordTiming(script_tokens[i], w.start, w.end)
    i = 0
    while i < len(out):
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] is None:
            j += 1
        lo = out[i - 1].end if i > 0 and out[i - 1] else start
        hi = out[j].start if j < len(out) and out[j] else end
        if hi <= lo:
            hi = lo + 0.05 * (j - i)
        for k, w in enumerate(distribute_words(script_tokens[i:j], lo, hi)):
            out[i + k] = w
        i = j
    res = [w for w in out if w is not None]
    # enforce monotonic, non-negative, positive-duration timings
    prev = start
    for w in res:
        w.start = max(w.start, prev)
        w.end = max(w.end, w.start + 0.01)
        prev = w.end
    matched = len(anchors) / len(script_tokens)
    return res, matched


def similarity(script: str, transcript: str) -> float:
    a = [norm(t) for t in tokenize(script) if norm(t)]
    b = [norm(t) for t in tokenize(transcript) if norm(t)]
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def phrases_to_words(phrases: Sequence[Tuple[float, float, str]]) -> List[WordTiming]:
    out: List[WordTiming] = []
    for s, e, t in phrases:
        out.extend(distribute_words(tokenize(t), s, e))
    return out


# --------------------------------------------------------------------------- Mode B scenes
def scenes_from_words(words: Sequence[WordTiming], audio_duration: float, settings: PipelineSettings,
                      timing_method: str) -> List[Scene]:
    """Split a timed transcript into meaningful scenes that tile [0, audio_duration] exactly.

    Scene i starts where scene i-1 ends; each boundary sits at the first word of the next scene so that
    pauses stay with the preceding scene and no timeline gap exists."""
    words = [w for w in explode_words(words) if w.start < audio_duration]
    if not words:
        return []
    for w in words:
        w.end = min(w.end, audio_duration)
    text = " ".join(w.word for w in words)
    total_speech = max(words[-1].end - words[0].start, 0.5)
    wps = len(words) / total_speech
    max_words = max(6, int(settings.max_scene_seconds * wps))
    units: List[str] = []
    for sent in split_sentences(text):
        units.extend(split_long(sent, max_words))
    counts = [len(u.split()) for u in units]
    # token accounting must be exact; if the splitter changed tokenisation, fall back to one unit
    if sum(counts) != len(words):
        units, counts = [text], [len(words)]

    # (start_idx, end_idx_exclusive)
    spans: List[Tuple[int, int]] = []
    cur = 0
    for c in counts:
        spans.append((cur, cur + c))
        cur += c

    def dur(span: Tuple[int, int], nxt_start: Optional[float]) -> float:
        s = words[span[0]].start
        e = nxt_start if nxt_start is not None else audio_duration
        return e - s

    # merge scenes shorter than min_scene_seconds into the following (or previous) scene
    changed = True
    while changed and len(spans) > 1:
        changed = False
        for i, sp in enumerate(spans):
            nxt = words[spans[i + 1][0]].start if i + 1 < len(spans) else None
            d = dur(sp, nxt) if i > 0 else (nxt if nxt is not None else audio_duration)
            if d < settings.min_scene_seconds:
                if i + 1 < len(spans):
                    spans[i:i + 2] = [(sp[0], spans[i + 1][1])]
                else:
                    spans[i - 1:i + 1] = [(spans[i - 1][0], sp[1])]
                changed = True
                break

    scenes: List[Scene] = []
    for i, (a, b) in enumerate(spans):
        start = 0.0 if i == 0 else round(words[a].start, 3)
        end = round(words[spans[i + 1][0]].start, 3) if i + 1 < len(spans) else round(audio_duration, 3)
        if end <= start:
            end = round(start + 0.05, 3)
        ws = [WordTiming(w.word, w.start, w.end) for w in words[a:b]]
        scenes.append(Scene(f"scene-{i + 1:03d}", i + 1, " ".join(w.word for w in ws), start, end, ws, timing_method))
    return scenes
