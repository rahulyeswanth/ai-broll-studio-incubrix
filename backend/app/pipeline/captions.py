"""Caption building from timed words, SRT writing and strict SRT validation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import PipelineSettings
from .models import WordTiming


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str  # may contain a single "\n" line break


def fmt_ts(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _break_lines(words: Sequence[str], max_chars: int, max_lines: int) -> str:
    text = " ".join(words)
    if len(text) <= max_chars or max_lines < 2 or len(words) < 2:
        return text
    best, best_score = None, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        over = max(0, len(a) - max_chars) + max(0, len(b) - max_chars)
        score = (over, abs(len(a) - len(b)))
        if best_score is None or score < best_score:
            best, best_score = f"{a}\n{b}", score
    return best or text


def build_cues(words: Sequence[WordTiming], audio_duration: float, settings: PipelineSettings) -> List[Cue]:
    mc, ml = settings.caption_max_chars_per_line, settings.caption_max_lines
    cap = mc * ml
    groups: List[List[WordTiming]] = []
    cur: List[WordTiming] = []
    for i, w in enumerate(words):
        if cur:
            chars = len(" ".join(x.word for x in cur)) + 1 + len(w.word)
            gap = w.start - cur[-1].end
            too_long = chars > cap or (w.end - cur[0].start) > settings.caption_max_seconds
            pause = gap > 0.7
            prev = cur[-1].word
            sentence_end = bool(re.search(r"[.!?]$", prev)) and len(cur) >= 3
            clause_end = bool(re.search(r"[,;:]$", prev)) and chars > cap * 0.65
            if too_long or pause or sentence_end or clause_end:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    raw = [(g[0].start, g[-1].end, [x.word for x in g]) for g in groups]
    # merge groups that cannot reach the minimum duration even after extension
    cues: List[Tuple[float, float, List[str]]] = []
    for s, e, ws in raw:
        s = max(0.0, s)
        if cues and (e - s) < settings.caption_min_seconds and len(" ".join(cues[-1][2] + ws)) <= cap \
                and (e - cues[-1][0]) <= settings.caption_max_seconds:
            ps, pe, pw = cues[-1]
            cues[-1] = (ps, e, pw + ws)
        else:
            cues.append((s, e, ws))

    out: List[Cue] = []
    for i, (s, e, ws) in enumerate(cues):
        nxt = cues[i + 1][0] if i + 1 < len(cues) else audio_duration
        e = max(e, s + settings.caption_min_seconds)       # hold on screen at least min duration...
        e = min(e, max(nxt, s + 0.01), audio_duration)     # ...but never overlap the next cue or run past audio
        if out:
            s = max(s, out[-1].end)
        s = min(s, audio_duration)
        if e - s < 0.05:   # nothing sensible left: attach text to previous cue if possible
            if out:
                out[-1].text = _break_lines((out[-1].text.replace("\n", " ") + " " + " ".join(ws)).split(), mc, ml)
                out[-1].end = max(out[-1].end, min(audio_duration, s + 0.05))
            continue
        out.append(Cue(len(out) + 1, round(s, 3), round(e, 3), _break_lines(ws, mc, ml)))
    # final strict pass
    fixed: List[Cue] = []
    for c in out:
        if fixed and c.start < fixed[-1].end:
            c.start = fixed[-1].end
        c.end = min(c.end, audio_duration)
        if c.end > c.start:
            c.index = len(fixed) + 1
            fixed.append(c)
    return fixed


def to_srt(cues: Sequence[Cue]) -> str:
    blocks = [f"{c.index}\n{fmt_ts(c.start)} --> {fmt_ts(c.end)}\n{c.text}\n" for c in cues]
    return "\n".join(blocks)


_TS = r"(\d{2,}):(\d{2}):(\d{2}),(\d{3})"
_TIME_RE = re.compile(rf"^{_TS} --> {_TS}$")


def parse_srt(text: str) -> List[Cue]:
    cues: List[Cue] = []
    for block in re.split(r"\n\s*\n", text.strip().replace("\r\n", "\n")):
        lines = block.split("\n")
        if len(lines) < 3 or not lines[0].strip().isdigit():
            raise ValueError(f"Malformed SRT block: {block[:60]!r}")
        m = _TIME_RE.match(lines[1].strip())
        if not m:
            raise ValueError(f"Malformed SRT timestamp line: {lines[1]!r}")
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        cues.append(Cue(int(lines[0]), start, end, "\n".join(lines[2:])))
    return cues


def validate_srt(text: str, audio_duration: Optional[float], settings: Optional[PipelineSettings] = None) -> List[str]:
    """Return a list of problems (empty == valid)."""
    problems: List[str] = []
    try:
        cues = parse_srt(text)
    except ValueError as exc:
        return [str(exc)]
    if not cues:
        return ["SRT contains no captions."]
    prev_end = 0.0
    for n, c in enumerate(cues, 1):
        if c.index != n:
            problems.append(f"cue {n}: index is {c.index}, expected {n}")
        if c.start < 0:
            problems.append(f"cue {n}: negative start")
        if c.end <= c.start:
            problems.append(f"cue {n}: zero or negative duration")
        if c.start + 1e-6 < prev_end:
            problems.append(f"cue {n}: overlaps or precedes the previous cue")
        if audio_duration is not None and c.end > audio_duration + 0.002:
            problems.append(f"cue {n}: ends at {c.end:.3f}s, after audio end {audio_duration:.3f}s")
        if not c.text.strip():
            problems.append(f"cue {n}: empty text")
        if settings:
            lines = c.text.split("\n")
            if len(lines) > settings.caption_max_lines:
                problems.append(f"cue {n}: {len(lines)} lines")
        prev_end = c.end
    return problems
