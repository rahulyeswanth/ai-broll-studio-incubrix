"""Resolution-aware caption layout for burned-in subtitles.

Why this exists: FFmpeg's ``subtitles`` filter converts SRT to ASS with a 384x288 script canvas, so ``force_style``
sizes are interpreted on that canvas and then scaled to the real frame (a "FontSize=76" on a 1080x1920 video renders
roughly 500 px tall). Nothing measured text width either, so portrait/square captions overflowed.

This module (1) derives a style from the *actual* output width/height, (2) writes an ASS file whose PlayRes equals the frame
size so every number is a real pixel, (3) wraps by measured text width, and (4) splits cues that do not fit into
consecutive shorter cues instead of shrinking the text. Words are never cut except as a documented last resort for a
single token wider than the whole text area.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .captions import Cue

# (file, ASS family name, bold). Times New Roman first; Liberation Serif / Tinos are its metric-compatible stand-ins on Linux.
FONT_CANDIDATES = (
    ("C:/Windows/Fonts/times.ttf", "Times New Roman", False),
    ("/Library/Fonts/Times New Roman.ttf", "Times New Roman", False),
    ("/System/Library/Fonts/Supplemental/Times New Roman.ttf", "Times New Roman", False),
    ("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", "Times New Roman", False),
    ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", "Liberation Serif", False),
    ("/usr/share/fonts/liberation/LiberationSerif-Regular.ttf", "Liberation Serif", False),
    ("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf", "Liberation Serif", False),
    ("/usr/share/fonts/truetype/crosextra/Tinos-Regular.ttf", "Tinos", False),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "DejaVu Serif", False),
    ("/usr/share/fonts/dejavu/DejaVuSerif.ttf", "DejaVu Serif", False),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVu Sans", True),
    ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", "DejaVu Sans", True),
    ("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "DejaVu Sans", True),
    ("/Library/Fonts/Arial Bold.ttf", "Arial", True),
    ("C:/Windows/Fonts/arialbd.ttf", "Arial", True),
)
SAFETY = 0.97              # measured width may be at most 97% of the usable width (rounding, kerning, hinting)
MIN_SHRINK = 0.78          # a single over-wide token may shrink to this fraction of the base size before it is hyphenated
EM_FALLBACK = 0.64         # average bold glyph width in em, used only when no font file can be loaded


@dataclass(frozen=True)
class CaptionLayout:
    width: int
    height: int
    orientation: str
    font_px: int            # ASS font size in real pixels (libass cell height = ascent + descent)
    margin_x: int
    margin_bottom: int
    outline_px: int
    shadow_px: int
    max_lines: int
    font_name: str = "DejaVu Sans"
    bold: bool = True

    @property
    def text_width(self) -> int:
        return self.width - 2 * self.margin_x

    def to_dict(self) -> Dict[str, object]:
        return {"width": self.width, "height": self.height, "orientation": self.orientation, "font_px": self.font_px,
                "margin_x": self.margin_x, "margin_bottom": self.margin_bottom, "outline_px": self.outline_px,
                "shadow_px": self.shadow_px, "max_lines": self.max_lines, "text_width": self.text_width,
                "font_name": self.font_name, "bold": self.bold}


# font_px is a fraction of the short side; margins are fractions of width/height. Tuned from rendered frames
# (see tests/test_caption_layout.py and docs/evidence/captions/).
_PROFILES = {
    #              font/min(W,H)  margin_x/W  margin_bottom/H  max_lines
    "landscape": (0.030, 0.08, 0.07, 2),
    "square": (0.030, 0.08, 0.08, 2),
    "portrait": (0.032, 0.075, 0.110, 2),     # larger bottom margin keeps clear of platform UI overlays
}


def layout_for(width: int, height: int, font_name: Optional[str] = None) -> CaptionLayout:
    if width <= 0 or height <= 0:
        raise ValueError("Frame size must be positive.")
    ratio = width / height
    orientation = "landscape" if ratio > 1.15 else ("portrait" if ratio < 0.87 else "square")
    f, mx, mb, lines = _PROFILES[orientation]
    font_px = max(12, round(min(width, height) * f))
    override = os.environ.get("CAPTION_FONT_PX")       # exact size at 1080-class output; scaled for other sizes
    if override and override.isdigit() and int(override) > 0:
        font_px = max(8, round(int(override) * min(width, height) / 1080))
    _, family, bold = find_font()
    outline = max(2, round(font_px * 0.085))
    return CaptionLayout(width, height, orientation, font_px, round(width * mx), round(height * mb), outline,
                         max(1, round(outline / 2)), lines, font_name or family, bold)


def find_font() -> Tuple[Optional[Path], str, bool]:
    env = os.environ.get("CAPTION_FONT_FILE")
    if env and Path(env).is_file():
        return Path(env), os.environ.get("CAPTION_FONT_NAME", "Times New Roman"), "bold" in Path(env).name.lower()
    for path, family, bold in FONT_CANDIDATES:
        if Path(path).is_file():
            return Path(path), family, bold
    return None, "serif", False


def find_font_file() -> Optional[Path]:
    return find_font()[0]


class TextMeter:
    """Measures text width exactly as libass will draw it (same font file, ascent+descent size convention)."""

    def __init__(self, font_file: Optional[Path] = None):
        self.font_file = font_file or find_font_file()
        self._cell = 1.0            # (ascent + descent) / em for the font
        self._fonts: Dict[int, object] = {}
        self.exact = False
        if self.font_file:
            try:
                from PIL import ImageFont
                probe = ImageFont.truetype(str(self.font_file), 1000)
                asc, desc = probe.getmetrics()
                self._cell = (asc + desc) / 1000.0
                self.exact = True
            except Exception:  # noqa: BLE001 - PIL missing or unreadable font: fall back to an estimate
                self.font_file = None

    def _font(self, em_px: int):
        from PIL import ImageFont
        f = self._fonts.get(em_px)
        if f is None:
            f = self._fonts[em_px] = ImageFont.truetype(str(self.font_file), max(1, em_px))
        return f

    def width(self, text: str, ass_font_px: float) -> float:
        if not text:
            return 0.0
        if not self.exact:
            return len(text) * EM_FALLBACK * (ass_font_px / 1.164)
        em = ass_font_px / self._cell
        # measure at 8x then scale down, so sub-pixel advances of small sizes are not rounded away
        return self._font(round(em * 8)).getlength(text) / 8.0


@lru_cache(maxsize=1)
def default_meter() -> TextMeter:
    return TextMeter()


@dataclass
class FitReport:
    cues_in: int = 0
    cues_out: int = 0
    split_cues: int = 0
    shrunk_cues: int = 0
    hyphenated_words: int = 0
    max_measured_width: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in ("cues_in", "cues_out", "split_cues", "shrunk_cues", "hyphenated_words")} | {
            "max_measured_width_px": round(self.max_measured_width, 1)}


@dataclass
class FittedCue:
    index: int
    start: float
    end: float
    lines: List[str]
    font_px: int            # may be smaller than layout.font_px for a cue with one over-wide token


def _wrap(words: Sequence[str], meter: TextMeter, font_px: float, max_w: float, max_lines: int) -> Optional[List[str]]:
    """Greedy wrap, then rebalance lines. None if it needs more than ``max_lines`` or a token is wider than ``max_w``."""
    lines: List[List[str]] = [[]]
    for w in words:
        if meter.width(w, font_px) > max_w:
            return None
        cand = lines[-1] + [w]
        if lines[-1] and meter.width(" ".join(cand), font_px) > max_w:
            lines.append([w])
        else:
            lines[-1] = cand
    if len(lines) > max_lines:
        return None
    if len(lines) == 2:     # balance: move words down while the top line stays the longer one and everything still fits
        while len(lines[0]) > 1:
            top, bottom = " ".join(lines[0][:-1]), " ".join([lines[0][-1]] + lines[1])
            if meter.width(bottom, font_px) <= max_w and abs(meter.width(top, font_px) - meter.width(bottom, font_px)) < \
                    abs(meter.width(" ".join(lines[0]), font_px) - meter.width(" ".join(lines[1]), font_px)):
                lines = [lines[0][:-1], [lines[0][-1]] + lines[1]]
            else:
                break
    return [" ".join(l) for l in lines]


_PUNCT_END = re.compile(r"[.!?…,;:)\u3002\uff0c\u0964]$")


def _hyphenate(word: str, meter: TextMeter, font_px: float, max_w: float) -> List[str]:
    """Last resort for one token wider than the frame (e.g. a URL): split with a trailing hyphen."""
    parts, cur = [], ""
    for ch in word:
        if cur and meter.width(cur + ch + "-", font_px) > max_w:
            parts.append(cur + "-")
            cur = ch
        else:
            cur += ch
    parts.append(cur)
    return parts


def fit_cues(cues: Sequence[Cue], layout: CaptionLayout, meter: Optional[TextMeter] = None) -> Tuple[List[FittedCue], FitReport]:
    """Turn cues into measured, wrapped, in-frame caption events. Cues that cannot fit in ``max_lines`` at the base size are
    split into consecutive cues that share the original time window in proportion to their text length."""
    meter = meter or default_meter()
    max_w = layout.text_width * SAFETY
    rep = FitReport(cues_in=len(cues))
    out: List[FittedCue] = []
    for cue in cues:
        words = [w for w in re.split(r"\s+", cue.text.replace("\n", " ").strip()) if w]
        if not words:
            continue            # empty cue: nothing to draw
        chunks = _balance(_chunk(words, layout, meter, max_w, rep), words, layout, meter, max_w)
        if len(chunks) > 1:
            rep.split_cues += 1
        total_chars = sum(len(" ".join(c[0])) for c in chunks) or 1
        t = cue.start
        for i, (cw, lines, fpx) in enumerate(chunks):
            share = len(" ".join(cw)) / total_chars
            end = cue.end if i == len(chunks) - 1 else t + (cue.end - cue.start) * share
            out.append(FittedCue(len(out) + 1, round(t, 3), round(end, 3), lines, fpx))
            for ln in lines:
                rep.max_measured_width = max(rep.max_measured_width, meter.width(ln, fpx))
            t = end
    rep.cues_out = len(out)
    return out, rep


def _chunk(words: List[str], layout: CaptionLayout, meter: TextMeter, max_w: float, rep: FitReport):
    """Split ``words`` into chunks that each wrap into <= max_lines at a legible size. Returns [(words, lines, font_px)]."""
    base = layout.font_px
    chunks: List[Tuple[List[str], List[str], int]] = []
    pending = list(words)
    while pending:
        # take the longest prefix that fits, preferring to stop at punctuation in the second half of it
        best_n, best_lines, last_punct = 0, None, 0
        for n in range(1, len(pending) + 1):
            lines = _wrap(pending[:n], meter, base, max_w, layout.max_lines)
            if lines is None:
                break
            best_n, best_lines = n, lines
            if _PUNCT_END.search(pending[n - 1]) and n >= max(2, int(best_n * 0.5)):
                last_punct = n
        if best_n == 0:
            # the first word alone is wider than the text area: shrink for this cue, then hyphenate as a last resort
            tok = pending[0]
            fpx = base
            while fpx > base * MIN_SHRINK and meter.width(tok, fpx) > max_w:
                fpx -= 1
            if meter.width(tok, fpx) <= max_w:
                rep.shrunk_cues += 1
                chunks.append(([tok], [tok], int(fpx)))
                pending.pop(0)
                continue
            pieces = _hyphenate(tok, meter, base, max_w)
            rep.hyphenated_words += 1
            rep.notes.append(f"token of {len(tok)} chars hyphenated to fit")
            for p in pieces:
                chunks.append(([p], [p], base))
            pending.pop(0)
            continue
        take = best_n
        if best_n < len(pending) and last_punct and last_punct >= best_n * 0.6:
            take = last_punct
        elif best_n < len(pending) and best_n >= 4:
            # avoid leaving a 1-word orphan as the next chunk
            if len(pending) - best_n == 1:
                take = best_n - 1
        lines = _wrap(pending[:take], meter, base, max_w, layout.max_lines) or best_lines
        chunks.append((pending[:take], lines, base))
        pending = pending[take:]
    return chunks


def _balance(chunks, words: List[str], layout: CaptionLayout, meter: TextMeter, max_w: float):
    """Greedy filling leaves a short orphan last chunk. Re-split into the same number of chunks of similar length (cutting
    at punctuation when it is close to the ideal cut). Falls back to the greedy result if any balanced chunk would not fit."""
    k = len(chunks)
    if k < 2 or any(c[2] != layout.font_px or len(c[0]) < 1 for c in chunks) or sum(len(c[0]) for c in chunks) != len(words):
        return chunks
    total = len(" ".join(words))
    cuts, acc, n_chunk = [], 0, 1
    for i, w in enumerate(words[:-1]):
        acc += len(w) + 1
        target = total * n_chunk / k
        near = abs(acc - target) <= max(4, 0.12 * total / k)
        if (acc >= target or (near and _PUNCT_END.search(w))) and n_chunk < k:
            cuts.append(i + 1)
            n_chunk += 1
    if len(cuts) != k - 1:
        return chunks
    out, prev = [], 0
    for cut in cuts + [len(words)]:
        part = words[prev:cut]
        lines = _wrap(part, meter, layout.font_px, max_w, layout.max_lines) if part else None
        if lines is None:
            return chunks
        out.append((part, lines, layout.font_px))
        prev = cut
    return out


def _ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\u29f5").replace("{", "(").replace("}", ")")


def to_ass(fitted: Sequence[FittedCue], layout: CaptionLayout) -> str:
    """ASS document with PlayRes == output frame, so all sizes/margins above are real pixels."""
    L = layout
    # wrap style 2 = no automatic wrapping: our measured line breaks are the only breaks
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {L.width}\nPlayResY: {L.height}\nWrapStyle: 2\nScaledBorderAndShadow: yes\nYCbCr Matrix: None\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,"
        "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"Style: Caption,{L.font_name},{L.font_px},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,{-1 if L.bold else 0},0,0,0,100,100,0,0,1,"
        f"{L.outline_px},{L.shadow_px},2,{L.margin_x},{L.margin_x},{L.margin_bottom},1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    rows = []
    for c in fitted:
        text = "\\N".join(_ass_escape(l) for l in c.lines)
        if c.font_px != L.font_px:
            text = f"{{\\fs{c.font_px}}}" + text
        rows.append(f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(max(c.end, c.start + 0.01))},Caption,,0,0,0,,{text}")
    return head + "\n".join(rows) + "\n"


def font_dir_for_ffmpeg() -> Optional[str]:
    f = find_font_file()
    return str(f.parent) if f else None
