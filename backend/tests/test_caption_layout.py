"""Caption layout tests: measured fit for every ratio plus real FFmpeg-rendered frame checks."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.pipeline.caption_layout import default_meter, fit_cues, layout_for, to_ass, font_dir_for_ffmpeg
from app.pipeline.captions import Cue

SIZES = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}
TEXTS = {
    "short": "Hello world.",
    "long": "Scientists study coral reefs to protect marine life, and fragile ocean ecosystems across the entire world today.",
    "long_word": "The word Pneumonoultramicroscopicsilicovolcanoconiosis is famously long and awkward to display.",
    "numbers": "In 2024, revenue grew 37.5% to $1,234,567.89 (up from 12,000) - a record!",
    "unicode": "Café señor naïve — 日本語のテキスト and Привет мир, “quotes” and emoji-free text.",
    "url": "Visit https://example.com/a/very/long/path/that/keeps/going/and/going/forever/and/ever now",
}
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _cues(text, dur=4.0):
    return [Cue(1, 0.0, dur, text)]


@pytest.mark.parametrize("ratio", SIZES)
@pytest.mark.parametrize("name", TEXTS)
def test_measured_width_inside_text_area(ratio, name):
    w, h = SIZES[ratio]
    layout = layout_for(w, h)
    meter = default_meter()
    fitted, report = fit_cues(_cues(TEXTS[name]), layout, meter)
    assert fitted
    for c in fitted:
        assert len(c.lines) <= layout.max_lines
        for line in c.lines:
            assert meter.width(line, c.font_px) <= layout.text_width, (ratio, name, line)
    assert report.max_measured_width <= layout.text_width


@pytest.mark.parametrize("ratio", SIZES)
def test_timing_is_contiguous_and_preserved(ratio):
    layout = layout_for(*SIZES[ratio])
    fitted, _ = fit_cues([Cue(1, 1.0, 7.0, TEXTS["long"])], layout)
    assert fitted[0].start == pytest.approx(1.0)
    assert fitted[-1].end == pytest.approx(7.0)
    for a, b in zip(fitted, fitted[1:]):
        assert a.end == pytest.approx(b.start)
        assert a.end > a.start


@pytest.mark.parametrize("ratio", SIZES)
def test_words_are_not_cut_except_last_resort(ratio):
    layout = layout_for(*SIZES[ratio])
    fitted, report = fit_cues(_cues(TEXTS["long"]), layout)
    out = " ".join(" ".join(c.lines) for c in fitted).split()
    assert out == TEXTS["long"].split()
    assert report.hyphenated_words == 0


def test_portrait_splits_instead_of_shrinking():
    layout = layout_for(1080, 1920)
    fitted, report = fit_cues(_cues(TEXTS["long"] + " " + TEXTS["long"]), layout)
    assert len(fitted) >= 2 and all(c.font_px == layout.font_px for c in fitted)
    assert report.split_cues == 1


@pytest.mark.parametrize("ratio", SIZES)
def test_empty_cues_are_dropped_and_never_crash(ratio):
    layout = layout_for(*SIZES[ratio])
    fitted, _ = fit_cues([Cue(1, 0, 1, ""), Cue(2, 1, 2, "   "), Cue(3, 2, 3, "Real text")], layout)
    assert [" ".join(c.lines) for c in fitted] == ["Real text"]
    ass = to_ass(fitted, layout)
    assert "Real text" in ass
    assert fit_cues([], layout)[0] == []


def test_layout_font_sizes_scale_with_frame():
    f = {r: layout_for(*s) for r, s in SIZES.items()}
    assert f["9:16"].font_px >= f["1:1"].font_px - 8
    assert all(l.margin_x > 0 and l.margin_bottom > 0 for l in f.values())
    assert f["9:16"].margin_bottom > f["16:9"].margin_bottom
    assert layout_for(540, 960).font_px < f["9:16"].font_px


def test_ass_uses_real_pixel_canvas_and_escapes():
    layout = layout_for(1080, 1920)
    fitted, _ = fit_cues(_cues("Braces {bold} and back\\slash"), layout)
    ass = to_ass(fitted, layout)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "{bold}" not in ass.split("[Events]")[1]


def test_invalid_frame_size():
    with pytest.raises(ValueError):
        layout_for(0, 100)


def _render_frame(tmp_path: Path, ratio: str, text: str, bg: str, t: float = 0.5):
    w, h = SIZES[ratio]
    layout = layout_for(w, h)
    fitted, _ = fit_cues(_cues(text), layout)
    ass = tmp_path / f"c_{ratio.replace(':', 'x')}_{bg}.ass"
    ass.write_text(to_ass(fitted, layout), encoding="utf-8")
    out = tmp_path / f"{ass.stem}.png"
    fonts = font_dir_for_ffmpeg()
    vf = f"subtitles=filename='{ass.name}'" + (f":fontsdir='{fonts}'" if fonts else "")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={bg}:s={w}x{h}:d=5:r=30",
                    "-vf", vf, "-frames:v", "1", "-ss", str(t), str(out)], cwd=tmp_path, check=True, timeout=120)
    return layout, np.asarray(Image.open(out).convert("RGB")).astype(int), fitted


def _bbox(img, bg):
    ref = np.array({"black": (0, 0, 0), "white": (255, 255, 255)}[bg])
    mask = (np.abs(img - ref).sum(axis=2) > 60)
    ys, xs = np.where(mask)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None


@needs_ffmpeg
@pytest.mark.parametrize("bg", ["black", "white"])
@pytest.mark.parametrize("ratio", SIZES)
@pytest.mark.parametrize("name", ["short", "long", "long_word", "numbers", "unicode", "url"])
def test_rendered_text_inside_safe_area_and_visible(tmp_path, ratio, name, bg):
    layout, img, _ = _render_frame(tmp_path, ratio, TEXTS[name], bg)
    box = _bbox(img, bg)
    assert box is not None, "no caption pixels rendered"
    x0, y0, x1, y1 = box
    w, h = SIZES[ratio]
    tol = layout.outline_px + 4
    assert x0 >= layout.margin_x - tol and x1 <= w - layout.margin_x + tol, box
    assert y1 <= h - layout.margin_bottom + tol, box
    assert y0 > h * 0.4, "caption should sit in the lower part of the frame"
    assert x0 > 0 and x1 < w - 1 and y1 < h - 1


@needs_ffmpeg
def test_rendered_caption_is_horizontally_centred(tmp_path):
    _, img, _ = _render_frame(tmp_path, "9:16", TEXTS["long"], "black")
    x0, _, x1, _ = _bbox(img, "black")
    assert abs((x0 + x1) / 2 - 540) < 25
