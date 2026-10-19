"""End-to-end generation pipeline (script/audio -> narration -> B-roll -> captions -> MP4).

The modules in this package are intentionally dependency-light (standard library,
plus FFmpeg/FFprobe on PATH, and optionally OpenCV/NumPy) so that the core logic can be
unit-tested without FastAPI, Pydantic or any paid/network service.
"""
