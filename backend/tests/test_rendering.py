import pytest
from app.schemas.rendering import RenderRequest
from pydantic import ValidationError

def test_render_request_accepts_simple_video_filenames():
    payload = RenderRequest(clips=["ocean.mp4"], target_aspect_ratio="16:9")
    assert payload.clips == ["ocean.mp4"]

def test_render_request_rejects_path_traversal():
    with pytest.raises(ValidationError):
        RenderRequest(clips=["../secret.mp4"])

def test_render_request_rejects_unsupported_output_extension():
    with pytest.raises(ValidationError):
        RenderRequest(clips=["ocean.mp4"], output_filename="output.avi")

def test_render_request_accepts_all_supported_aspect_ratios():
    for ratio in ["16:9", "9:16", "1:1"]:
        assert RenderRequest(clips=["ocean.mp4"], target_aspect_ratio=ratio).target_aspect_ratio == ratio
