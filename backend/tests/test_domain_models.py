import pytest
from pydantic import ValidationError

from app.schemas.domain import (
    ScriptInput,
    SceneHintInput,
    AudioInputMetadata,
    SceneSegment,
    CaptionSegment,
    Job,
    JobStatus,
    GeneratedVideoMetadata,
)


def test_script_input_valid():
    script = ScriptInput(
        text="Discover the future of artificial intelligence with AI B-roll Studio.",
        target_aspect_ratio="16:9",
        scene_hints=[SceneHintInput(visual_hint="futuristic technology laboratory")],
    )
    assert script.text.startswith("Discover")
    assert script.target_aspect_ratio == "16:9"
    assert len(script.scene_hints) == 1


def test_script_input_all_valid_aspect_ratios():
    for ratio in ["16:9", "9:16", "1:1"]:
        script = ScriptInput(text="Valid narration script", target_aspect_ratio=ratio)
        assert script.target_aspect_ratio == ratio


def test_script_input_empty_or_whitespace_rejected():
    with pytest.raises(ValidationError) as exc:
        ScriptInput(text="   \n\t  ")
    assert "Script text cannot be empty" in str(exc.value)


def test_script_input_invalid_aspect_ratio_rejected():
    with pytest.raises(ValidationError) as exc:
        ScriptInput(text="Valid narration", target_aspect_ratio="4:3")
    assert "Invalid aspect ratio '4:3'" in str(exc.value)


def test_audio_metadata_valid():
    audio = AudioInputMetadata(
        filename="voiceover_interview.mp3",
        file_size_bytes=1024 * 1024 * 5,  # 5 MB
        content_type="audio/mpeg",
        duration_seconds=120.5,
    )
    assert audio.filename == "voiceover_interview.mp3"
    assert audio.file_size_bytes == 5242880


def test_audio_metadata_unsupported_extension_rejected():
    with pytest.raises(ValidationError) as exc:
        AudioInputMetadata(filename="unsupported_script.pdf", file_size_bytes=1000)
    assert "Unsupported audio format '.pdf'" in str(exc.value)


def test_audio_metadata_oversized_file_rejected():
    oversized_bytes = 51 * 1024 * 1024  # 51 MB (limit is 50 MB)
    with pytest.raises(ValidationError) as exc:
        AudioInputMetadata(filename="sample.wav", file_size_bytes=oversized_bytes)
    assert "exceeds maximum limit" in str(exc.value)


def test_audio_metadata_zero_size_rejected():
    with pytest.raises(ValidationError):
        AudioInputMetadata(filename="empty.wav", file_size_bytes=0)


def test_scene_segment_valid_timing():
    scene = SceneSegment(
        start_time=0.0,
        end_time=4.5,
        narration_text="Scene one opening statement.",
        broll_query="modern corporate office workers",
    )
    assert scene.end_time > scene.start_time
    assert scene.broll_query == "modern corporate office workers"


def test_scene_segment_invalid_timing_rejected():
    with pytest.raises(ValidationError) as exc:
        SceneSegment(
            start_time=5.0,
            end_time=3.0,  # Invalid: end_time before start_time
            narration_text="Invalid timing segment.",
            broll_query="sunset",
        )
    assert "must be strictly greater than start_time" in str(exc.value)


def test_job_model_lifecycle_and_bounds():
    job = Job(aspect_ratio="9:16")
    assert job.status == JobStatus.PENDING
    assert job.progress == 0.0

    job.status = JobStatus.PROCESSING
    job.progress = 50.0
    assert job.progress == 50.0

    with pytest.raises(ValidationError):
        Job(progress=150.0)  # Progress > 100 rejected

    with pytest.raises(ValidationError):
        Job(progress=-5.0)  # Progress < 0 rejected
