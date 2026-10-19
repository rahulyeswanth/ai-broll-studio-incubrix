import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from app.schemas.rendering import RenderRequest, RenderResponse

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

class VideoRenderingError(Exception):
    pass

class VideoRenderingService:
    """Render real local video clips into a single MP4 using FFmpeg."""
    def __init__(self, storage_root: str | Path | None = None):
        # default is anchored to the backend folder (not the current working directory)
        self.storage_root = Path(storage_root).resolve() if storage_root else (Path(__file__).resolve().parents[2] / "storage")
        self.source_dir = self.storage_root / "source_videos"
        self.audio_dir = self.storage_root / "source_audio"
        self.output_dir = self.storage_root / "rendered_videos"

    def render_video(self, payload: RenderRequest) -> RenderResponse:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise VideoRenderingError("FFmpeg and FFprobe must be installed and available on PATH.")
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        clip_paths = [self._safe_existing(self.source_dir, name) for name in payload.clips]
        for path in clip_paths:
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise VideoRenderingError(f"Unsupported video format: {path.name}")
        audio_path = self._safe_existing(self.audio_dir, payload.audio_filename) if payload.audio_filename else None
        output_path = self._safe_output(payload.output_filename)
        dimensions = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}
        width, height = dimensions[payload.target_aspect_ratio]
        command = [ffmpeg, "-hide_banner", "-y"]
        for path in clip_paths:
            command += ["-i", str(path)]
        audio_input_index = None
        if audio_path:
            audio_input_index = len(clip_paths)
            command += ["-i", str(audio_path)]
        filters = []
        for index in range(len(clip_paths)):
            trim = ""
            if payload.clip_durations is not None:
                trim = f"trim=duration={payload.clip_durations[index]},"
            filters.append(f"[{index}:v:0]{trim}setpts=PTS-STARTPTS,scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,fps=30,format=yuv420p[v{index}]")
        concat_inputs = "".join(f"[v{i}]" for i in range(len(clip_paths)))
        filters.append(f"{concat_inputs}concat=n={len(clip_paths)}:v=1:a=0[vconcat]")

        subtitle_temp_path = None
        video_output_label = "vconcat"
        if payload.subtitle_text and payload.subtitle_text.strip():
            # Store the temporary SRT beside the output and run FFmpeg with
            # output_dir as cwd. A simple relative filename avoids Windows
            # drive-letter escaping issues in FFmpeg's subtitles filter.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                suffix=".srt",
                prefix="captions_",
                dir=self.output_dir,
                delete=False,
            ) as subtitle_file:
                subtitle_file.write(payload.subtitle_text)
                subtitle_temp_path = Path(subtitle_file.name)
            subtitle_name = subtitle_temp_path.name.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
            style = "FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=40"
            filters.append(f"[vconcat]subtitles=filename='{subtitle_name}':force_style='{style}'[vsub]")
            video_output_label = "vsub"

        command += ["-filter_complex", ";".join(filters), "-map", f"[{video_output_label}]"]
        if audio_input_index is not None:
            command += ["-map", f"{audio_input_index}:a:0", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest"]
        elif payload.include_audio_from_clips and len(clip_paths) == 1:
            command += ["-map", "0:a?", "-c:a", "aac", "-b:a", "192k"]
        else:
            command += ["-an"]
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-r", "30", "-movflags", "+faststart", str(output_path)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=1800, check=False, cwd=str(self.output_dir))
        except subprocess.TimeoutExpired as exc:
            raise VideoRenderingError("Video rendering timed out after 30 minutes.") from exc
        finally:
            if subtitle_temp_path is not None:
                subtitle_temp_path.unlink(missing_ok=True)
        if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            detail = (completed.stderr or "FFmpeg did not produce an output file.")[-2000:]
            raise VideoRenderingError(f"FFmpeg rendering failed: {detail}")
        probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)], capture_output=True, text=True, timeout=30, check=False)
        try:
            duration = round(float(probe.stdout.strip()), 3)
        except (ValueError, AttributeError):
            duration = 0.0
        return RenderResponse(output_filename=output_path.name, output_path=str(output_path), target_aspect_ratio=payload.target_aspect_ratio, clip_count=len(clip_paths), duration_seconds=duration, width=width, height=height, file_size_bytes=output_path.stat().st_size, message="Video rendered successfully.")

    def _safe_existing(self, directory: Path, filename: str | None) -> Path:
        if filename is None:
            raise VideoRenderingError("A filename is required.")
        path = (directory / filename).resolve()
        if path.parent != directory.resolve():
            raise VideoRenderingError("Paths outside the configured storage folder are not allowed.")
        if not path.is_file():
            raise VideoRenderingError(f"File not found: {filename}. Place it in {directory}.")
        return path

    def _safe_output(self, filename: str) -> Path:
        path = (self.output_dir / filename).resolve()
        if path.parent != self.output_dir.resolve():
            raise VideoRenderingError("Output paths outside rendered_videos are not allowed.")
        return path
