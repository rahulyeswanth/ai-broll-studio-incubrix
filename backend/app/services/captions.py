from app.schemas.captions import CaptionRequest, CaptionResponse, CaptionSegmentOutput

def _timestamp(seconds: float, vtt: bool = False) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{separator}{millis:03d}"

class CaptionService:
    def generate(self, payload: CaptionRequest) -> CaptionResponse:
        output = []
        index = 1
        for scene in payload.scenes:
            words = scene.narration_text.split()
            chunks = [words[i:i + payload.max_words_per_caption] for i in range(0, len(words), payload.max_words_per_caption)]
            if not chunks:
                continue
            weights = [max(1, len(chunk)) for chunk in chunks]
            total_weight = sum(weights)
            duration = scene.end_time - scene.start_time
            cursor = scene.start_time
            for chunk_index, chunk in enumerate(chunks):
                start = cursor
                if chunk_index == len(chunks) - 1:
                    end = scene.end_time
                else:
                    end = round(start + duration * weights[chunk_index] / total_weight, 3)
                # Guard against rounding collapsing a very short caption.
                if end <= start:
                    end = min(scene.end_time, round(start + 0.01, 3))
                output.append(CaptionSegmentOutput(index=index, start_time=round(start, 3), end_time=round(end, 3), text=" ".join(chunk)))
                index += 1
                cursor = end
        is_vtt = payload.format == "vtt"
        lines = ["WEBVTT", ""] if is_vtt else []
        for caption in output:
            lines.extend([str(caption.index), f"{_timestamp(caption.start_time, is_vtt)} --> {_timestamp(caption.end_time, is_vtt)}", caption.text, ""])
        total_duration = max(scene.end_time for scene in payload.scenes)
        return CaptionResponse(format=payload.format, total_captions=len(output), total_duration_seconds=round(total_duration, 3), captions=output, subtitle_text="\n".join(lines).rstrip() + "\n")
