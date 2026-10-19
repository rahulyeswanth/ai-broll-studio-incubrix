import re
from typing import List, Optional

from app.core.config import get_settings
from app.schemas.domain import SceneHintInput, SceneSegment
from app.services.query_extractor import BaseQueryExtractor, RuleBasedQueryExtractor


class ScriptSegmentationService:
    """
    Segments raw script text into structured scenes with cumulative timeline timings
    and visual B-roll search queries.
    """

    # Matches sentence boundary: '.', '!', '?' followed by whitespace or end-of-string,
    # ignoring common title abbreviations (e.g. Mr., Dr., Inc.)
    SENTENCE_SPLIT_REGEX = re.compile(
        r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Inc\.)(?<!Ltd\.)(?<!vs\.)(?<!approx\.)(?<=[.!?])\s+(?=[A-Z0-9\"'])",
        re.IGNORECASE | re.UNICODE
    )

    def __init__(
        self,
        query_extractor: Optional[BaseQueryExtractor] = None,
        words_per_minute: Optional[int] = None,
        min_scene_duration: Optional[float] = None,
    ):
        settings = get_settings()
        self.query_extractor = query_extractor or RuleBasedQueryExtractor()
        self.wpm = words_per_minute or settings.DEFAULT_WORDS_PER_MINUTE
        self.words_per_second = self.wpm / 60.0
        self.min_scene_duration = min_scene_duration or settings.MIN_SCENE_DURATION_SECONDS

    def estimate_duration(self, text: str) -> float:
        """
        Estimates narration duration in seconds:
        - Base duration: (word count / words_per_second)
        - Punctuation pause weighting:
            * Comma/Semicolon/Colon: +0.20s
            * Terminal Period/Exclamation/Question: +0.40s
        - Enforces minimum scene duration constraint.
        """
        words = [w for w in text.split() if w]
        word_count = len(words)
        base_duration = word_count / self.words_per_second if self.words_per_second > 0 else 2.0

        # Punctuation pauses
        minor_pauses = len(re.findall(r"[,;:]", text)) * 0.20
        major_pauses = len(re.findall(r"[.!?]", text)) * 0.40

        total_est = base_duration + minor_pauses + major_pauses
        return max(round(total_est, 2), self.min_scene_duration)

    def _split_into_sentences(self, text: str) -> List[str]:
        # Split on paragraph breaks first (one or more empty lines)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        sentence_units: List[str] = []

        for p in paragraphs:
            # Normalize internal newlines to spaces
            normalized = re.sub(r"\s+", " ", p).strip()
            # Split by sentence boundaries
            splits = self.SENTENCE_SPLIT_REGEX.split(normalized)
            for s in splits:
                cleaned = s.strip()
                if cleaned:
                    sentence_units.append(cleaned)

        return sentence_units

    def segment_script(
        self,
        script_text: str,
        scene_hints: Optional[List[SceneHintInput]] = None,
    ) -> List[SceneSegment]:
        """
        Splits script text into ordered SceneSegments with timing and B-roll queries.
        Preserves original narration text verbatim.
        """
        sentences = self._split_into_sentences(script_text)
        if not sentences:
            sentences = [script_text.strip()]

        hints = scene_hints or []
        scenes: List[SceneSegment] = []
        current_time = 0.0

        for idx, sentence in enumerate(sentences, start=1):
            hint_obj = hints[idx - 1] if idx <= len(hints) else None
            visual_hint = hint_obj.visual_hint if hint_obj else None

            # Generate query using extractor
            broll_query = self.query_extractor.extract_query(sentence, visual_hint=visual_hint)

            duration = self.estimate_duration(sentence)
            start_time = round(current_time, 2)
            end_time = round(current_time + duration, 2)
            current_time = end_time

            scene = SceneSegment(
                scene_id=f"scene-{idx:03d}",
                scene_index=idx,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                narration_text=sentence,
                broll_query=broll_query,
                visual_hint=visual_hint,
                selected_asset=None,
            )
            scenes.append(scene)

        return scenes
