import pytest
from app.services.segmentation import ScriptSegmentationService
from app.services.query_extractor import RuleBasedQueryExtractor
from app.schemas.domain import SceneHintInput


def test_rule_based_query_extractor_filters_stopwords():
    extractor = RuleBasedQueryExtractor()
    query = extractor.extract_query("Today we are going to explore the vast ocean coral reefs.")
    assert "ocean" in query
    assert "coral" in query
    assert "reefs" in query
    assert "today" not in query
    assert "are" not in query


def test_rule_based_query_extractor_prefers_visual_hint():
    extractor = RuleBasedQueryExtractor()
    query = extractor.extract_query(
        narration_text="This part is very abstract and conceptual.",
        visual_hint="futuristic quantum computer glowing",
    )
    assert query == "futuristic quantum computer glowing"


def test_rule_based_query_extractor_fallback():
    extractor = RuleBasedQueryExtractor()
    # Sentence composed solely of common words
    query = extractor.extract_query("Here we are.")
    assert len(query) > 0


def test_segmentation_single_sentence():
    service = ScriptSegmentationService()
    script = "AI B-roll Studio automates video creation seamlessly."
    scenes = service.segment_script(script)

    assert len(scenes) == 1
    assert scenes[0].scene_index == 1
    assert scenes[0].scene_id == "scene-001"
    assert scenes[0].narration_text == script
    assert scenes[0].start_time == 0.0
    assert scenes[0].end_time > 0.0
    assert scenes[0].duration_seconds == scenes[0].end_time


def test_segmentation_multiple_paragraphs_and_sentences():
    service = ScriptSegmentationService()
    script = (
        "Artificial intelligence is transforming content creation.\n\n"
        "Modern creators require rapid video production workflows. "
        "High quality B-roll footage enhances audience engagement dramatically."
    )
    scenes = service.segment_script(script)

    assert len(scenes) == 3
    assert [s.scene_index for s in scenes] == [1, 2, 3]
    assert [s.scene_id for s in scenes] == ["scene-001", "scene-002", "scene-003"]

    # Check continuous timeline
    assert scenes[0].start_time == 0.0
    assert scenes[1].start_time == scenes[0].end_time
    assert scenes[2].start_time == scenes[1].end_time
    assert scenes[2].end_time > scenes[2].start_time


def test_segmentation_attaches_visual_hints():
    service = ScriptSegmentationService()
    script = "Sentence one stands alone. Sentence two has a hint."
    hints = [
        SceneHintInput(visual_hint="drone landscape"),
        SceneHintInput(visual_hint="cyberpunk city"),
    ]
    scenes = service.segment_script(script, scene_hints=hints)

    assert len(scenes) == 2
    assert scenes[0].visual_hint == "drone landscape"
    assert scenes[0].broll_query == "drone landscape"
    assert scenes[1].visual_hint == "cyberpunk city"
    assert scenes[1].broll_query == "cyberpunk city"


def test_duration_estimation_calculation():
    # 140 WPM = 2.333 words/sec
    # Text with 14 words: ~6.0s base + 1 comma (0.2s) + 1 period (0.4s) = ~6.6s
    service = ScriptSegmentationService(words_per_minute=140)
    text = "In this opening chapter, we will examine the principles of automated cinematography and sound design."
    duration = service.estimate_duration(text)
    assert 6.0 <= duration <= 7.5


def test_duration_minimum_bound_enforced():
    service = ScriptSegmentationService(min_scene_duration=3.0)
    duration = service.estimate_duration("Quick word.")
    assert duration == 3.0
