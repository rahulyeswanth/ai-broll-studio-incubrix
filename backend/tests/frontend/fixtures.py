LONG = "word " * 400 + "https://example.com/" + "a" * 300


def scene(i, status="matched", text="A farmer waters tomato plants.", start=0.0, end=4.0):
    clip = None if status == "unmatched" else {"provider": "pixabay", "clip_id": str(i), "title": "farmer watering " + "x" * 120,
                                              "source_page_url": "https://pixabay.com/videos/x-1/", "thumbnail": None}
    return {"scene_id": f"scene-{i:03d}", "index": i, "start": start, "end": end, "duration": end - start, "narration_text": text,
            "visual_description": "farmer watering", "visual_analysis": {"subject": "farmer", "action": "watering", "objects": ["tomato plants"], "setting": "field"},
            "match_status": "fallback_low_confidence" if status == "review" else ("unmatched" if status == "unmatched" else "matched"),
            "needs_review": status == "review", "placeholder": status == "unmatched", "clip": clip,
            "selection_reason": "needs review: title/tags indicate a different activity" if status == "review" else ("no candidate reached the floor" if status == "unmatched" else "ok"),
            "relevance": {"score": 0.97 if status == "matched" else (0.4 if status == "review" else None), "label": "High" if status == "matched" else ("Low" if status == "review" else None),
                          "passed_threshold": status == "matched", "visual_verified": False,
                          "evidence": {"matched_terms": {"action": "watering"}}},
            "retry_attempts": 0, "download_attempts": [], "alternatives": [], "queries_detail": []}


def bundle(statuses=("matched", "review", "unmatched"), text="A farmer waters tomato plants."):
    scenes = [scene(i + 1, s, text, i * 4.0, i * 4.0 + 4.0) for i, s in enumerate(statuses)]
    summary = [{"level": "warning", "text": "1 scene has no matching footage."}, {"level": "info", "text": "Some footage could not be verified visually."}]
    manifest = {"status": "completed", "outputs": {"video": {"duration_seconds": 12.0, "width": 1920, "height": 1080, "fps": 30}},
                "diagnostics": {"summary": summary, "counts": {}, "items": [
                    {"code": "no_footage", "severity": "warning", "scene_id": "scene-003", "title": "No matching footage", "message": "no candidate reached the floor", "details": {}},
                    {"code": "metadata_only", "severity": "info", "scene_id": None, "title": "Not verified visually", "message": "titles/tags only", "details": {}}]},
                "providers": {"tts": {"name": "mock"}, "footage": {"name": "pixabay"}, "visual_verifier": {}}, "matching": {"threshold": 0.45, "review_floor": 0.3},
                "licensing": {"disclaimer": "verify before publishing"}}
    return manifest, {"scenes": scenes}
