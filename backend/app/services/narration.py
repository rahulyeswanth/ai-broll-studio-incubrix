from app.providers.tts import get_tts_provider
from app.schemas.audio import NarrationRequest, NarrationResponse, NarrationSceneResult
from app.services.segmentation import ScriptSegmentationService


class NarrationService:
    """Creates scene-level narration metadata and a continuous synchronized timeline."""

    async def _create_real_narration(self, payload: NarrationRequest) -> NarrationResponse:
        """Real TTS through the pipeline (edge-tts). Writes <storage>/audio/<id>/narration.mp3."""
        import uuid
        from app.pipeline.config import load_settings
        from app.pipeline.narration import get_pipeline_tts, narrate_scenes
        from app.pipeline.text import build_scene_texts

        settings = load_settings()
        texts = build_scene_texts(payload.script)
        out_dir = settings.storage_root / "audio" / uuid.uuid4().hex[:12]
        res = await narrate_scenes(texts, get_pipeline_tts(payload.provider, settings),
                                   payload.voice_id or settings.default_voice, settings.default_language, out_dir, settings)
        scenes = [NarrationSceneResult(scene_id=sc.scene_id, scene_index=sc.index, narration_text=sc.text,
                                       audio_path_or_url=str(res.audio_path), start_time=sc.start, end_time=sc.end,
                                       duration_seconds=round(sc.end - sc.start, 3)) for sc in res.scenes]
        return NarrationResponse(provider_used=payload.provider, target_aspect_ratio=payload.target_aspect_ratio,
                                 total_scenes=len(scenes), total_duration_seconds=res.duration_seconds,
                                 audio_is_placeholder=not res.is_real, audio_file=str(res.audio_path),
                                 scenes=scenes, warnings=res.warnings)

    async def create_narration(self, payload: NarrationRequest) -> NarrationResponse:
        if payload.provider != "mock":
            return await self._create_real_narration(payload)
        scenes = ScriptSegmentationService().segment_script(
            script_text=payload.script,
            scene_hints=payload.scene_hints,
        )
        provider = get_tts_provider(payload.provider)
        output_scenes = []
        cursor = 0.0

        for scene in scenes:
            voiceover = await provider.synthesize(scene.narration_text, payload.voice_id)
            duration = max(0.1, round(voiceover.duration_seconds, 2))
            start = round(cursor, 2)
            end = round(start + duration, 2)
            output_scenes.append(
                NarrationSceneResult(
                    scene_id=scene.scene_id,
                    scene_index=scene.scene_index,
                    narration_text=scene.narration_text,
                    audio_path_or_url=voiceover.audio_path_or_url,
                    start_time=start,
                    end_time=end,
                    duration_seconds=duration,
                )
            )
            cursor = end

        return NarrationResponse(
            provider_used=payload.provider,
            target_aspect_ratio=payload.target_aspect_ratio,
            total_scenes=len(output_scenes),
            total_duration_seconds=round(cursor, 2),
            audio_is_placeholder=payload.provider == "mock",
            scenes=output_scenes,
            warnings=["Mock provider returns placeholder audio references; no playable audio file is generated."]
            if payload.provider == "mock" else [],
        )
