# Audio fixtures

`speech_clear.*` is synthetic narration generated offline with Piper TTS (voice `en_US-lessac-medium`) from the sentence set
in `tests/test_audio_inputs.py::SPEECH_TEXT`. They are test inputs only and are never shipped as product output.
`silent.wav`, `music_only.mp3`, `quiet_speech.wav` are derived with FFmpeg. `corrupt.mp3` is a truncated MP3 followed by random
bytes; `not_audio.mp3` is plain text with an audio extension; `empty.mp3` is 0 bytes.
