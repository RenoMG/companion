"""Speech-to-text for browser-uploaded audio using faster-whisper."""

from __future__ import annotations

import logging

import numpy as np
from faster_whisper import WhisperModel

from companion.config import WhisperConfig

logger = logging.getLogger(__name__)


class SpeechToText:
    """Transcribe uploaded audio with faster-whisper."""

    def __init__(self, config: WhisperConfig) -> None:
        compute_type = "float16" if config.device == "cuda" else "int8"
        logger.info(
            "Loading Whisper model '%s' on %s (compute=%s) ...",
            config.model_size,
            config.device,
            compute_type,
        )
        self._model = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=compute_type,
        )
        self._language = config.language
        logger.info("Whisper model loaded.")

    def transcribe_audio(self, audio: np.ndarray) -> str | None:
        """Transcribe a 16 kHz float32 mono audio array."""
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            return None

        logger.info("Transcribed (from file): %s", text)
        return text
