"""Text-to-speech synthesis for browser playback using Kokoro."""

from __future__ import annotations

import io
import logging
import re

import numpy as np
import soundfile as sf
from kokoro import KPipeline

from companion.config import KokoroConfig

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 24000

_MD_BOLD_ITALIC = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_MD_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_MD_INLINE_CODE = re.compile(r"`([^`]+)`")
_MD_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_HRULE = re.compile(r"^-{3,}$", re.MULTILINE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _strip_markdown(text: str) -> str:
    """Remove common markdown formatting so the TTS reads clean text."""
    text = _MD_CODE_BLOCK.sub("", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_BOLD_ITALIC.sub(r"\1", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_HRULE.sub("", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence-ending punctuation for incremental TTS."""
    parts = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in parts if s.strip()]


class TextToSpeech:
    """Kokoro TTS that returns WAV audio for browser playback."""

    def __init__(self, config: KokoroConfig) -> None:
        self._voice = config.voice
        self._speed = config.speed
        lang_code = config.voice[0]
        logger.info(
            "Loading Kokoro pipeline (voice=%s, lang=%s, device=%s) ...",
            config.voice,
            lang_code,
            config.device,
        )
        self._pipeline = KPipeline(lang_code=lang_code, device=config.device)
        logger.info("Kokoro pipeline loaded.")

    def synthesize_wav(self, text: str) -> bytes | None:
        """Synthesise *text* and return WAV bytes."""
        clean = _strip_markdown(text)
        if not clean:
            return None

        chunks: list[np.ndarray] = []
        for sentence in _split_sentences(clean):
            try:
                for _graphemes, _phonemes, audio in self._pipeline(
                    sentence,
                    voice=self._voice,
                    speed=self._speed,
                    split_pattern=None,
                ):
                    if audio is None:
                        continue
                    audio_np = audio.cpu().numpy().astype(np.float32)
                    if audio_np.ndim > 1:
                        audio_np = audio_np.flatten()
                    chunks.append(audio_np)
            except RuntimeError as exc:
                logger.error("TTS synthesis error: %s", exc)

        if not chunks:
            return None

        combined = np.concatenate(chunks)
        buf = io.BytesIO()
        sf.write(buf, combined, _SAMPLE_RATE, format="WAV")
        return buf.getvalue()

    def stop(self) -> None:
        """Release runtime resources."""
        logger.info("TTS stopped.")
