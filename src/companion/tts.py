"""Text-to-speech using Kokoro with threaded playback queue."""

from __future__ import annotations

import io
import logging
import queue
import re
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from kokoro import KPipeline

from companion.config import KokoroConfig

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 24000

# Regex patterns for stripping markdown before speaking
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
    """Kokoro TTS with a background playback thread."""

    def __init__(self, config: KokoroConfig, playback: bool = True) -> None:
        self._voice = config.voice
        self._speed = config.speed
        lang_code = config.voice[0]
        logger.info(
            "Loading Kokoro pipeline (voice=%s, lang=%s, device=%s) …",
            config.voice,
            lang_code,
            config.device,
        )
        self._pipeline = KPipeline(lang_code=lang_code, device=config.device)

        if playback:
            self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
            self._speaking = threading.Event()
            self._running = True
            self._worker = threading.Thread(
                target=self._playback_worker, daemon=True
            )
            self._worker.start()
            logger.info("TTS playback worker started.")
        else:
            self._queue = None
            self._speaking = None
            self._running = False
            self._worker = None
            logger.info("TTS initialised (synthesis only, no speaker playback).")

    def _playback_worker(self) -> None:
        """Drain the audio queue and play each chunk through the speakers."""
        while self._running:
            try:
                audio = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if audio is None:
                # Sentinel — current utterance is complete
                self._speaking.clear()
                continue

            try:
                audio_data = audio.cpu().numpy().astype(np.float32)
                if audio_data.ndim == 1:
                    audio_data = audio_data.reshape(-1, 1)
                with sd.OutputStream(
                    samplerate=_SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    latency="high",
                ) as stream:
                    stream.write(audio_data)
            except sd.PortAudioError as exc:
                logger.error("Playback error: %s", exc)

    def speak(self, text: str) -> None:
        """Synthesise *text* sentence-by-sentence and queue audio for playback."""
        if self._worker is None:
            return
        clean = _strip_markdown(text)
        if not clean:
            return

        self._speaking.set()
        sentences = _split_sentences(clean)

        for sentence in sentences:
            if not self._running:
                break
            try:
                for _graphemes, _phonemes, audio in self._pipeline(
                    sentence,
                    voice=self._voice,
                    speed=self._speed,
                    split_pattern=None,
                ):
                    if audio is not None:
                        self._queue.put(audio)
            except RuntimeError as exc:
                logger.error("TTS synthesis error: %s", exc)

        # Sentinel to signal end of this utterance
        self._queue.put(None)

    def synthesize_wav(self, text: str) -> bytes | None:
        """Synthesise *text* and return WAV bytes (no speaker playback).

        Returns ``None`` if the text is empty after markdown stripping.
        """
        clean = _strip_markdown(text)
        if not clean:
            return None

        chunks: list[np.ndarray] = []
        sentences = _split_sentences(clean)

        for sentence in sentences:
            try:
                for _graphemes, _phonemes, audio in self._pipeline(
                    sentence,
                    voice=self._voice,
                    speed=self._speed,
                    split_pattern=None,
                ):
                    if audio is not None:
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

    def interrupt(self) -> None:
        """Stop current playback and discard queued audio."""
        if self._worker is None:
            return
        sd.stop()
        # Drain the queue
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._speaking.clear()

    def is_speaking(self) -> bool:
        """Return ``True`` if audio is playing or queued."""
        if self._worker is None:
            return False
        return self._speaking.is_set() or not self._queue.empty()

    def stop(self) -> None:
        """Shut down the playback worker."""
        if self._worker is None:
            return
        self.interrupt()
        self._running = False
        self._worker.join(timeout=2.0)
        logger.info("TTS stopped.")
