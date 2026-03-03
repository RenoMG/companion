"""Speech-to-text using faster-whisper with energy-based voice activity detection."""

from __future__ import annotations

import logging
import queue
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from companion.config import WhisperConfig

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_FRAMES = 1024


class SpeechToText:
    """Captures microphone audio and transcribes speech with faster-whisper."""

    def __init__(self, config: WhisperConfig) -> None:
        compute_type = "float16" if config.device == "cuda" else "int8"
        logger.info(
            "Loading Whisper model '%s' on %s (compute=%s) …",
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
        self._energy_threshold = config.energy_threshold
        self._pause_duration = config.pause_duration
        logger.info("Whisper model loaded.")

    def listen_once(self, timeout: float = 15.0) -> str | None:
        """Block until the user finishes speaking, then return the transcript.

        Returns ``None`` if no speech is detected before *timeout* seconds.
        """
        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def _callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: sd.CallbackFlags,
        ) -> None:
            if status:
                logger.warning("Audio callback status: %s", status)
            audio_queue.put(indata.copy())

        silence_chunks_needed = int(
            self._pause_duration * SAMPLE_RATE / CHUNK_FRAMES
        )

        recorded_chunks: list[np.ndarray] = []
        speech_started = False
        silence_count = 0
        deadline = time.monotonic() + timeout

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=CHUNK_FRAMES,
            callback=_callback,
        ):
            while time.monotonic() < deadline:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                energy = np.abs(chunk).mean()

                if not speech_started:
                    if energy >= self._energy_threshold:
                        speech_started = True
                        silence_count = 0
                        recorded_chunks.append(chunk)
                else:
                    recorded_chunks.append(chunk)
                    if energy < self._energy_threshold:
                        silence_count += 1
                        if silence_count >= silence_chunks_needed:
                            break
                    else:
                        silence_count = 0

        if not recorded_chunks:
            return None

        audio = np.concatenate(recorded_chunks).flatten().astype(np.float32) / 32768.0

        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=True,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            return None

        logger.info("Transcribed: %s", text)
        return text
