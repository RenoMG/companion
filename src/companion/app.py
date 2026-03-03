"""Main application loop — orchestrates STT, LLM, TTS, and memory."""

from __future__ import annotations

import logging
import re
import threading
import time

from companion.config import CompanionConfig
from companion.llm import OllamaClient
from companion.memory import MemoryStore
from companion.stt import SpeechToText
from companion.tts import TextToSpeech

logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"exit", "quit", "goodbye", "bye"}
_SENTENCE_RE = re.compile(r"([^.!?]*[.!?]+)")


class CompanionApp:
    """Real-time voice companion that listens, thinks, and speaks."""

    def __init__(self, config: CompanionConfig) -> None:
        self._config = config
        self._running = False
        self._memory: MemoryStore | None = None
        self._llm: OllamaClient | None = None
        self._stt: SpeechToText | None = None
        self._tts: TextToSpeech | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Initialise components and enter the listen → respond loop."""
        self._memory = MemoryStore(self._config.memory)
        self._llm = OllamaClient(self._config.ollama)

        if not self._llm.check_connection():
            logger.error(
                "Cannot reach Ollama at %s — is it running?",
                self._config.ollama.base_url,
            )
            print(
                f"[Error] Cannot connect to Ollama at "
                f"{self._config.ollama.base_url}.\n"
                f"Start it with: ollama serve"
            )
            self._llm.close()
            self._memory.close()
            return

        print("Loading speech-to-text model …")
        self._stt = SpeechToText(self._config.whisper)

        print("Loading text-to-speech model …")
        self._tts = TextToSpeech(self._config.kokoro)

        name = self._config.companion_name
        greeting = f"Hello! I'm {name}. What would you like to talk about?"
        print(f"\n[{name}]: {greeting}")
        self._tts.speak(greeting)

        self._running = True
        logger.info("Companion is running. Say 'exit' or 'goodbye' to quit.")

        try:
            while self._running:
                # Don't listen while speaking — avoid capturing own voice
                if self._tts.is_speaking():
                    time.sleep(0.1)
                    continue

                transcript = self._stt.listen_once(timeout=30.0)
                if transcript is None:
                    continue

                command = transcript.strip().lower()
                if command in EXIT_COMMANDS:
                    farewell = "Goodbye! It was nice talking with you."
                    print(f"\n[{name}]: {farewell}")
                    self._tts.speak(farewell)
                    while self._tts.is_speaking():
                        time.sleep(0.1)
                    break

                self._handle_turn(transcript)
        finally:
            self.stop()

    def stop(self) -> None:
        """Shut down all components."""
        self._running = False
        if self._tts:
            self._tts.stop()
        if self._llm:
            self._llm.close()
        if self._memory:
            self._memory.close()
        logger.info("Companion stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_turn(self, user_text: str) -> None:
        """Process a single user turn: store, think, speak."""
        assert self._memory is not None
        assert self._llm is not None
        assert self._tts is not None

        name = self._config.companion_name

        print(f"\n[You]: {user_text}")
        self._memory.add_message("user", user_text)

        context = self._memory.build_context(self._config.system_prompt)
        print(f"[{name}]: ", end="", flush=True)

        full_response = ""
        sentence_buffer = ""

        for chunk in self._llm.stream_chat(context):
            full_response += chunk
            sentence_buffer += chunk
            print(chunk, end="", flush=True)

            # Detect completed sentences and send them to TTS immediately
            matches = _SENTENCE_RE.findall(sentence_buffer)
            if matches:
                for sentence in matches:
                    self._tts.speak(sentence)
                # Keep only the unmatched tail
                last_end = sentence_buffer.rfind(matches[-1]) + len(matches[-1])
                sentence_buffer = sentence_buffer[last_end:]

        # Speak any remaining text after the stream ends
        remainder = sentence_buffer.strip()
        if remainder:
            self._tts.speak(remainder)

        print()  # newline after streamed output

        self._memory.add_message("assistant", full_response)

        if self._memory.should_summarise:
            thread = threading.Thread(
                target=self._maybe_summarise, daemon=True
            )
            thread.start()

    def _maybe_summarise(self) -> None:
        """Summarise the conversation in a background thread."""
        assert self._memory is not None
        assert self._llm is not None

        logger.info("Summarising conversation …")
        messages = self._memory.get_recent_messages(
            self._config.memory.summary_threshold
        )
        conversation_text = "\n".join(
            f"{m.role.title()}: {m.content}" for m in messages
        )
        summary = self._llm.summarise(conversation_text)
        if summary:
            self._memory.add_summary(summary)
            self._memory.clear_messages()
            logger.info("Conversation summarised and history cleared.")
