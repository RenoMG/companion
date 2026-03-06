"""Main application loop — orchestrates STT, LLM, TTS, and memory."""

from __future__ import annotations

import json
import logging
import re
import signal
import sys
import threading
import time
from pathlib import Path

from companion.config import CompanionConfig, load_config
from companion.llm import FACT_TOOLS, OllamaClient
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
        self._stopped = False
        self._summarising = threading.Event()
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
        greeting = f"Hello! I'm {name}. Start talking to initiate a conversation!"
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
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        if self._tts:
            self._tts.stop()
        # Wait for any background summarisation to finish.
        if self._summarising.is_set():
            logger.info("Waiting for background summarisation to finish …")
            start = time.monotonic()
            while self._summarising.is_set() and (time.monotonic() - start) < 30:
                time.sleep(0.5)
        # Summarise remaining messages before closing (best-effort).
        if self._memory and self._llm:
            try:
                self._summarise_on_exit()
            except Exception:
                logger.exception("Exit summarisation failed.")
                # Still wipe messages so next session starts clean.
                try:
                    self._memory.clear_messages()
                except Exception:
                    logger.exception("Failed to clear messages on exit.")
        if self._llm:
            self._llm.close()
        if self._memory:
            self._memory.close()
        logger.info("Companion stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _summarise_on_exit(self) -> None:
        """Summarise all remaining messages and wipe the messages table."""
        assert self._memory is not None
        assert self._llm is not None

        messages = self._memory.get_all_messages()
        if not messages:
            logger.info("No messages to summarise on exit.")
            return

        print("Summarising conversation before exit …")
        logger.info("Summarising %d messages on exit.", len(messages))

        existing_summary = self._memory.get_latest_summary()
        conversation_text = "\n".join(
            f"{m.role.title()}: {m.content}" for m in messages
        )

        summary = self._llm.summarise(conversation_text, existing_summary)
        if summary:
            self._memory.add_summary(summary)
            logger.info("Exit summary saved.")

        self._memory.clear_messages()
        logger.info("Messages wiped for next session.")

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

        tool_calls: list = []
        full_response = self._stream_and_speak(context, tool_calls)

        # If the model requested tool calls, execute them and get a follow-up.
        if tool_calls:
            tool_results = self._execute_tool_calls(tool_calls)
            # Build the follow-up context with the tool interaction.
            context.append({"role": "assistant", "content": full_response, "tool_calls": tool_calls})
            for result in tool_results:
                context.append(result)
            # Stream the model's follow-up response (no tools this round).
            full_response = self._stream_and_speak(context)

        print()  # newline after streamed output

        self._memory.add_message("assistant", full_response)

        if self._memory.should_summarise and not self._summarising.is_set():
            self._summarising.set()
            thread = threading.Thread(
                target=self._maybe_summarise, daemon=True
            )
            thread.start()

    def _stream_and_speak(
        self,
        context: list[dict],
        tool_calls_out: list | None = None,
    ) -> str:
        """Stream an LLM response, printing and speaking as sentences arrive.

        Returns the full response text.  If *tool_calls_out* is provided,
        tool call information is appended to it.
        """
        assert self._llm is not None
        assert self._tts is not None

        full_response = ""
        sentence_buffer = ""

        tools = FACT_TOOLS if tool_calls_out is not None else None
        for chunk in self._llm.stream_chat(context, tools=tools, tool_calls_out=tool_calls_out):
            full_response += chunk
            sentence_buffer += chunk
            print(chunk, end="", flush=True)

            matches = _SENTENCE_RE.findall(sentence_buffer)
            if matches:
                for sentence in matches:
                    self._tts.speak(sentence)
                last_end = sentence_buffer.rfind(matches[-1]) + len(matches[-1])
                sentence_buffer = sentence_buffer[last_end:]

        remainder = sentence_buffer.strip()
        if remainder:
            self._tts.speak(remainder)

        return full_response

    def _execute_tool_calls(self, tool_calls: list) -> list[dict]:
        """Execute tool calls against the memory store.

        Returns a list of ``{"role": "tool", ...}`` messages for the LLM.
        """
        assert self._memory is not None
        results: list[dict] = []

        for tc in tool_calls:
            func = tc.get("function", {})
            func_name = func.get("name", "")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if func_name == "store_fact":
                key = args.get("key", "")
                value = args.get("value", "")
                if key and value:
                    self._memory.set_fact(key, value)
                    result_text = f"Stored fact: {key} = {value}"
                    logger.info("Tool call: store_fact(%s, %s)", key, value)
                else:
                    result_text = "Error: key and value are required"
            elif func_name == "delete_fact":
                key = args.get("key", "")
                if key:
                    self._memory.delete_fact(key)
                    result_text = f"Deleted fact: {key}"
                    logger.info("Tool call: delete_fact(%s)", key)
                else:
                    result_text = "Error: key is required"
            else:
                result_text = f"Unknown tool: {func_name}"

            results.append({"role": "tool", "content": result_text})

        return results

    def _maybe_summarise(self) -> None:
        """Summarise old messages in a background thread (rolling window).

        Keeps the most recent ``max_context_messages`` in the database and
        only summarises/deletes the older ones.  An existing summary is fed
        back into the LLM prompt so context is never lost across cycles.
        """
        assert self._memory is not None
        assert self._llm is not None

        try:
            logger.info("Summarising conversation …")

            existing_summary = self._memory.get_latest_summary()

            old_messages = self._memory.get_messages_for_summary(
                self._config.memory.max_context_messages
            )
            if not old_messages:
                logger.info("No old messages to summarise.")
                return

            max_id = old_messages[-1].id

            conversation_text = "\n".join(
                f"{m.role.title()}: {m.content}" for m in old_messages
            )

            summary = self._llm.summarise(conversation_text, existing_summary)

            if summary:
                self._memory.add_summary(summary)
                self._memory.delete_messages_up_to(max_id)
                logger.info(
                    "Conversation summarised; deleted messages up to id=%d.",
                    max_id,
                )
        finally:
            self._summarising.clear()


def setup_logging() -> None:
    """Configure logging to write to files in a ``Logs/`` directory.

    * ``Logs/companion.log`` — all INFO-and-above messages.
    * ``Logs/error.log``     — ERROR-and-above messages only.

    No StreamHandler is added, so the terminal stays clean for the
    interactive conversation UI.
    """
    log_dir = Path("Logs")
    log_dir.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / "companion.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    error_handler = logging.FileHandler(log_dir / "error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(error_handler)


def main() -> None:
    """Load configuration and start the voice companion."""
    setup_logging()

    config = load_config()
    app = CompanionApp(config)

    def _handle_signal(signum: int, _frame: object) -> None:
        logging.getLogger(__name__).info(
            "Received signal %s — shutting down.", signum
        )
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    app.run()


def main_web() -> None:
    """Start the Flask web interface."""
    setup_logging()

    from companion.web import create_app

    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
