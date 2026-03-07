"""Flask web application for the web-first Companion interface."""

from __future__ import annotations

import atexit
import contextlib
import io
import json
import logging
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, render_template, request

from companion.config import (
    CompanionConfig,
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)
from companion.llm import FACT_TOOLS, OllamaClient
from companion.memory import MemoryStore
from companion.stt import SpeechToText
from companion.tts import TextToSpeech

logger = logging.getLogger(__name__)

try:
    import av
except ModuleNotFoundError:
    av = None


def _resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample mono audio with linear interpolation."""
    if from_rate == to_rate or audio.size == 0:
        return audio

    duration = audio.shape[0] / from_rate
    target_length = max(1, int(round(duration * to_rate)))
    src_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def _decode_audio_with_pyav(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode compressed audio bytes (for example webm/opus) to mono float32."""
    if av is None:
        raise ValueError("PyAV is not installed")

    with av.open(io.BytesIO(audio_bytes), mode="r") as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("No audio stream found")

        chunks: list[np.ndarray] = []
        sample_rate = int(stream.codec_context.sample_rate or 0)

        for frame in container.decode(stream):
            frame_array = frame.to_ndarray().astype(np.float32)
            mono = frame_array if frame_array.ndim == 1 else frame_array.mean(axis=0)
            chunks.append(mono)
            if sample_rate <= 0:
                sample_rate = int(frame.sample_rate or 0)

        if not chunks:
            raise ValueError("Audio stream had no decodable frames")

        audio = np.concatenate(chunks)
        fmt = stream.codec_context.format
        if fmt is not None and fmt.name and fmt.name[0] in ("s", "u"):
            bits = max(int(fmt.bits or 16), 1)
            max_abs = float(2 ** (bits - 1))
            if max_abs > 0:
                audio /= max_abs

        if sample_rate <= 0:
            raise ValueError("Could not determine sample rate")

        return np.clip(audio, -1.0, 1.0), sample_rate


def _execute_tool_calls(tool_calls: list, memory: MemoryStore) -> list[dict]:
    """Execute LLM memory tool calls and return tool messages for the follow-up."""
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
                memory.set_fact(key, value)
                result_text = f"Stored fact: {key} = {value}"
                logger.info("Tool call: store_fact(%s, %s)", key, value)
            else:
                result_text = "Error: key and value are required"
        elif func_name == "delete_fact":
            key = args.get("key", "")
            if key:
                memory.delete_fact(key)
                result_text = f"Deleted fact: {key}"
                logger.info("Tool call: delete_fact(%s)", key)
            else:
                result_text = "Error: key is required"
        else:
            result_text = f"Unknown tool: {func_name}"

        results.append({"role": "tool", "content": result_text})
    return results


def _summarise_messages(
    memory: MemoryStore,
    llm: OllamaClient,
    keep_recent: int | None,
) -> None:
    """Summarise messages into rolling memory."""
    if keep_recent is None:
        messages = memory.get_all_messages()
    else:
        messages = memory.get_messages_for_summary(keep_recent)

    if not messages:
        return

    existing_summary = memory.get_latest_summary()
    conversation_text = "\n".join(f"{m.role.title()}: {m.content}" for m in messages)
    summary = llm.summarise(conversation_text, existing_summary)
    if summary:
        memory.add_summary(summary)

    if keep_recent is None:
        memory.clear_messages()
    else:
        memory.delete_messages_up_to(messages[-1].id)


class RuntimeState:
    """Mutable application state shared by the Flask routes."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self._config_path = config_path
        self._lock = threading.RLock()
        self._operation_lock = threading.RLock()
        self._stt_lock = threading.Lock()
        self._tts_lock = threading.Lock()
        self._summarising = threading.Event()
        self._started_at = time.time()
        self._phase = "idle"
        self._phase_changed_at = self._started_at
        self._last_user_message = ""
        self._last_response = ""
        self._session_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }

        self._config = load_config(config_path)
        self._memory = MemoryStore(self._config.memory)
        self._llm = OllamaClient(self._config.ollama)
        self._llm_available = self._llm.check_connection()
        self._stt: SpeechToText | None = None
        self._tts: TextToSpeech | None = None

        self._cleanup_previous_session()

    @contextlib.contextmanager
    def operation(self) -> Iterator[None]:
        """Serialize mutating operations and config swaps."""
        with self._operation_lock:
            yield

    def get_config(self) -> CompanionConfig:
        """Return the live configuration object."""
        with self._lock:
            return self._config

    def get_public_config(self) -> dict:
        """Return UI-facing config fields."""
        with self._lock:
            config = self._config
            return {
                "companion_name": config.companion_name,
                "audio_enabled_default": config.web.audio_enabled_default,
                "voice_chat_default": config.web.voice_chat_default,
            }

    def get_settings(self) -> dict:
        """Return the full editable settings payload."""
        with self._lock:
            return config_to_dict(self._config)

    def get_history(self) -> list[dict]:
        """Return the current session history."""
        memory = self._memory
        return [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in memory.get_all_messages()
        ]

    def get_stats(self) -> dict:
        """Return runtime and usage stats for the dashboard."""
        with self._lock:
            config = self._config
            phase = self._phase
            phase_changed_at = self._phase_changed_at
            last_user_message = self._last_user_message
            last_response = self._last_response
            llm_available = self._llm_available
            session_usage = dict(self._session_usage)

        memory = self._memory
        lifetime_usage = memory.get_usage_totals()
        return {
            "phase": phase,
            "phase_changed_at": phase_changed_at,
            "started_at": self._started_at,
            "companion_name": config.companion_name,
            "current_model": config.ollama.model,
            "llm_base_url": config.ollama.base_url,
            "llm_available": llm_available,
            "session": session_usage,
            "lifetime": lifetime_usage,
            "memory": {
                "messages": memory.get_message_count(),
                "facts": memory.get_fact_count(),
                "summaries": memory.get_summary_count(),
            },
            "speech": {
                "whisper_model": config.whisper.model_size,
                "voice": config.kokoro.voice,
                "voice_speed": config.kokoro.speed,
            },
            "last_user_message": last_user_message,
            "last_response_preview": last_response[:240],
        }

    def set_phase(self, phase: str) -> None:
        """Update the dashboard activity phase."""
        with self._lock:
            self._phase = phase
            self._phase_changed_at = time.time()

    def note_turn(self, user_text: str, response_text: str) -> None:
        """Store previews for the dashboard."""
        with self._lock:
            self._last_user_message = user_text
            self._last_response = response_text

    def record_usage(self, usage: dict) -> None:
        """Store session and lifetime usage counters."""
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)

        with self._lock:
            self._session_usage["prompt_tokens"] += prompt_tokens
            self._session_usage["completion_tokens"] += completion_tokens
            self._session_usage["total_tokens"] += prompt_tokens + completion_tokens
            self._session_usage["requests"] += 1

        self._memory.add_usage(prompt_tokens, completion_tokens, requests=1)

    def ensure_stt(self) -> SpeechToText:
        """Load the speech-to-text model on first use."""
        with self._stt_lock:
            if self._stt is None:
                self._stt = SpeechToText(self.get_config().whisper)
            return self._stt

    def ensure_tts(self) -> TextToSpeech:
        """Load the text-to-speech pipeline on first use."""
        with self._tts_lock:
            if self._tts is None:
                self._tts = TextToSpeech(self.get_config().kokoro)
            return self._tts

    def maybe_start_summarisation(self) -> None:
        """Kick off a background summarisation pass if needed."""
        with self._lock:
            if (
                self._summarising.is_set()
                or not self._memory.should_summarise
                or not self._llm_available
            ):
                return
            memory = self._memory
            llm = self._llm
            keep_recent = self._config.memory.max_context_messages
            self._summarising.set()

        thread = threading.Thread(
            target=self._run_summarisation,
            args=(memory, llm, keep_recent),
            daemon=True,
        )
        thread.start()

    def apply_settings(self, raw: dict) -> dict:
        """Persist and hot-apply a new configuration."""
        new_config = config_from_dict(raw)
        new_memory: MemoryStore | None = None

        with self._lock:
            current_config = self._config

        if new_config.memory.db_path != current_config.memory.db_path:
            new_memory = MemoryStore(new_config.memory)

        new_llm = OllamaClient(new_config.ollama)
        llm_available = new_llm.check_connection()
        save_config(new_config, self._config_path)

        with self.operation():
            self._wait_for_summarisation()
            with self._lock:
                old_llm = self._llm
                old_memory = self._memory
                old_tts = self._tts

                self._config = new_config
                self._llm = new_llm
                self._llm_available = llm_available

                if new_memory is None:
                    self._memory.update_config(new_config.memory)
                else:
                    self._memory = new_memory

                self._stt = None
                self._tts = None

            if old_tts is not None:
                old_tts.stop()
            if new_memory is not None:
                old_memory.close()
            old_llm.close()

        return {
            "config": config_to_dict(new_config),
            "stats": self.get_stats(),
            "llm_available": llm_available,
        }

    def cleanup(self) -> None:
        """Summarise session messages and close open resources."""
        with self.operation():
            self._wait_for_summarisation()

            memory = self._memory
            llm = self._llm
            tts_engine = self._tts
            llm_available = self._llm_available

            if llm_available:
                try:
                    _summarise_messages(memory, llm, keep_recent=None)
                except Exception:
                    logger.exception("Shutdown summarisation failed.")
                    try:
                        memory.clear_messages()
                    except Exception:
                        logger.exception("Failed to clear messages on shutdown.")
            else:
                memory.clear_messages()

            llm.close()
            if tts_engine is not None:
                tts_engine.stop()
            memory.close()
            logger.info("Web app cleanup complete.")

    def _run_summarisation(
        self, memory: MemoryStore, llm: OllamaClient, keep_recent: int
    ) -> None:
        try:
            _summarise_messages(memory, llm, keep_recent=keep_recent)
        except Exception:
            logger.exception("Background summarisation failed.")
        finally:
            self._summarising.clear()

    def _wait_for_summarisation(self) -> None:
        if not self._summarising.is_set():
            return

        start = time.monotonic()
        while self._summarising.is_set() and (time.monotonic() - start) < 30:
            time.sleep(0.2)

    def _cleanup_previous_session(self) -> None:
        messages = self._memory.get_all_messages()
        if not messages:
            return

        logger.info(
            "Found %d leftover messages from previous session, summarising ...",
            len(messages),
        )
        if not self._llm_available:
            self._memory.clear_messages()
            return
        try:
            _summarise_messages(self._memory, self._llm, keep_recent=None)
        except Exception:
            logger.exception("Failed to summarise leftover messages on startup.")
            self._memory.clear_messages()


def create_app(config: CompanionConfig | None = None) -> Flask:
    """Application factory."""
    config_path = "config.yaml"
    runtime = RuntimeState(config_path=config_path)
    if config is not None:
        runtime.apply_settings(config_to_dict(config))

    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/bootstrap")
    def bootstrap():
        return jsonify(
            {
                "config": runtime.get_public_config(),
                "history": runtime.get_history(),
                "stats": runtime.get_stats(),
                "settings": runtime.get_settings(),
            }
        )

    @app.route("/api/config")
    def get_config():
        return jsonify(runtime.get_public_config())

    @app.route("/api/history")
    def history():
        return jsonify(runtime.get_history())

    @app.route("/api/stats")
    def stats():
        return jsonify(runtime.get_stats())

    @app.route("/api/settings", methods=["GET", "POST"])
    def settings():
        if request.method == "GET":
            return jsonify(runtime.get_settings())

        payload = request.get_json(force=True) or {}
        result = runtime.apply_settings(payload)
        return jsonify(result)

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json(force=True) or {}
        user_text = data.get("message", "").strip()
        if not user_text:
            return jsonify({"error": "Empty message"}), 400

        def generate():
            runtime.set_phase("thinking")
            yield f"data: {json.dumps({'state': 'thinking'})}\n\n"

            with runtime.operation():
                memory = runtime._memory
                llm = runtime._llm
                config = runtime.get_config()

                memory.add_message("user", user_text)
                context = memory.build_context(config.system_prompt)

                full_response = ""
                tool_calls: list = []
                usage_records: list[dict] = []
                has_started_response = False

                try:
                    usage = {}
                    for chunk in llm.stream_chat(
                        context,
                        tools=FACT_TOOLS,
                        tool_calls_out=tool_calls,
                        usage_out=usage,
                    ):
                        if not has_started_response:
                            has_started_response = True
                            runtime.set_phase("responding")
                            yield f"data: {json.dumps({'state': 'responding'})}\n\n"
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    usage_records.append(usage)

                    if tool_calls:
                        tool_results = _execute_tool_calls(tool_calls, memory)
                        followup_ctx = list(context)
                        followup_ctx.append(
                            {
                                "role": "assistant",
                                "content": full_response,
                                "tool_calls": tool_calls,
                            }
                        )
                        followup_ctx.extend(tool_results)

                        full_response = ""
                        usage = {}
                        for chunk in llm.stream_chat(
                            followup_ctx,
                            usage_out=usage,
                        ):
                            if not has_started_response:
                                has_started_response = True
                                runtime.set_phase("responding")
                                yield f"data: {json.dumps({'state': 'responding'})}\n\n"
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                        usage_records.append(usage)

                    memory.add_message("assistant", full_response)
                    runtime.note_turn(user_text, full_response)

                    merged_usage = {
                        "model": config.ollama.model,
                        "prompt_tokens": sum(
                            int(item.get("prompt_tokens") or 0) for item in usage_records
                        ),
                        "completion_tokens": sum(
                            int(item.get("completion_tokens") or 0)
                            for item in usage_records
                        ),
                    }
                    runtime.record_usage(merged_usage)
                    runtime.maybe_start_summarisation()
                    runtime.set_phase("idle")
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "done": True,
                                "stats": runtime.get_stats(),
                                "usage": merged_usage,
                            }
                        )
                        + "\n\n"
                    )
                except Exception as exc:
                    logger.exception("Chat request failed.")
                    runtime.set_phase("idle")
                    yield (
                        "data: "
                        + json.dumps({"error": str(exc), "done": True, "stats": runtime.get_stats()})
                        + "\n\n"
                    )

        return Response(generate(), content_type="text/event-stream")

    @app.route("/api/transcribe", methods=["POST"])
    def transcribe():
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_bytes = request.files["audio"].read()
        try:
            audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:
            logger.warning("soundfile decode failed, trying PyAV fallback: %s", exc)
            try:
                audio, sample_rate = _decode_audio_with_pyav(audio_bytes)
            except Exception as fallback_exc:
                logger.error("Failed to read audio: %s", fallback_exc)
                return jsonify({"error": "Audio conversion failed"}), 500

        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = audio.mean(axis=1)

        if audio.size == 0:
            return jsonify({"text": ""})

        audio = _resample_audio(audio, int(sample_rate), 16000)

        try:
            with runtime.operation():
                runtime.set_phase("listening")
                stt = runtime.ensure_stt()
                text = stt.transcribe_audio(audio)
        finally:
            runtime.set_phase("idle")

        return jsonify({"text": text or "", "stats": runtime.get_stats()})

    @app.route("/api/tts", methods=["POST"])
    def tts():
        data = request.get_json(force=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty text"}), 400

        try:
            with runtime.operation():
                runtime.set_phase("speaking")
                tts_engine = runtime.ensure_tts()
                wav_bytes = tts_engine.synthesize_wav(text)
        finally:
            runtime.set_phase("idle")

        if wav_bytes is None:
            return jsonify({"error": "Synthesis produced no audio"}), 500

        return Response(wav_bytes, mimetype="audio/wav")

    atexit.register(runtime.cleanup)

    return app
