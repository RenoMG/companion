"""Flask web application — serves a chat UI and exposes REST/SSE endpoints."""

from __future__ import annotations

import io
import json
import logging
import threading
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request, send_from_directory

from companion.config import CompanionConfig, load_config
from companion.llm import FACT_TOOLS, OllamaClient
from companion.memory import MemoryStore
from companion.stt import SpeechToText
from companion.tts import TextToSpeech

logger = logging.getLogger(__name__)


def create_app(config: CompanionConfig | None = None) -> Flask:
    """Application factory — initialise components and register routes."""

    if config is None:
        config = load_config()

    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

    # ------------------------------------------------------------------
    # Shared state (initialised lazily on first request)
    # ------------------------------------------------------------------
    state: dict = {}
    _init_lock = threading.Lock()
    _stt_lock = threading.Lock()
    _tts_lock = threading.Lock()
    _summarising = threading.Event()

    def _get(key: str):
        return state.get(key)

    @app.before_request
    def _ensure_initialised():
        if state:
            return
        with _init_lock:
            if state:
                return
            logger.info("Initialising backend components …")
            state["config"] = config
            state["memory"] = MemoryStore(config.memory)
            state["llm"] = OllamaClient(config.ollama)

            if not state["llm"].check_connection():
                logger.error("Cannot reach Ollama at %s", config.ollama.base_url)

            logger.info("Loading speech-to-text model …")
            state["stt"] = SpeechToText(config.whisper)
            logger.info("Loading text-to-speech model …")
            state["tts"] = TextToSpeech(config.kokoro, playback=False)
            logger.info("All components initialised.")

    def _resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample mono audio with linear interpolation."""
        if from_rate == to_rate:
            return audio
        if audio.size == 0:
            return audio

        duration = audio.shape[0] / from_rate
        target_length = max(1, int(round(duration * to_rate)))
        src_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        dst_x = np.linspace(0.0, duration, num=target_length, endpoint=False)
        return np.interp(dst_x, src_x, audio).astype(np.float32)

    def _decode_audio_with_pyav(audio_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode compressed audio bytes (e.g. webm/opus) to float32 mono audio."""
        with av.open(io.BytesIO(audio_bytes), mode="r") as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise ValueError("No audio stream found")

            chunks: list[np.ndarray] = []
            sample_rate = int(stream.codec_context.sample_rate or 0)

            for frame in container.decode(stream):
                frame_array = frame.to_ndarray().astype(np.float32)
                if frame_array.ndim == 1:
                    mono = frame_array
                else:
                    mono = frame_array.mean(axis=0)
                chunks.append(mono)
                if sample_rate <= 0:
                    sample_rate = int(frame.sample_rate or 0)

            if not chunks:
                raise ValueError("Audio stream had no decodable frames")

            audio = np.concatenate(chunks)
            # Normalize integer PCM formats (s16, s32, u8, etc.) to [-1.0, 1.0].
            # Float formats (flt, fltp, dbl, dblp) are already in that range.
            fmt = stream.codec_context.format
            if fmt is not None and fmt.name and fmt.name[0] in ("s", "u"):
                bits = max(int(fmt.bits or 16), 1)
                max_abs = float(2 ** (bits - 1))
                if max_abs > 0:
                    audio /= max_abs

            if sample_rate <= 0:
                raise ValueError("Could not determine sample rate")

            return np.clip(audio, -1.0, 1.0), sample_rate

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        return send_from_directory(str(template_dir), "index.html")

    @app.route("/api/config")
    def get_config():
        return jsonify({
            "companion_name": config.companion_name,
            "audio_enabled_default": config.web.audio_enabled_default,
            "voice_chat_default": config.web.voice_chat_default,
        })

    @app.route("/api/history")
    def history():
        memory: MemoryStore = _get("memory")
        messages = memory.get_recent_messages()
        return jsonify([
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in messages
        ])

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json(force=True)
        user_text = data.get("message", "").strip()
        if not user_text:
            return jsonify({"error": "Empty message"}), 400

        memory: MemoryStore = _get("memory")
        llm: OllamaClient = _get("llm")

        memory.add_message("user", user_text)
        context = memory.build_context(config.system_prompt)

        def generate():
            full_response = ""
            tool_calls: list = []

            for chunk in llm.stream_chat(context, tools=FACT_TOOLS, tool_calls_out=tool_calls):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Handle tool calls (store_fact / delete_fact).
            if tool_calls:
                tool_results = _execute_tool_calls(tool_calls, memory)
                followup_ctx = list(context)
                followup_ctx.append({
                    "role": "assistant",
                    "content": full_response,
                    "tool_calls": tool_calls,
                })
                for result in tool_results:
                    followup_ctx.append(result)

                full_response = ""
                for chunk in llm.stream_chat(followup_ctx):
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            memory.add_message("assistant", full_response)

            # Trigger background summarisation if needed.
            if memory.should_summarise and not _summarising.is_set():
                _summarising.set()
                thread = threading.Thread(
                    target=_maybe_summarise,
                    args=(memory, llm, config),
                    daemon=True,
                )
                thread.start()

            yield f"data: {json.dumps({'done': True})}\n\n"

        return Response(generate(), content_type="text/event-stream")

    @app.route("/api/transcribe", methods=["POST"])
    def transcribe():
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        stt: SpeechToText = _get("stt")

        audio_bytes = audio_file.read()

        try:
            audio_buf = io.BytesIO(audio_bytes)
            audio, sample_rate = sf.read(audio_buf, dtype="float32")
        except Exception as exc:
            logger.warning("soundfile decode failed, trying PyAV fallback: %s", exc)
            try:
                audio, sample_rate = _decode_audio_with_pyav(audio_bytes)
            except Exception as fallback_exc:
                logger.error("Failed to read audio: %s", fallback_exc)
                return jsonify({"error": "Audio conversion failed"}), 500

        # Ensure mono.
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if audio.size == 0:
            return jsonify({"text": ""})

        audio = _resample_audio(audio, int(sample_rate), 16000)

        with _stt_lock:
            text = stt.transcribe_audio(audio)

        return jsonify({"text": text or ""})

    @app.route("/api/tts", methods=["POST"])
    def tts():
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty text"}), 400

        tts_engine: TextToSpeech = _get("tts")
        with _tts_lock:
            wav_bytes = tts_engine.synthesize_wav(text)

        if wav_bytes is None:
            return jsonify({"error": "Synthesis produced no audio"}), 500

        return Response(wav_bytes, mimetype="audio/wav")

    # ------------------------------------------------------------------
    # Helpers (mirrored from app.py)
    # ------------------------------------------------------------------

    def _execute_tool_calls(tool_calls: list, memory: MemoryStore) -> list[dict]:
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

    def _maybe_summarise(
        memory: MemoryStore, llm: OllamaClient, cfg: CompanionConfig
    ) -> None:
        try:
            logger.info("Background summarisation started.")
            existing_summary = memory.get_latest_summary()
            old_messages = memory.get_messages_for_summary(cfg.memory.max_context_messages)
            if not old_messages:
                return
            max_id = old_messages[-1].id
            conversation_text = "\n".join(
                f"{m.role.title()}: {m.content}" for m in old_messages
            )
            summary = llm.summarise(conversation_text, existing_summary)
            if summary:
                memory.add_summary(summary)
                memory.delete_messages_up_to(max_id)
                logger.info("Summarised; deleted messages up to id=%d.", max_id)
        finally:
            _summarising.clear()

    return app
