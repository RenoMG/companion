"""Flask web application — serves a chat UI and exposes REST/SSE endpoints."""

from __future__ import annotations

import io
import json
import logging
import threading
from pathlib import Path

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
    _stt_lock = threading.Lock()
    _tts_lock = threading.Lock()
    _summarising = threading.Event()

    def _get(key: str):
        return state.get(key)

    @app.before_request
    def _ensure_initialised():
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

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        return send_from_directory(str(template_dir), "index.html")

    @app.route("/api/config")
    def get_config():
        return jsonify({"companion_name": config.companion_name})

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

        try:
            audio_buf = io.BytesIO(audio_file.read())
            audio, _sample_rate = sf.read(audio_buf, dtype="float32")
        except Exception as exc:
            logger.error("Failed to read audio: %s", exc)
            return jsonify({"error": "Audio conversion failed"}), 500

        # Ensure mono.
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if audio.size == 0:
            return jsonify({"text": ""})

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
