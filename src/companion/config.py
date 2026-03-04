"""Configuration dataclasses and YAML loader for Companion."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly, curious, and helpful voice companion. "
    "Keep your responses concise and conversational — typically one to three sentences. "
    "You speak naturally, like a close friend. Avoid lists, markdown formatting, or "
    "overly formal language. Ask follow-up questions to keep the conversation flowing.\n\n"
    "You have tools for remembering facts about the user. When the user shares "
    "personal information (name, preferences, occupation, hobbies, family, "
    "important dates, etc.), use the store_fact tool to save it. When the user "
    "says something is no longer true or asks you to forget something, use the "
    "delete_fact tool. Do not mention the tools to the user — just naturally "
    "remember and recall things."
)


@dataclass
class OllamaConfig:
    """Settings for the Ollama LLM backend."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    temperature: float = 0.7
    context_window: int = 4096


@dataclass
class WhisperConfig:
    """Settings for faster-whisper speech-to-text."""

    model_size: str = "base"
    device: str = "cuda"
    language: str = "en"
    energy_threshold: int = 300
    pause_duration: float = 1.2


@dataclass
class KokoroConfig:
    """Settings for Kokoro text-to-speech."""

    voice: str = "af_sky"
    speed: float = 1.0
    device: str = "cuda"


@dataclass
class MemoryConfig:
    """Settings for the SQLite memory store."""

    db_path: str = "data/companion.db"
    max_context_messages: int = 20
    summary_threshold: int = 50


@dataclass
class CompanionConfig:
    """Top-level application configuration."""

    companion_name: str = "Companion"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    kokoro: KokoroConfig = field(default_factory=KokoroConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def _apply_dict_to_dataclass(dc: Any, data: dict[str, Any]) -> None:
    """Apply dictionary values to a dataclass, ignoring unknown keys."""
    for key, value in data.items():
        if hasattr(dc, key):
            setattr(dc, key, value)


def load_config(path: str = "config.yaml") -> CompanionConfig:
    """Load configuration from a YAML file.

    Returns sensible defaults if the file does not exist.
    """
    config_path = Path(path)

    if not config_path.exists():
        logger.warning(
            "Config file '%s' not found — using defaults.", config_path
        )
        return CompanionConfig()

    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] | None = yaml.safe_load(f)

    if not raw:
        logger.warning("Config file '%s' is empty — using defaults.", config_path)
        return CompanionConfig()

    config = CompanionConfig()

    # Map nested sections to their sub-dataclasses
    nested_map: dict[str, Any] = {
        "ollama": config.ollama,
        "whisper": config.whisper,
        "kokoro": config.kokoro,
        "memory": config.memory,
    }

    for key, value in raw.items():
        if key in nested_map and isinstance(value, dict):
            _apply_dict_to_dataclass(nested_map[key], value)
        elif hasattr(config, key):
            setattr(config, key, value)

    return config
