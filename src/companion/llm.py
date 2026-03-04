"""Ollama LLM streaming client."""

from __future__ import annotations

import json
import logging
from typing import Generator

import httpx

from companion.config import OllamaConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """HTTP client for the Ollama REST API."""

    def __init__(self, config: OllamaConfig) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._model = config.model
        self._temperature = config.temperature
        self._context_window = config.context_window
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=5.0, read=120.0, write=10.0, pool=5.0
            )
        )

    def stream_chat(
        self, messages: list[dict]
    ) -> Generator[str, None, None]:
        """Stream chat completions from Ollama, yielding text chunks."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self._temperature,
                "num_ctx": self._context_window,
            },
        }

        try:
            with self._client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self._base_url)
            yield "[Error: Ollama is not running. Start it with ollama serve.]"
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama HTTP %s: %s", exc.response.status_code, exc)
            yield f"[Error: Ollama returned status {exc.response.status_code}.]"
        except httpx.RequestError as exc:
            logger.error("Ollama request error: %s", exc)
            yield "[Error: Could not reach Ollama. Check your connection.]"

    def summarise(
        self, conversation_text: str, existing_summary: str | None = None
    ) -> str:
        """Ask the model to produce a concise rolling summary of a conversation.

        When *existing_summary* is provided, the LLM merges it with the new
        messages so that historical context is preserved across cycles.
        """
        if existing_summary:
            prompt = (
                "Below is an existing summary of an earlier conversation, followed by "
                "new conversation messages. Write an updated 3-5 sentence factual "
                "summary in third person that incorporates both the existing summary "
                "and the new messages. Focus on key topics discussed, decisions made, "
                "and any personal details shared.\n\n"
                f"[Existing summary]:\n{existing_summary}\n\n"
                f"[New messages]:\n{conversation_text}"
            )
        else:
            prompt = (
                "Write a 3-5 sentence factual summary of the following "
                "conversation in third person. Focus on key topics discussed, "
                "decisions made, and any personal details shared.\n\n"
                f"{conversation_text}"
            )
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": self._context_window,
            },
        }

        try:
            response = self._client.post(
                f"{self._base_url}/api/chat", json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except httpx.HTTPError as exc:
            logger.error("Summarisation failed: %s", exc)
            return ""

    def check_connection(self) -> bool:
        """Return ``True`` if Ollama is reachable."""
        try:
            resp = self._client.get(f"{self._base_url}/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
        logger.info("Ollama client closed.")
