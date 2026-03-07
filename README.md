# Companion

Local web-first AI companion: browser chat and voice capture on top of Whisper STT, Ollama, Kokoro TTS, and SQLite memory.

## What Changed

- The project now launches into a web dashboard by default.
- Terminal-driven conversation flow has been removed.
- All runtime configuration is editable from the in-app settings page.
- Saving settings updates the live server without restarting it.

## Stack

| Component | Library | Role |
|-----------|---------|------|
| Web app | Flask | Dashboard, APIs, streaming chat |
| STT | faster-whisper | Browser audio transcription |
| LLM | Ollama via httpx | Local streaming chat completions |
| TTS | Kokoro | Browser-played voice synthesis |
| Memory | sqlite3 | Messages, summaries, facts, token totals |
| Audio decode | PyAV + soundfile | Upload/browser audio conversion |

## Requirements

- Python 3.11+
- Ollama installed and running at your configured URL
- A local Ollama model that supports tool calling
- `espeak-ng` installed for Kokoro on Linux
- CUDA is optional; CPU mode works if configured

## Install

```bash
uv sync
cp config.example.yaml config.yaml
ollama pull llama3
```

## Run

```bash
uv run python main.py
```

Then open `http://localhost:5000`.

## Settings

Every config field from `config.yaml` is exposed in the browser settings page:

- `companion_name`
- `system_prompt`
- `ollama.*`
- `whisper.*`
- `kokoro.*`
- `memory.*`
- `web.*`

When you click save, the app writes the new values to `config.yaml` and applies them live. LLM and memory changes affect the next request immediately. STT and TTS changes are applied the next time those systems are used.

## Memory And Stats

The SQLite database stores:

- Current session messages
- Rolling conversation summaries
- User facts collected through tool calls
- Lifetime token/request totals

The dashboard shows:

- Current model and backend status
- Session token usage
- Lifetime token usage
- Message/fact/summary counts
- Current runtime phase

## Project Structure

```text
main.py
config.example.yaml
src/companion/app.py
src/companion/config.py
src/companion/web.py
src/companion/stt.py
src/companion/tts.py
src/companion/llm.py
src/companion/memory.py
src/companion/templates/index.html
```

## License

MIT
