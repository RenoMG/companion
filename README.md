# Companion

Your local AI voice friend — everything runs on your machine, no cloud APIs required.

## Architecture

```
┌───────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│ Microphone │ ──▶ │ Whisper STT  │ ──▶ │  Ollama LLM  │ ──▶ │ Kokoro TTS│ ──▶ Speakers
│ sounddevice│     │ faster-whisper│     │  httpx client│     │  KPipeline│
└───────────┘     └──────────────┘     └──────────────┘     └───────────┘
                                              │
                                       ┌──────┴──────┐
                                       │ SQLite Memory│
                                       │  messages    │
                                       │  summaries   │
                                       │  facts       │
                                       └─────────────┘
```

## Stack

| Component | Library          | Notes                                 |
|-----------|------------------|---------------------------------------|
| STT       | faster-whisper   | GPU-accelerated Whisper, CTranslate2  |
| LLM       | Ollama (httpx)   | Streaming HTTP client, any local model|
| TTS       | Kokoro           | 82M params, Apache 2.0, 24kHz output  |
| Audio I/O | sounddevice      | PortAudio bindings for mic + speakers  |
| Memory    | sqlite3 (stdlib) | Messages, summaries, and user facts    |

## Requirements

- **Python** 3.11+
- **NVIDIA GPU** with CUDA (recommended) — CPU-only mode also works
- **Ollama** installed and running (`ollama serve`) with a model that supports tool calling
- **PortAudio** system library (`sudo apt install libportaudio2` on Debian/Ubuntu)
- **espeak-ng** (required by Kokoro: `sudo apt install espeak-ng`)

## Installation

```bash
# Clone the repo
git clone https://github.com/RenoMG/companion.git
cd companion

# Create a virtual environment and install with UV
uv sync

# For GPU acceleration, ensure PyTorch with CUDA is installed:
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Copy and edit config
cp config.example.yaml config.yaml

# Pull an Ollama model (must support tool calling)
ollama pull llama3
```

## Usage

```bash
# Start the companion
uv run python main.py

# Or, if installed as a package:
uv run companion
```

Speak into your microphone — Companion will listen, think, and respond through
your speakers. Say **"exit"**, **"quit"**, **"goodbye"**, or **"bye"** to end
the session.

## Configuration

Copy `config.example.yaml` to `config.yaml`. All fields are optional — omitted
values use sensible defaults.

### Whisper Model Sizes

| Size   | Params | VRAM (fp16) | Relative Speed | Accuracy |
|--------|--------|-------------|----------------|----------|
| tiny   | 39M    | ~75 MB      | ~10x           | Fair     |
| base   | 74M    | ~150 MB     | ~7x            | Good     |
| small  | 244M   | ~500 MB     | ~4x            | Better   |
| medium | 769M   | ~1.5 GB     | ~2x            | Great    |
| large  | 1550M  | ~3.0 GB     | 1x (baseline)  | Best     |

### Kokoro Voices

| Voice        | Description     |
|--------------|-----------------|
| `af_sky`     | American Female |
| `af_bella`   | American Female |
| `af_sarah`   | American Female |
| `af_nicole`  | American Female |
| `am_adam`    | American Male   |
| `am_michael` | American Male   |
| `bf_emma`    | British Female  |
| `bf_isabella`| British Female  |
| `bm_george`  | British Male    |
| `bm_lewis`   | British Male    |

## Memory System

Companion remembers things about you across sessions using three SQLite tables:

| Table      | Purpose                                           |
|------------|---------------------------------------------------|
| `messages` | Current session conversation (wiped on exit)      |
| `summaries`| Rolling LLM-generated summaries across sessions   |
| `facts`    | Key-value pairs the LLM stores about the user     |

### Session lifecycle

1. **Start** — The session begins with zero messages. The LLM receives the
   latest summary and stored facts as context, so it remembers previous
   conversations.
2. **During** — Messages accumulate normally. When the count exceeds
   `summary_threshold` (default 50), a background thread summarises older
   messages and deletes them, keeping the most recent `max_context_messages`
   (default 20) in the database.
3. **Exit** — All remaining messages are summarised and the messages table is
   wiped. The next session starts clean.

### Fact storage

The LLM uses Ollama tool calling to store and delete facts about you
automatically. When you share personal details (name, preferences, hobbies,
etc.), the model calls `store_fact` to save them. When you say something is
no longer true, it calls `delete_fact`. Facts persist across sessions and
appear in the LLM context so the model naturally recalls them.

### LLM context (in order)

1. System prompt
2. Latest conversation summary (if any)
3. Stored user facts (if any)
4. Most recent N messages (configurable, default 20)

## Project Structure

```
companion/
├── main.py                  # Entry point
├── config.example.yaml      # Example configuration
├── pyproject.toml            # Project metadata & dependencies
├── README.md
├── data/                    # Auto-created at runtime
│   └── companion.db         # SQLite database
└── src/companion/
    ├── __init__.py
    ├── app.py               # Main event loop & orchestration
    ├── config.py            # Config dataclasses & YAML loader
    ├── stt.py               # Speech-to-text (faster-whisper)
    ├── llm.py               # Ollama streaming client + tool definitions
    ├── tts.py               # Kokoro TTS with playback queue
    └── memory.py            # SQLite memory store
```

## GPU Notes

Companion is designed to share a GPU with other applications (e.g. a game).

**Approximate VRAM usage:**

| Component       | VRAM       |
|-----------------|------------|
| Kokoro TTS      | ~200 MB    |
| Whisper (base)  | ~300 MB    |
| **Total pipeline** | **~500 MB** |

Ollama manages its own GPU memory separately. If you're running a large Ollama
model alongside a game, you can limit how many GPU layers Ollama uses:

```bash
OLLAMA_NUM_GPU=20 ollama serve
```

Lower values offload more layers to CPU (slower responses but less VRAM).

## License

MIT
