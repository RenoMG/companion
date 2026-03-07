"""Application entrypoints for the web-first Companion runtime."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging() -> None:
    """Configure file-based application logging."""
    log_dir = Path("Logs")
    log_dir.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    if root.handlers:
        return

    file_handler = logging.FileHandler(log_dir / "companion.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    error_handler = logging.FileHandler(log_dir / "error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)

    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(error_handler)


def main() -> None:
    """Start the Flask web interface."""
    setup_logging()

    from companion.web import create_app

    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


def main_web() -> None:
    """Backward-compatible alias for the web entrypoint."""
    main()
