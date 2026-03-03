"""Companion — entry point."""

import logging
import signal
import sys

from companion.app import CompanionApp
from companion.config import load_config


def main() -> None:
    """Load configuration and start the voice companion."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

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


if __name__ == "__main__":
    main()
