"""Companion — entry point."""

import sys

if __name__ == "__main__":
    if "--web" in sys.argv:
        from companion.app import main_web

        main_web()
    else:
        from companion.app import main

        main()
