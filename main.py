#!/usr/bin/env python3
"""Entry point: python3 main.py [--config path/to/config.json]"""
import argparse
import os
import sys

from app import HandTrackerApp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Real-time hand tracker with customizable actions and air-cursor control.")
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "config.json"),
                         help="Path to config.json (default: ./config.json)")
    args = parser.parse_args()

    app = HandTrackerApp(args.config)
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted, shutting down.")
        sys.exit(0)


if __name__ == "__main__":
    main()
