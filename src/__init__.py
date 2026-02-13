"""
Cognitive Soccer ML: machine learning pipeline for cognitive and performance data.
"""

from pathlib import Path


def main() -> None:
    """
    Entry point for the `vaila` console script.
    Run from the project root so that data/, best_param/, etc. are found.
    """
    from .pipeline import run

    run(Path.cwd())
