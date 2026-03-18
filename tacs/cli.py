from __future__ import annotations

import logging

import click

from tacs.config import config


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


@click.group()
def cli() -> None:
    """Tool Augmented Conversation Simulator (TACS)."""
    _setup_logging()
