from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def init_logging(log_level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Initialize root logger with sane defaults."""

    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file = Path(log_file).expanduser().absolute()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))

    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        handlers=handlers,
    )


__all__ = ['init_logging']
