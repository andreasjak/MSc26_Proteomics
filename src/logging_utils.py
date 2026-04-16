"""
logging_utils.py
----------------
Shared logging configuration for the MSc26 proteomics project.

Provides a single ``setup_logging`` function used by all pipeline scripts.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(
    save_results: bool = False,
    log_subdir: str = "",
    script_name: str = "script",
) -> logging.Logger:
    """Configure and return a project logger.

    Parameters
    ----------
    save_results : bool, optional
        When ``False`` (default), log to the terminal via a
        ``StreamHandler``. When ``True``, log to a timestamped file
        under ``logs/<log_subdir>/`` and suppress terminal output.
    log_subdir : str, optional
        Subdirectory under ``logs/`` for the log file. Only used when
        *save_results* is ``True``.
    script_name : str, optional
        Name used for ``logging.getLogger`` **and** as the log-file
        prefix (default: ``"script"``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if save_results:
        log_dir = Path("logs") / log_subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{script_name}_{timestamp}.log"
        handler: logging.Handler = logging.FileHandler(log_path)
    else:
        handler = logging.StreamHandler()

    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger