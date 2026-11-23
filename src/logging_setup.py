"""File logger: everything the system does or rejects is appended to logs/system.log."""

import logging
import os

LOG_PATH = os.path.join("logs", "system.log")


def get_logger(name: str = "recommender") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
