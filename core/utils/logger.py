import logging


def get_logger(name: str) -> logging.Logger:
    """Return a consistently-formatted logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger
