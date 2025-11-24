"""
Defines logger
"""

import sys
import logging


# see here for explanation: https://docs.python.org/3/howto/logging.html


def attach_stdout_handler():
    """Attach a handler to stdout if not already present."""

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            return

    h = logging.StreamHandler(stream=sys.stdout)
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)

def unattach_stdout_handler():
    """Remove stdout handler"""
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            logger.removeHandler(handler)

class FileLogging:
    """Context manager to temporarily attach a file logger."""

    def __init__(self, filename):
        self.filename = filename
        self._handler = None

    def __enter__(self):
        # Check if already attached
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == self.filename:
                self._handler = handler
                return

        # Attach new handler
        h = logging.FileHandler(self.filename, mode='w')
        h.setLevel(logging.DEBUG)
        h.setFormatter(formatter)
        logger.addHandler(h)
        self._handler = h

    def __exit__(self, *args):
        if self._handler:
            logger.removeHandler(self._handler)
            self._handler.close()


# create logger
logger = logging.getLogger('fbpinns')
logger.setLevel(logging.INFO)

# make sure the logger doesn't propagate to its parents (standalone logger)
logger.propagate = False

# create formatter
formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

# set stdout handler
attach_stdout_handler()


if __name__ == "__main__":

    logger.debug("hello world")
    logger.setLevel("DEBUG")
    logger.debug("hello world")
    logger.info("hello world")
    unattach_stdout_handler()
    with FileLogging("logger.txt"):
        logger.info("hello world")
    attach_stdout_handler()