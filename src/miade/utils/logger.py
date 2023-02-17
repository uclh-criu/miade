"""Loggers
"""
import logging


def add_handlers(log):
    if len(log.handlers) == 0:
        formatter = logging.Formatter(fmt="[%(asctime)s] %(process)d/%(levelname)s/%(name)s: %(message)s")

        fh = logging.FileHandler("miade.log")
        ch = logging.StreamHandler()

        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        log.addHandler(fh)
        log.addHandler(ch)

    return log
