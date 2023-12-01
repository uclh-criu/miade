"""Loggers
"""
import logging


def add_handlers(log):
    if len(log.handlers) == 0:
        formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(name)s.%(funcName)s(): %(message)s")

        fh = logging.FileHandler("miade.log")
        ch = logging.StreamHandler()

        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        log.addHandler(fh)
        log.addHandler(ch)

    return log
