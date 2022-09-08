import logging

from .note import Note
from .concept import Concept
from .core import NoteProcessor
from .dosageprocessor import DosageProcessor
from .utils.logger import add_handlers

__all__ = ["Note", "Concept", "NoteProcessor", "DosageProcessor"]

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
add_handlers(log)
