import logging

from .note import Note
from .concept import Concept
from .core import NoteProcessor
from .utils.logger import add_handlers

__all__ = [
    "Note",
    "Concept",
    "NoteProcessor"
]

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
add_handlers(log)
