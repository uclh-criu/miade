import logging

from .note import Note
from .concept import Concept
from .core import NoteProcessor
from .deduplicate import deduplicate
from .utils.logger import add_handlers

__all__ = [
    "Note",
    "Concept",
    "NoteProcessor",
    "deduplicate"
]

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
add_handlers(log)
