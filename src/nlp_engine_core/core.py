from .note import Note
from .list import List

class NoteProcessor(object):
    """docstring for NoteProcessor."""

    def __init__(self, arg):
        super(NoteProcessor, self).__init__()
        self.arg = arg

    def process(note: Note, list: List = None):
        pass
