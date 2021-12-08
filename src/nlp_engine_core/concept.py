class Concept(object):
    """docstring for Concept."""

    def __init__(self, id: str, name: str):
        self.name = name
        self.id = id

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return (self.__dict__ == other.__dict__)
