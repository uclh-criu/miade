from enum import Enum
from pathlib import Path

from pandas import read_csv


class Direction(Enum):
    FORWARD = 'forward'
    REVERSE = 'reverse'


class ConceptGraph:

    def __init__(self):
        self.dict = {}

    def _relationship_name_filter(self, relationship_id: str) -> str:
        if relationship_id == 116680003:
            return "parent"
        return relationship_id

    def _add_relationship(self, source: str, dest: str, type: str, direction: Direction):
        if not self.dict.get(source):
            self.dict[source] = {
                Direction.FORWARD.value: {},
                Direction.REVERSE.value: {}
            }
        if not self.dict[source][direction.value].get(type):
            self.dict[source][direction.value][type] = set()
        self.dict[source][direction.value][type].add(dest)

    def _add_relationship_pair(self, source: str, dest: str, type: str):
        self._add_relationship(source, dest, type, Direction.FORWARD)
        self._add_relationship(dest, source, type, Direction.REVERSE)

    @classmethod
    def from_snomed_snapshot(cls, filename: Path):
        graph = cls()
        relationships = read_csv(filename, sep="\t")
        for _, relationship in relationships.iterrows():
            concept = relationship['sourceId']
            relative = relationship['destinationId']
            relationship_type = graph._relationship_name_filter(relationship['typeId'])
            graph._add_relationship_pair(concept, relative, relationship_type)

        return graph

    def __str__(self):
        string = ""
        for concept, relationships in self.dict.items():
            string += f"{concept} -> {relationships['forward']}\n{' '*len(str(concept))} <- {relationships['reverse']}\n"
        return string


if __name__ == "__main__":
    graph = ConceptGraph.from_snomed_snapshot("data/snomed/SnomedCT_UKEditionRF2_PRODUCTION_20211124T000001Z/Snapshot/Terminology/sct2_Relationship_UKEDSnapshot_GB_20211124.txt")
    print(graph)
