# Module for classes and methods related to SNOMED relationships

import pandas

from typing import List, Optional, Dict


class Transitive:
    def __init__(self, df: pandas.DataFrame):
        """
        Initializes a transitive closure table from a pandas DataFrame
        with columns ancestorId and descendantId.
        The dictionary maps descendantIds to ancestorIds for fast lookups.
        """
        self.ancestor_dict = {}
        self.descendant_dict = {}

        for row in df.itertuples():
            descendant = str(row.descendantId)
            ancestor = str(row.ancestorId)

            if descendant not in self.ancestor_dict:
                self.ancestor_dict[descendant] = set()

            if ancestor not in self.descendant_dict:
                self.descendant_dict[ancestor] = set()

            self.ancestor_dict[descendant].add(ancestor)
            self.descendant_dict[ancestor].add(descendant)

    def get_ancestorIds(self, descendantIds: set):
        """
        Returns a tuple containing the unique ancestors of the given descendants.
        Ensures that none of the input descendant IDs appear in the output.
        """
        ancestorIds = set()

        for descendant in descendantIds:
            if descendant in self.ancestor_dict:
                ancestorIds.update(self.ancestor_dict[descendant])

        return ancestorIds - descendantIds

    def get_descendantIds(self, ancestorIds: set):
        """
        Returns a tuple containing the unique descendants of the given ancestors.
        Ensures that none of the input ancestor IDs appear in the output.
        """
        descendantIds = set()

        for ancestor in ancestorIds:
            if ancestor in self.descendant_dict:
                descendantIds.update(self.descendant_dict[ancestor])

        return descendantIds - ancestorIds
