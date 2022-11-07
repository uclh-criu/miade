from typing import Optional, Dict

from .utils.metaannotationstypes import *


class MetaAnnotations(object):
    def __init__(self,
                 presence: Optional[Presence] = None,
                 relevance: Optional[Relevance] = None,
                 laterality: Optional[Laterality] = None,
                 reaction: Optional[Reaction] = None,
                 substance: Optional[Substance] = None,
                 severity: Optional[Severity] = None):
        self.presence = presence
        self.relevance = relevance
        self.laterality = laterality
        self.reaction = reaction
        self.substance = substance
        self.severity = severity

    @classmethod
    def from_dict(cls, meta_anns: [Dict]):
        presence = None
        relevance = None
        laterality = None
        reaction = None
        substance = None
        severity = None

        for meta_ann in meta_anns.values():
            if meta_ann["name"] == "presence":
                if meta_ann["value"] == "confirmed":
                    presence = Presence.CONFIRMED
                elif meta_ann["value"] == "negated":
                    presence = Presence.NEGATED
                elif meta_ann["value"] == "suspected":
                    presence = Presence.SUSPECTED
            elif meta_ann["name"] == "relevance":
                if meta_ann["value"] == "present":
                    relevance = Relevance.PRESENT
                elif meta_ann["value"] == "historic":
                    relevance = Relevance.HISTORIC
                elif meta_ann["value"] == "irrelevant":
                    relevance = Relevance.IRRELEVANT
            elif meta_ann["name"] == "laterality (generic)":
                if meta_ann["value"] == "none":
                    laterality = Laterality.NO_LATERALITY
                elif meta_ann["value"] == "left":
                    laterality = Laterality.LEFT
                elif meta_ann["value"] == "right":
                    laterality = Laterality.RIGHT
                elif meta_ann["value"] == "bilateral":
                    laterality = Laterality.BILATERAL
            elif meta_ann["name"] == "reaction":
                reaction = None
            elif meta_ann["name"] == "substance":
                substance = None
            elif meta_ann["name"] == "severity":
                severity = None

        return cls(presence=presence,
                   relevance=relevance,
                   laterality=laterality,
                   reaction=reaction,
                   substance=substance,
                   severity=severity)

    def __str__(self):
        return self.__dict__

    def __eq__(self, other):
        return self.__dict__ == other.__dict__




