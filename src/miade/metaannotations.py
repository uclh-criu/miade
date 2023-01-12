from typing import Optional, Dict

from .utils.metaannotationstypes import *


class MetaAnnotations(object):
    def __init__(
            self,
            presence: Optional[Presence] = None,
            relevance: Optional[Relevance] = None,
            laterality: Optional[Laterality] = None,
            confidences: Optional[Dict] = None,
    ):
        self.presence = presence
        self.relevance = relevance
        self.laterality = laterality
        self.confidences = confidences

    @classmethod
    def from_dict(cls, meta_anns: [Dict]):
        presence = None
        relevance = None
        laterality = None
        confidences = {}

        for meta_ann in meta_anns.values():
            if meta_ann["name"] == "presence":
                confidences[Presence] = meta_ann["confidence"]
                if meta_ann["value"] == "confirmed":
                    presence = Presence.CONFIRMED
                elif meta_ann["value"] == "negated":
                    presence = Presence.NEGATED
                elif meta_ann["value"] == "suspected":
                    presence = Presence.SUSPECTED
            elif meta_ann["name"] == "relevance":
                confidences[Relevance] = meta_ann["confidence"]
                if meta_ann["value"] == "present":
                    relevance = Relevance.PRESENT
                elif meta_ann["value"] == "historic":
                    relevance = Relevance.HISTORIC
                elif meta_ann["value"] == "irrelevant":
                    relevance = Relevance.IRRELEVANT
            elif meta_ann["name"] == "laterality (generic)":
                confidences[Laterality] = meta_ann["confidence"]
                if meta_ann["value"] == "none":
                    laterality = Laterality.NO_LATERALITY
                elif meta_ann["value"] == "left":
                    laterality = Laterality.LEFT
                elif meta_ann["value"] == "right":
                    laterality = Laterality.RIGHT
                elif meta_ann["value"] == "bilateral":
                    laterality = Laterality.BILATERAL

        return cls(
            presence=presence,
            relevance=relevance,
            laterality=laterality,
            confidences=confidences,
        )

    def __str__(self):
        return self.__dict__

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
