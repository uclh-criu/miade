from enum import Enum


#  Problem meta-annotation types
class Presence(Enum):
    CONFIRMED = "confirmed"
    SUSPECTED = "suspected"
    NEGATED = "negated"


class Relevance(Enum):
    PRESENT = "present"
    HISTORIC = "historic"
    IRRELEVANT = "irrelevant"


class Laterality(Enum):
    NO_LATERALITY = "none"
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"


# Medication and Allergy meta-annotation types
class ReactionPos(Enum):
    NOT_REACTION = "none"
    AFTER_SUBSTANCE = "after"
    BEFORE_SUBSTANCE = "before"


class SubstanceCategory(Enum):
    IRRELEVANT = "irrelevant"
    TAKING = "taking"
    ADVERSE_REACTION = "adverse reaction"


class AllergyType(Enum):
    UNSPECIFIED = "unspecified"
    ALLERGY = "allergy"
    INTOLERANCE = "intolerance"


class Severity(Enum):
    UNSPECIFIED = "unspecified"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
