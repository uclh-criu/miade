from enum import Enum


#  Problem meta-annotation types
class Presence(Enum):
    CONFIRMED = 1
    SUSPECTED = 2
    NEGATED = 3


class Relevance(Enum):
    PRESENT = 1
    HISTORIC = 2
    IRRELEVANT = 3


class Laterality(Enum):
    NO_LATERALITY = 1
    LEFT = 2
    RIGHT = 3
    BILATERAL = 4


# Medication and Allergy meta-annotation types
class ReactionPosition(Enum):
    N0T_REACTION = 1
    AFTER_SUBSTANCE = 2
    BEFORE_SUBSTANCE = 3


class SubstanceCategory(Enum):
    IRRELEVANT = 1
    TAKING = 2
    ADVERSE_REACTION = 3


class AllergyType(Enum):
    UNSPECIFIED = 1
    ALLERGY = 2
    INTOLERANCE = 3


class Severity(Enum):
    UNSPECIFIED = 1
    MILD = 2
    MODERATE = 3
    SEVERE = 4
