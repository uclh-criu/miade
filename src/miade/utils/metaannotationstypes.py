from enum import Enum


#  Problem meta-annotation types
# 1 = keep, -1 = discard, 0 = further action/conversion needed
class Presence(Enum):
    CONFIRMED = 1
    SUSPECTED = 0
    NEGATED = 0


class Relevance(Enum):
    PRESENT = 1
    HISTORIC = 0
    IRRELEVANT = -1


class Laterality(Enum):
    NO_LATERALITY = 0
    LEFT = 0
    RIGHT = 0
    BILATERAL = 0


# Medication and Allergy meta-annotation types
class Reaction(Enum):
    N0T_REACTION = 0
    AFTER_SUBSTANCE = 0
    BEFORE_SUBSTANCE = 0


class Substance(Enum):
    IRRELEVANT = -1
    TAKING = 1
    ALLERGIC = 1
    ADVERSE_REACTION = 1
    INTOLERANT = 0
    UNSPECIFIED = 0


class Severity(Enum):
    NOT_APPLICABLE = -1
    MILD = 0
    MODERATE = 0
    SEVERE = 0
