from typing import Optional
from pydantic import BaseModel, validator

from .utils.metaannotationstypes import *


META_ANNS_DICT = {
    "presence": Presence,
    "relevance": Relevance,
    "laterality (generic)": Laterality,
    "substance_category": SubstanceCategory,
    "reaction_pos": ReactionPos,
    "allergy_type": AllergyType,
    "severity": Severity
}


class MetaAnnotations(BaseModel):
    name: str
    value: Enum
    confidence: Optional[float]

    @validator('value', pre=True)
    def validate_value(cls, value, values):
        enum_dict = META_ANNS_DICT
        if isinstance(value, str):
            enum_type = enum_dict.get(values["name"])
            if enum_type is not None:
                try:
                    return enum_type(value)
                except ValueError:
                    raise ValueError(f"Invalid value: {value}")
            else:
                raise ValueError(f"Invalid mapping for {values['name']}")

        return value

