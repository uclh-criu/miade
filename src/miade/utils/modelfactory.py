from pydantic import BaseModel
from typing import Dict

from .miade_cat import MiADE_CAT


class ModelFactory(BaseModel):
    models: Dict[str, MiADE_CAT]
    class Config:
        arbitrary_types_allowed = True