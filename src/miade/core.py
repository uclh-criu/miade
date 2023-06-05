import os
import yaml
import logging

from negspacy.negation import Negex
from pathlib import Path
from typing import List, Optional

from .concept import Concept, Category
from .note import Note
from .annotators import Annotator
from .dosageextractor import DosageExtractor
from .utils.miade_cat import MiADE_CAT
from .utils.modelfactory import ModelFactory

log = logging.getLogger(__name__)


def create_annotator(name: str, model_factory: ModelFactory):
    name = name.lower()
    if name not in model_factory.models:
        raise ValueError(f"MedCAT model for {name} does not exist: either not configured in Config.yaml or "
                         f"missing from models directory")

    if name in model_factory.annotators.keys():
        return model_factory.annotators[name](model_factory.models[name])
    else:
        log.warning(f"Annotator {name} does not exist, loading generic Annotator")
        return Annotator(model_factory.models[name])


class NoteProcessor:
    """docstring for NoteProcessor."""

    def __init__(
        self,
        model_directory: Path,
        log_level: int = logging.INFO,
        device: str = "cpu"
    ):
        logging.getLogger("miade").setLevel(log_level)
        self.device: str = device

        self.annotators: List[Annotator] = []
        self.model_directory: Path = model_directory
        self.model_factory: ModelFactory = self._load_model_factory()
        self.dosage_extractor: DosageExtractor = DosageExtractor()

    def _load_config(self):
        config_path = os.path.join(self.model_directory, "config.yaml")
        if os.path.isfile(config_path):
            log.info(f"Found config file {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def _load_model_factory(self):

        meta_cat_config_dict = {"general": {"device": self.device}}
        config_dict = self._load_config()
        loaded_models = {}

        # get model {id: cat_model}
        log.info(f"Loading MedCAT models from {self.model_directory}")
        for model_pack_filepath in self.model_directory.glob("*.zip"):
            try:
                cat = MiADE_CAT.load_model_pack(str(model_pack_filepath), meta_cat_config_dict=meta_cat_config_dict)
                cat_id = cat.config.version["id"]
                loaded_models[cat_id] = cat
            except Exception as e:
                raise Exception(f"Error loading MedCAT models: {e}")

        mapped_models = {}
        # map to name if given {name: <class CAT>}
        for name, model_id in config_dict["models"].items():
            cat_model = loaded_models.get(model_id)
            if cat_model is None:
                log.warning(f"No match for model id {model_id} in {self.model_directory}, skipping")
                continue
            mapped_models[name] = cat_model

        mapped_annotators = {}
        # {name: <class Annotator>}
        for name, annotator_string in config_dict["annotators"].items():
            try:
                annotator_class = globals()[annotator_string]
                mapped_annotators[name] = annotator_class
            except Exception as e:
                log.warning(f"{annotator_string} not found: {e}")

        model_factory_config = {"models": mapped_models,
                                "annotators": mapped_annotators}

        return ModelFactory(**model_factory_config)


    def add_annotator(self, name: str, use_negex=True) -> None:
        try:
            annotator = create_annotator(name, self.model_factory)
            log.info(f"Added {type(annotator).__name__} to processor")
        except Exception as e:
            raise Exception(f"Error creating annotator: {e}")

        if use_negex:
            annotator.add_negex_pipeline()
            log.info(f"Added Negex context detection for {type(annotator).__name__}")

        self.annotators.append(annotator)

    def remove_annotator(self, name: str) -> None:
        annotator_found = False
        annotator_name = self.model_factory.annotators[name]

        for annotator in self.annotators:
            if type(annotator).__name__ == annotator_name.__name__:
                self.annotators.remove(annotator)
                annotator_found = True
                log.info(f"Removed {type(annotator).__name__} from processor")
                break

        if not annotator_found:
            log.warning(f"Annotator {type(name).__name__} not found in processor")

    def print_model_cards(self):
        for annotator in self.annotators:
            print(f"{type(annotator).__name__}: {annotator.cat}")

    def process(self, note: Note, record_concepts: Optional[List[Concept]] = None) -> List[Concept]:
        if not self.annotators:
            log.warning("No annotators loaded, use .add_annotator() to load annotators")
            return []

        concepts: List[Concept] = []

        for annotator in self.annotators:
            if Category.MEDICATION in annotator.concept_types:
                concepts.extend(annotator(note, record_concepts, self.dosage_extractor))
            else:
                concepts.extend(annotator(note, record_concepts))

            log.debug(
                f"{type(annotator).__name__} detected concepts: "
                f"{[(concept.id, concept.name, concept.category) for concept in concepts]}"
            )

        return concepts
