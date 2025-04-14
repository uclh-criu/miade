import os
import sys
import spacy
import yaml
import logging

from negspacy.negation import Negex  # noqa: F401
from pathlib import Path
from typing import List, Optional, Dict

from .concept import Concept, Category
from .note import Note
from .annotators import Annotator, ProblemsAnnotator, MedsAllergiesAnnotator  # noqa: F401
from .dosageextractor import DosageExtractor, DummyDosageExtractor
from .utils.miade_cat import MiADE_CAT
from .utils.modelfactory import ModelFactory
from .utils.annotatorconfig import AnnotatorConfig

log = logging.getLogger(__name__)


def create_annotator(name: str, model_factory: ModelFactory) -> Annotator:
    """
    Returns Annotator created from ModelFactory configs.

    Args:
        name (str): Alias of the model.
        model_factory (ModelFactory): Model factory loaded from config.yaml containing mapping of alias/name
            to MedCAT model id and MiADE annotator.

    Returns:
        Annotator object created from the ModelFactory configs.

    Raises:
        ValueError: If the MedCAT model for the given name does not exist, either because it is not configured
            in config.yaml or missing from the models directory.

    """
    name = name.lower()
    if name not in model_factory.models:
        raise ValueError(
            f"MedCAT model for {name} does not exist: either not configured in config.yaml or "
            f"missing from models directory"
        )

    if name in model_factory.annotators.keys():
        return model_factory.annotators[name](
            cat=model_factory.models.get(name), config=model_factory.configs.get(name)
        )
    else:
        log.warning(f"Annotator {name} does not exist, loading generic Annotator")
        return Annotator(model_factory.models[name])


class NoteProcessor:
    """
    Main processor of MiADE which extract, postprocesses, and deduplicates concepts given
    annotators (MedCAT models), Note, and existing concepts

    Args:
        model_directory (Path): Path to directory that contains medcat models and a config.yaml file
        model_config_path (Path, optional): Path to the model config file. Defaults to None.
        log_level (int, optional): Log level. Defaults to logging.INFO.
        dosage_extractor_log_level (int, optional): Log level for dosage extractor. Defaults to logging.INFO.
        device (str, optional): Device to run inference on (cpu or gpu). Defaults to "cpu".
        custom_annotators (List[Annotator], optional): List of custom annotators. Defaults to None.
    """

    def __init__(
        self,
        model_directory: Path,
        model_config_path: Path = None,
        log_level: int = logging.INFO,
        dosage_extractor_log_level: int = logging.INFO,
        device: str = "cpu",
        custom_annotators: Optional[List[Annotator]] = None,
    ):
        logging.getLogger("miade").setLevel(log_level)
        logging.getLogger("miade.dosageextractor").setLevel(dosage_extractor_log_level)
        logging.getLogger("miade.drugdoseade").setLevel(dosage_extractor_log_level)

        self.device: str = device

        self.annotators: List[Annotator] = []
        self.model_directory: Path = model_directory
        self.model_config_path: Path = model_config_path
        self.model_factory: ModelFactory = self._load_model_factory(custom_annotators)
        # self.dosage_extractor: DosageExtractor = DosageExtractor()
        self.dosage_extractor: DosageExtractor = DummyDosageExtractor()

    def _load_config(self) -> Dict:
        """
        Loads the configuration file (config.yaml) in the configured model path.
        If the model path is not explicitly passed, it defaults to the model directory.

        Returns:
            A dictionary containing the loaded config file.
        """
        if self.model_config_path is None:
            config_path = os.path.join(self.model_directory, "config.yaml")
        else:
            config_path = self.model_config_path

        if os.path.isfile(config_path):
            log.info(f"Found config file {config_path}")
        else:
            log.error(f"No model config file found at {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def _load_model_factory(self, custom_annotators: Optional[List[Annotator]] = None) -> ModelFactory:
        """
        Loads the model factory which maps model aliases to MedCAT model IDs and MiADE annotators.

        Args:
            custom_annotators (List[Annotators], optional): List of custom annotators to initialize. Defaults to None.

        Returns:
            The initialized ModelFactory object.

        Raises:
            Exception: If there is an error loading MedCAT models.

        """
        meta_cat_config_dict = {"general": {"device": self.device}}
        config_dict = self._load_config()
        loaded_models = {}

        # get model {id: cat_model}
        log.info(f"Loading MedCAT models from {self.model_directory}")
        for model_pack_filepath in self.model_directory.glob("*.zip"):
            try:
                cat = MiADE_CAT.load_model_pack(str(model_pack_filepath), meta_cat_config_dict=meta_cat_config_dict)
                # temp fix reload to load stop words
                cat.pipe._nlp = spacy.load(
                    cat.config.general.spacy_model, disable=cat.config.general.spacy_disabled_components
                )
                cat._create_pipeline(config=cat.config)
                cat_id = cat.config.version["id"]
                loaded_models[cat_id] = cat
            except Exception as e:
                raise Exception(f"Error loading MedCAT models: {e}")

        mapped_models = {}
        # map to name if given {name: <class CAT>}
        if "models" in config_dict:
            for name, model_id in config_dict["models"].items():
                cat_model = loaded_models.get(model_id)
                if cat_model is None:
                    log.warning(f"No match for model id {model_id} in {self.model_directory}, skipping")
                    continue
                mapped_models[name] = cat_model
        else:
            log.warning("No model ids configured!")

        mapped_annotators = {}
        # {name: <class Annotator>}
        if "annotators" in config_dict:
            for name, annotator_string in config_dict["annotators"].items():
                if custom_annotators is not None:
                    for annotator_class in custom_annotators:
                        if annotator_class.__name__ == annotator_string:
                            mapped_annotators[name] = annotator_class
                            break
                if name not in mapped_annotators:
                    try:
                        annotator_class = getattr(sys.modules[__name__], annotator_string)
                        mapped_annotators[name] = annotator_class
                    except AttributeError as e:
                        log.warning(f"{annotator_string} not found: {e}")
        else:
            log.warning("No annotators configured!")

        mapped_configs = {}
        if "general" in config_dict:
            for name, config in config_dict["general"].items():
                try:
                    mapped_configs[name] = AnnotatorConfig(**config)
                except Exception as e:
                    log.error(f"Error processing config for '{name}': {str(e)}")
        else:
            log.warning("No general settings configured, using default settings.")

        model_factory_config = {"models": mapped_models, "annotators": mapped_annotators, "configs": mapped_configs}

        return ModelFactory(**model_factory_config)

    def add_annotator(self, name: str) -> None:
        """
        Adds an annotator to the processor.

        Args:
            name (str): The alias of the annotator to add.

        Returns:
            None

        Raises:
            Exception: If there is an error creating the annotator.
        """
        try:
            annotator = create_annotator(name, self.model_factory)
            log.info(
                f"Added {type(annotator).__name__} to processor with config {self.model_factory.configs.get(name)}"
            )
        except Exception as e:
            raise Exception(f"Error creating annotator: {e}")

        self.annotators.append(annotator)

    def remove_annotator(self, name: str) -> None:
        """
        Removes an annotator from the processor.

        Args:
            name (str): The alias of the annotator to remove.

        Returns:
            None
        """
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

    def print_model_cards(self) -> None:
        """
        Prints the model cards for each annotator in the `annotators` list.

        Each model card includes the name of the annotator's class and its category.
        """
        for annotator in self.annotators:
            print(f"{type(annotator).__name__}: {annotator.cat}")

    def process(self, note: Note, record_concepts: Optional[List[Concept]] = None) -> List[Concept]:
        """
        Process the given note and extract concepts using the loaded annotators.

        Args:
            note (Note): The note to be processed.
            record_concepts (Optional[List[Concept]]): A list of existing concepts in the EHR record.

        Returns:
            A list of extracted concepts.

        """
        if not self.annotators:
            log.warning("No annotators loaded, use .add_annotator() to load annotators")
            return []

        concepts: List[Concept] = []

        for annotator in self.annotators:
            log.debug(f"Processing concepts with {type(annotator).__name__}")
            if Category.MEDICATION in annotator.concept_types:
                detected_concepts = annotator(note, record_concepts, self.dosage_extractor)
                concepts.extend(detected_concepts)
            else:
                detected_concepts = annotator(note, record_concepts)
                concepts.extend(detected_concepts)

        return concepts

    def get_concept_dicts(
        self, note: Note, filter_uncategorized: bool = True, record_concepts: Optional[List[Concept]] = None
    ) -> List[Dict]:
        """
        Returns concepts in dictionary format.

        Args:
            note (Note): Note containing text to extract concepts from.
            filter_uncategorized (bool): If True, does not return concepts where category=None. Default is True.
            record_concepts (Optional[List[Concept]]): List of concepts in existing record.

        Returns:
            Extracted concepts in JSON-compatible dictionary format.
        """
        concepts = self.process(note, record_concepts)
        concept_list = []
        for concept in concepts:
            if filter_uncategorized and concept.category is None:
                continue
            concept_dict = concept.__dict__
            if concept.dosage is not None:
                concept_dict["dosage"] = {
                    "dose": concept.dosage.dose.dict() if concept.dosage.dose else None,
                    "duration": concept.dosage.duration.dict() if concept.dosage.duration else None,
                    "frequency": concept.dosage.frequency.dict() if concept.dosage.frequency else None,
                    "route": concept.dosage.route.dict() if concept.dosage.route else None,
                }
            if concept.meta is not None:
                meta_anns = []
                for meta in concept.meta:
                    meta_dict = meta.__dict__
                    meta_dict["value"] = meta.value.name
                    meta_anns.append(meta_dict)
                concept_dict["meta"] = meta_anns
            if concept.category is not None:
                concept_dict["category"] = concept.category.name
            concept_list.append(concept_dict)

        return concept_list
