import spacy
import logging

from spacy import Language
from typing import Optional

from .dosage import Dosage
from .drugdoseade.preprocessor import Preprocessor
from .drugdoseade.pattern_matcher import PatternMatcher
from .drugdoseade.entities_refiner import EntitiesRefiner


log = logging.getLogger(__name__)


class DosageExtractor:
    """
    Parses and extracts drug dosage
    LIMeADE - Long Instruction Medication AI Data Extractor??
    """

    def __init__(self, model: str = "en_core_med7_lg"):
        self.model = model
        self.dosage_extractor = self._create_drugdoseade_pipeline()

    def _create_drugdoseade_pipeline(self) -> Language:
        """
        Creates a spacy pipeline with given model (default med7)
        and customised pipeline components for dosage extraction
        :return: nlp (spacy.Language)
        """
        nlp = spacy.load(self.model)
        nlp.add_pipe("preprocessor", first=True)
        nlp.add_pipe("pattern_matcher", before="ner")
        nlp.add_pipe("entities_refiner", after="ner")

        return nlp

    def extract(self, text: str, calculate: bool = True) -> Optional[Dosage]:
        """
        Processes a string that contains dosage instructions (excluding drug concept as this is handled by core)
        :param text: (str) string containing dosage
        :param calculate: (bool) whether to calculate duration from total and daily dose, if given
        :return: dosage: (Dosage) dosage object with parsed dosages in CDA format
        """
        log.info(f"Processing dose string: {text}")

        doc = self.dosage_extractor(text)

        log.debug(
            f"Med7 results: {[(e.text, e.label_, e._.total_dose) for e in doc.ents]}"
        )
        log.info(f"Parsing dosage from lookup results: {doc._.results}")

        dosage = Dosage.from_doc(doc=doc, calculate=calculate)

        if all(
            v is None
            for v in [dosage.dose, dosage.frequency, dosage.route, dosage.duration]
        ):
            return None

        return dosage

    def __call__(self, text: str, calculate: bool = True):
        return self.extract(text, calculate)
