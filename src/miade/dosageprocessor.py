import spacy
import logging

from .dosage import Dosage
from .dosage_extractor.preprocessor import Preprocessor
from .dosage_extractor.pattern_matcher import PatternMatcher
from .dosage_extractor.entities_refiner import EntitiesRefiner


log = logging.getLogger(__name__)


class DosageProcessor:
    """doc string for dosage processor"""

    def __init__(self, model: str = "en_core_med7_lg"):
        self.model = model
        self.dosage_extractor = self.create_extractor_pipeline()

    def create_extractor_pipeline(self):
        nlp = spacy.load(self.model)
        nlp.add_pipe("preprocessor", first=True)
        nlp.add_pipe("pattern_matcher", before="ner")
        nlp.add_pipe("entities_refiner", after="ner")

        return nlp

    def process(self, text: str):
        doc = self.dosage_extractor(text)

        log.debug(f"med7 results: {[(e.text, e.label_, e._.total_dose) for e in doc.ents]}")
        log.debug(f"lookup results: {doc._.results}")

        dosage = Dosage.from_doc(doc)

        return dosage
