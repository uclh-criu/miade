import spacy
import re
import io
import pkgutil
import logging
import pandas as pd

from typing import Dict
from spacy.language import Language
from spacy.tokens import Doc

from .utils import word_replace, numbers_replace

log = logging.getLogger(__name__)


@spacy.registry.misc("singleword_lookup_dict.v1")
def create_singleword_dict():
    singlewords_data = pkgutil.get_data(__name__, "../data/singlewords.csv")
    singlewords_dict = pd.read_csv(io.BytesIO(singlewords_data), index_col=0, squeeze=True).to_dict()

    return singlewords_dict


@spacy.registry.misc("multiword_lookup_dict.v1")
def create_multiword_dict():
    multiwords_data = pkgutil.get_data(__name__, "../data/multiwords.csv")
    multiwords_dict = pd.read_csv(io.BytesIO(multiwords_data), index_col=0, squeeze=True).to_dict()

    return multiwords_dict


@Language.factory("preprocessor", default_config={"singleword": {"@misc": "singleword_lookup_dict.v1"},
                                                  "multiword": {"@misc": "multiword_lookup_dict.v1"}})
def create_preprocessor(nlp: Language, name: str, singleword: Dict, multiword: Dict):
    return Preprocessor(nlp, singleword, multiword)


class Preprocessor:
    """
    Preprocessing steps to standardise text based on CALIBERdrugdose algorithm
    using singlrword and multiword lookup dicts
    """
    def __init__(self, nlp: Language, singleword: Dict, multiword: Dict):
        self.spellcheck_dict = singleword
        self.standardize_dict = multiword
        if not Doc.has_extension("original_text"):
            Doc.set_extension("original_text", default="")

    def __call__(self, doc: Doc) -> Doc:
        processed_text = []

        # singleword replacement
        for word in [token.text.lower() for token in doc]:
            # strip periods if not between two numbers
            word = re.sub(r"(?<!\d)\.(?!\d)", "", word)
            processed_text = word_replace(word, self.spellcheck_dict, processed_text)

        # numbers replace 1
        processed_text = "start {} ".format(" ".join(processed_text))

        # remove numbers relating to strength of med e.g. aspirin 200mg tablets...
        processed_text = re.sub(r" (\d+\.?\d*) (mg|ml|g|mcg|microgram|gram|%)"
                                r"(\s|/)(tab|cap|gel|cream|dose|pessaries)", "",
                                processed_text)
        processed_text = numbers_replace(processed_text)

        # multiword replacement
        for words in self.standardize_dict:
            pattern = r" {} ".format(words)
            if isinstance(self.standardize_dict[words], str):
                replacement = r" {} ".format(self.standardize_dict[words])
                if replacement == "   ":
                    replacement = " "
            else:
                replacement = " "
            new_text = re.sub(pattern, replacement, processed_text)
            if new_text != processed_text:
                if replacement == " ":
                    log.debug(f"Removed multiword match '{words}'")
                else:
                    log.debug(f"Replaced multiword match '{words}' with '{replacement}'")
            processed_text = new_text

        # numbers replace 2
        processed_text = numbers_replace(processed_text)

        # return new doc because spacy will always keep original and want ner to be on the standardised text
        # (maybe something to experiment with?)
        new_doc = Doc(doc.vocab, words=processed_text.split())
        new_doc._.original_text = doc.text

        log.debug(f"Final processed text: {new_doc.text}")

        return new_doc
