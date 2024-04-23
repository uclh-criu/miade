import re
import io
import pkgutil
import logging
import spacy

import pandas as pd

from spacy.language import Language
from spacy.tokens import Span, Doc

from typing import Dict

from miade.paragraph import ParagraphType

# TODO: this spacy pipeline doesn't play well so I will probably redo all this as a regular class


log = logging.getLogger(__name__)


def load_regex_config_mappings(filename: str) -> Dict:
    regex_config = pkgutil.get_data(__name__, filename)
    data = (
        pd.read_csv(
            io.BytesIO(regex_config),
            index_col=0,
        )
        .squeeze("columns")
        .T.to_dict()
    )
    regex_lookup = {}

    for paragraph, regex in data.items():
        paragraph_enum = None
        try:
            paragraph_enum = ParagraphType(paragraph)
        except ValueError as e:
            log.warning(e)

        if paragraph_enum is not None:
            regex_lookup[paragraph_enum] = regex

    return regex_lookup


@spacy.registry.misc("regex_config.v1")
def create_patterns_dict():
    regex_config = load_regex_config_mappings("./data/regex_para_chunk.csv")

    return regex_config


@Language.factory(
    "paragraph_segmenter",
    default_config={"regex_config": {"@misc": "regex_config.v1"}},
)
def create_paragraph_segmenter(nlp: Language, name: str, regex_config: Dict):
    return ParagraphSegmenter(nlp, regex_config)


class ParagraphSegmenter:
    def __init__(self, nlp: Language, regex_config: Dict):
        self.regex_config = regex_config
        # Set custom extensions
        if not Span.has_extension("heading"):
            Span.set_extension("heading", default=None)
        if not Span.has_extension("body"):
            Span.set_extension("body", default=None, force=True)
        if not Span.has_extension("type"):
            Span.set_extension("type", default=None, force=True)

    def __call__(self, doc: Doc) -> Doc:
        paragraphs = re.split(r"\n\n+", doc.text)
        start = 0
        new_spans = []

        for text in paragraphs:
            match = re.search(r"^(.*?)(?:\n|$)([\s\S]*)", text)
            if match:
                heading, body = match.group(1), match.group(2)
            else:
                heading, body = text, ""

            end = start + len(text)
            span = doc.char_span(start, end, label="PARAGRAPH")
            if span is not None:
                span._.heading = heading
                span._.body = body
                span._.type = ParagraphType.prose  # default type

                heading_lower = heading.lower()
                for paragraph_type, pattern in self.regex_config.items():
                    if re.search(pattern, heading_lower):
                        span._.type = ParagraphType[paragraph_type]
                        break

                new_spans.append(span)
            start = end + 2

        doc.spans["paragraphs"] = new_spans

        return doc
