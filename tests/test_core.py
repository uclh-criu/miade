import pytest

import nlp_engine_core


def test_core(model_filepath):
    processor = nlp_engine_core.core.NoteProcessor(model_filepath)
    assert processor.annotator.get_entities("He was diagnosed with kidney failure") == {
        "entities": {
            0: {
                "pretty_name": "Diagnosis",
                "cui": "C0011900",
                "type_ids": ["T060"],
                "types": ["Diagnostic Procedure"],
                "source_value": "diagnosed",
                "detected_name": "diagnosed",
                "acc": 0.391300890979873,
                "context_similarity": 0.391300890979873,
                "start": 7,
                "end": 16,
                "icd10": [],
                "ontologies": [],
                "snomed": [],
                "id": 0,
                "meta_anns": {
                    "Status": {
                        "value": "Affirmed",
                        "confidence": 0.9999933838844299,
                        "name": "Status",
                    }
                },
            },
            2: {
                "pretty_name": "Kidney Failure",
                "cui": "C0035078",
                "type_ids": ["T047"],
                "types": ["Disease or Syndrome"],
                "source_value": "kidney failure",
                "detected_name": "kidney~failure",
                "acc": 1.0,
                "context_similarity": 1.0,
                "start": 22,
                "end": 36,
                "icd10": [],
                "ontologies": [],
                "snomed": [],
                "id": 2,
                "meta_anns": {
                    "Status": {
                        "value": "Affirmed",
                        "confidence": 0.9999961853027344,
                        "name": "Status",
                    }
                },
            },
        },
        "tokens": [],
    }
