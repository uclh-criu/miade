# <img src="assets/miade-logo.png" width="40%">

[![Build Status](https://github.com/uclh-criu/miade/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/uclh-criu/miade/actions/workflows/ci.yml?query=Tests)
![License: Elastic License 2.0](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)

A set of tools for extracting formattable data from clinical notes stored in electronic health record systems. Powered by [MedCAT](https://github.com/CogStack/MedCAT) models.


## Installing

### Install MiADE

To install the stable release:
```bash
pip install miade
```

To install the latest development version of MiADE, clone this repository and run:
```bash
pip install .
```

### Downloading models
You may also need to download these additional models to run MiADE:

[spaCy](https://spacy.io/models/en) - required for MedCAT
```bash
python -m spacy download en_core_web_md
```
[med7](https://huggingface.co/kormilitzin/en_core_med7_lg) - required for medication dosage extraction
```bash
pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl
```

## Quickstart

Initialise MiADE with the path that you have saved your trained MedCAT models:

```python
miade = NoteProcessor(Path("path/to/model/dir"))
```
Add annotators:

```python
miade.add_annotator("problems")
miade.add_annotator("meds/allergies")
```

Create a note:

```python
text = "Patient has penicillin allergy with rash"
note = Note(text)
```

Extract concepts:

```python
concepts = miade.process(note)

for concept in concepts:
    print(concept)
    
# {name: breaking out - eruption, id: 271807003, category: Category.REACTION, start: 204, end: 208, dosage: None, negex: False, meta: None} 
# {name: penicillin, id: 764146007, category: Category.ALLERGY, start: 191, end: 201, dosage: None, negex: False, meta: None} 
```

## Contributing
See [contributing](CONTRIBUTING.md)

### Maintainers

| Name            | Email                       |
|-----------------|-----------------------------|
| James Brandreth | j.brandreth@ucl.ac.uk       |
| Jennifer Jiang  | jennifer.jiang.13@ucl.ac.uk |


## Acknowledgement

This project wouldn't be possible without the work at [Cogstack](https://cogstack.org/), [spaCy](https://spacy.io/), and [med7](https://huggingface.co/kormilitzin/en_core_med7_lg)!


## Licence

This project is licensed under the Elastic License 2.0. See [LICENSE](https://github.com/uclh-criu/miade/blob/documentation/LICENCE.md) for the full license text.