# Quickstart
## Extract concepts and dosages from a Note using MiADE

### Configuring the MiADE Processor
`NoteProcessor` is the MiADE core. It is initialised with a model directory path that contains all the MedCAT model pack .zip files we would like to use in our pipeline, and a config file that maps an alias to the model IDs (model IDs can be found in MedCAT `model_cards` or usually will be in the name) and annotators we would like to use:

```yaml title="config.yaml"
models:
  problems: f25ec9423958e8d6
  meds/allergies: a146c741501cf1f7
annotators:
  problems: ProblemsAnnotator
  meds/allergies: MedsAllergiesAnnotator
```
We can initialise a MiADE `NoteProcessor` object by passing in the model directory which contains our MedCAT models and `config.yaml` file:

```python
miade = NoteProcessor(Path("path/to/model/dir"))
```
Once `NoteProcessor` is initialised, we can add annotators by the aliases we have specified in `config.yaml` to our processor:

```python
miade.add_annotator("problems", use_negex=True)
miade.add_annotator("meds/allergies")
```

When adding annotators, we have the option to add [NegSpacy](https://spacy.io/universe/project/negspacy) to the MedCAT spaCy pipeline, which implements the [NegEx algorithm](https://www.sciencedirect.com/science/article/pii/S1532046401910299) (Chapman et al. 2001) for negation detection. This allows the models to perform simple rule-based negation detection in the absence of MetaCAT models.

### Creating a Note

Create a `Note` object which contains the text we would like to extract concepts and dosages from:

```python
text = """
Suspected heart failure

PMH:
prev history of Hypothyroidism
MI 10 years ago


Current meds:
Losartan 100mg daily
Atorvastatin 20mg daily
Paracetamol 500mg tablets 2 tabs qds prn

Allergies:
Penicillin - rash

Referred with swollen ankles and shortness of breath since 2 weeks.
"""

note = Note(text)
```

### Extracting Concepts and Dosages

MiADE currently extracts concepts in SNOMED CT. Each concept contains:

- `name`: name of concept
- `id`: concept ID
- `category`: type of concept e.g. problems, medictions
- `start`: start index of concept span
- `end`: end index of concept span
- `dosage`: for medication concepts
- `negex`: Negex result if configured
- `meta`: Meta annotations if MetaCAT models are used

The dosages associated with medication concepts are extracted by the built-in MiADE `DosageExtractor`, using a combination of NER model [Med7](https://github.com/kormilitzin/med7) and [the CALIBER rule-based drug dose lookup algorithm](https://rdrr.io/rforge/CALIBERdrugdose/). It returns:
The output format is directly translatable to HL7 CDA but can also easily be converted to FHIR.

- `dose`
- `duration`
- `frequency`
- `route`

Putting it all together, we can now extract concepts from our `Note` object:

=== "as Concept object"
    ```python
    concepts = miade.process(note)
    for concept in concepts:
        print(concept)
    
    # {name: breaking out - eruption, id: 271807003, category: Category.REACTION, start: 204, end: 208, dosage: None, negex: False, meta: None} 
    # {name: penicillin, id: 764146007, category: Category.ALLERGY, start: 191, end: 201, dosage: None, negex: False, meta: None} 
    ```
=== "as Dict"
    ```python
    concepts = miade.get_concept_dicts(note)
    print(concepts)

    # [{'name': 'hypothyroidism (historic)',
    # 'id': '161443002',
    # 'category': 'PROBLEM',
    # 'start': 46,
    # 'end': 60,
    # 'dosage': None,
    # 'negex': False,
    # 'meta': [{'name': 'relevance',
    #           'value': 'HISTORIC',
    #           'confidence': 0.999841570854187},
    # ...
    ```

#### Handling existing records: deduplication

MiADE is built to handle existing medication records from EHR systems that can be sent alongside the note. It will perform basic deduplication matching on id for existing record concepts.
```python
# create list of concepts that already exists in patient record
record_concepts = [
    Concept(id="161443002", name="hypothyroidism (historic)", category=Category.PROBLEM),
    Concept(id="267039000", name="swollen ankle", category=Category.PROBLEM)
]
```

We can pass in a list of existing concepts from the EHR to MiADE at runtime:

```python
miade.process(note=note, record_concepts=record_concepts)
```

## Customising MiADE
### Training Custom MedCAT Models
MiADE provides command line interface scripts for automatically building MedCAT model packs, unsupervised training, supervised training steps, and the creation and training of MetaCAT models. For more information on MedCAT models, see MedCAT [documentation](https://github.com/CogStack/MedCAT) and [paper](https://arxiv.org/abs/2010.01165).

The ```--synthetic-data-path``` option allows you to add synthetically generated training data in CSV format to the supervised and MetaCAT training steps. The CSV should have the following format:

| text                          | cui               | name                       | start | end | relevance | presence  | laterality |
| ----------------------------- | ----------------- | -------------------------- | ----- | --- | --------- | --------- | -------------------- |
| no history of liver failure | 59927004 | hepatic failure      | 14     | 26  | historic  | negated | none                 


```bash
# Trains unsupervised training step of MedCAT model
miade train $MODEL_PACK_PATH $TEXT_DATA_PATH --tag "miade-example"
```
```bash
# Trains supervised training step of MedCAT model
miade train-supervised $MODEL_PACK_PATH $MEDCAT_JSON_EXPORT --synthetic-data-path $SYNTHETIC_CSV_PATH
```
```bash
# Creates BBPE tokenizer for MetaCAT
miade create-bbpe-tokenizer $TEXT_DATA_PATH
```
```bash
# Initialises MetaCAT models to do training on
miade create-metacats $TOKENIZER_PATH $CATEGORY_NAMES
```
```bash
# Trains the MetaCAT Bi-LSTM models
miade train-metacats $METACAT_MODEL_PATH $MEDCAT_JSON_EXPORT --synthetic-data-path $SYNTHETIC_CSV_PATH
```
```bash
# Packages MetaCAT models with the main MedCAT model pack
miade add_metacat_models $MODEL_PACK_PATH $METACAT_MODEL_PATH
```
### Creating Custom MiADE Annotators

We can add custom annotators with more specialised postprocessing steps to MiADE by subclassing `Annotator` and initialising `NoteProcessor` with a list of custom annotators 

`Annotator` methods include:

- `.get_concepts()`: returns MedCAT output as MiADE ```Concepts```
- `.add_dosages_to_concepts()`: uses the MiADE built-in ```DosageExtractor``` to add dosages associated with medication concepts
- `.deduplicate()`: filters duplicate concepts in list 

An example custom `Annotator` class might look like this:

```python
class CustomAnnotator(Annotator):
    def __init__(self, cat: MiADE_CAT):
        super().__init__(cat)
        # we need to include MEDICATIONS in concept types so MiADE processor will also extract dosages
        self.concept_types = [Category.MEDICATION, Category.ALLERGY]
    
    def postprocess(self, concepts: List[Concept]) -> List[Concept]:
        # some example post-processing code
        reactions = ["271807003"]
        allergens = ["764146007"]
        for concept in concepts:
            if concept.id in reactions:
                concept.category = Category.REACTION
            elif concept.id in allergens:
                concept.category = Category.ALLERGY
        return concepts
    
    def __call__(
        self,
        note: Note,
        record_concepts: Optional[List[Concept]] = None,
        dosage_extractor: Optional[DosageExtractor] = None,
    ):
        concepts = self.get_concepts(note)
        concepts = self.postprocess(concepts)
        # run dosage extractor if given
        if dosage_extractor is not None:
            concepts = self.add_dosages_to_concepts(dosage_extractor, concepts, note)
        concepts = self.deduplicate(concepts, record_concepts)

        return concepts
        
```

Add custom annotator to config file:


```yaml title="config.yaml"
models:
  problems: f25ec9423958e8d6
  meds/allergies: a146c741501cf1f7
  custom: a146c741501cf1f7
annotators:
  problems: ProblemsAnnotator
  meds/allergies: MedsAllergiesAnnotator
  custom: CustomAnnotator
```

Initialise MiADE with the custom annotator:

```python
miade = NoteProcessor(Path(MODEL_DIR), custom_annotators=[CustomAnnotator])
miade.add_annotator("custom")
```
