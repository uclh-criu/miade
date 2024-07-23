# Configurations

## Annotator
The MiADE processor is configured by a `yaml` file that maps a human-readable key for each of your models to a MedCAT model ID and a MiADE annotator class. The config file must be in the same folder as the MedCAT models.

- `models`: The models section maps human-readable key-value pairing to the MedCAT model ID to use in MiADE
- `annotators`: The annotators section maps human-readable key-value pairing to `Annotator` processing classes to use in MiADE
- `general`
  - `lookup_data_path`: Specifies the lookup data to use
  - `negation_detection`: `negex` (rule-based algorithm) or `None` (use default MetaCAT models)
  - `structured_list_limit`: Specifies the maximum number of concepts detected in a structured paragraph section. If there are more than the specified number of concepts, then concepts in prose are ignored (to avoid returning too many concepts which could be less relevant). Default 0 so that this feature is disabled by default.
  - `disable`: Disable any specific pipeline components - the API here is similar to spacy pipelines
  - `add_numbering`: Option to add a number prefix to the concept display names e.g. "01 Diabetes"


```yaml title="config.yaml"
models:
  problems: f25ec9423958e8d6
  meds/allergies: a146c741501cf1f7
annotators:
  problems: ProblemsAnnotator
  meds/allergies: MedsAllergiesAnnotator
general:
  problems:
    lookup_data_path: ./lookup_data/
    negation_detection: None
    structured_list_limit: 0  # if more than this number of concepts in structure section, ignore concepts in prose
    disable: []
    add_numbering: True
  meds/allergies:
    lookup_data_path: ./lookup_data/
    negation_detection: None
    disable: []
    add_numbering: False
```

## Lookup Table

Lookup tables are by default not packaged with the main MiADE package to provide flexibility to customise the postprocessing steps. We provide example lookup data in [`miade-dataset`](https://github.com/uclh-criu/miade-datasets/tree/master/cdb_and_model_files_sep_2023/lookups) which you can download and use.

```
git clone https://github.com/uclh-criu/miade-datasets.git
```