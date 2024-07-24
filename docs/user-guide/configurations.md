# Configurations

## Annotator
The MiADE processor is configured by a `yaml` file that maps a human-readable key for each of your models to a MedCAT model ID and an MiADE `Annotator`. The config file must be in the same folder as the MedCAT models.

**Required**

- `models`: The models section maps human-readable key-value pairing to the MedCAT model ID to use in MiADE
- `annotators`: The annotators section maps human-readable key-value pairing to `Annotator` processing classes to use in MiADE

**Optional**

  - `lookup_data_path`: Specifies the lookup data to use. If `None` a default MiADE set will be used.
  - `negation_detection`: `negex` (default rule-based algorithm) or `None` (use MetaCAT models)
  - `structured_list_limit`: Specifies the maximum number of concepts detected in a structured paragraph section. If there are more than the set number of concepts in a structured list, then concepts detected in prose are ignored (prioritises concepts detected in structured lists over free-form text to avoid returning too many irrelevant concepts). Default `100` so this feature is essentially disabled.
  - `disable`: Disable any specific postprocessing pipeline components - the usage here is similar to [spaCy pipelines](https://spacy.io/usage/processing-pipelines#disabling).
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
    lookup_data_path: ./custom_lookup_data/
    structured_list_limit: 0  # setting as 0 will ignore all concepts found in prose
    add_numbering: True
  meds/allergies:
    disable: ["vtm_converter"]
```
The default configurations for annotators are defined below:

::: miade.utils.annotatorconfig.AnnotatorConfig

## Lookup Table

Lookup tables are used to convert and filter concepts in the MiADE postprocessing steps for `ProblemsAnnotator` and `MedsAllergiesAnnotator`. We have packaged default lookup data (curated and used at UCLH) with MiADE for sample use. 

For a more detailed explanation on the creation and format of the lookup data, check out [miade-dataset](https://github.com/uclh-criu/miade-datasets/tree/master).

To customise your own lookup tables, you can pass in a directory which contains your lookup data in the `config.yaml` `lookup_data_path` field. Note you currently need to have **ALL** of the required lookup data in your directory (this will be improved in the future).

**Problems**
```
negated.csv
historic.csv
suspected.csv
problem_blacklist.csv
```

**MedsAllergies**
```
reactions_subset.csv
allergens_subset.csv
allergy_type.csv
valid_meds.csv
vtm_to_text.csv
vtm_to_vmp.csv
```

