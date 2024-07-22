# Configurations

## Annotator Configurations
The MiADE processor is configured by a `yaml` file that maps a human-readable key for each of your models to a MedCAT model ID and a MiADE annotator class. The config file must be in the same folder as the MedCAT models. An example `config.yaml` is given below:

```yaml title="config.yaml"
models:
  problems: f25ec9423958e8d6
  meds/allergies: a146c741501cf1f7
annotators:
  problems: ProblemsAnnotator
  meds/allergies: MedsAllergiesAnnotator
```

## Lookup Table Configurations