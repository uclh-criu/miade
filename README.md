# <img src="assets/miade-logo.png" width="40%">

[![Build Status](https://github.com/uclh-criu/miade/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/uclh-criu/miade/actions/workflows/ci.yml?query=Tests)

A set of tools for extracting formattable data from clinical notes stored in electronic health record systems.
For the reference server implementation, see: [miade-server](https://github.com/uclh-criu/miade-server).

Built with Cogstack's [MedCAT](https://github.com/CogStack/MedCAT) package.

## Contents

1. [Contributors](#Contributors)
2. [Installing](#Installing)
3. [Testing](#Testing)
4. [MIADE Notereader Debug Mode](#miade-notereader-debug-mode)
5. [Contributing](#Contributing)
6. [Licence](#Licence)


## Contributors

| Name            | Email                       |
|-----------------|-----------------------------|
| James Brandreth | j.brandreth@ucl.ac.uk       |
| Jennifer Jiang  | jennifer.jiang.13@ucl.ac.uk |

## Installing

As the drug dosage extraction module uses Med7, you will need to download the model:
```bash
pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl
```
Then install MiADE:
```bash
pip install -e .
```
The `-e` flag sets the install to auto-update, useful when developing. Remove for production.

## Testing

This project uses pytest tests, which can be run with:
```bash
pytest ./tests/*
```
> Remember, if using a virtual environment, to install pytest within your environment itself, otherwise you will be using the system python install of pytest, which will use system python and will not find your modules.

## MIADE Notereader Debug Mode
The debug mode allows control over the MIADE backend through the EPIC Notereader interface.
Specific pre-configured contents and models can be triggered by typing in keywords in the Notereader text box, allowing users to seamlessly
test and debug the MIADE integration with EPIC Notereader.

There are 3 debug modes available, with varying degree of control over what concepts, models, and CDA specific content are returned to Notereader:
- ```PRESET```: bypasses the NLP model and returns a set of preloaded concepts
- ```CDA```: alters CDA specific content

**TO BE DEVELOPED**
- ```MODEL```: allows control over which version of model MIADE runs, once version control is in place

### Usage
The keywords to trigger the debug mode from Notereader are as follows:
```
***DEBUG PRESET [code]***
***DEBUG CDA [code]***
***DEBUG MODEL [code]***
```
Note that the keywords can appear anywhere in the text. ```CDA``` mode can be used with either ```PRESET``` mode or on its own
. When used alone, ```CDA```mode will return MIADE NLP results as normal and only alter the CDA field contents.
Example uses:
```
***DEBUG PRESET 0***
***DEBUG CDA 3***

The patient is allergic to bee venom. This is an example note and none of the text entered here matters in preset mode.
```
```
***DEBUG CDA 2***

The patient is diagnosed with liver failure. The NLP results will be returned in the CDA, along with any CDA content customisations.
```
### Configurations
Preloaded concepts and CDA customisation can be configured in [```configs/debug_configs.yml```](https://github.com/uclh-criu/miade/blob/master/src/miade/configs/debug_config.yml). An example config file is included as package data
with the MIADE module, but this can also be configured through the server.

The format is as following:
```yaml
Preset:
  [Code]:
    [Concept Name]:
      cui: [int]
      ontologies: [str]
      dosage: [Optional]
        dose:
          quantity: [int]
          unit: [{ucum} unit]
        duration:
          low: [date string]
          high: [date string]
        frequency:
          value: [float]
          unit: [h/d/w/m/y]
        route:
          cui: [NCI Thesaurus code]
          name: [str]
      reaction: [Optional]
        cui: [int]
        name: [str]
        ontologies: SNOMED CT
        severity:
          cui: [Act code]
          name: [str]
          ontologies: ActCode
CDA:
  [Code]:
      Problems:
        [CDA field name]: [value to insert]
      Medication:
      Allergy:

```
The following CDA fields currently allow customisation for testing. More can be added as needed, but require changes to the CDA parser.
#### Problems
```yaml
statusCode: whether problem is an active concern to clinician
actEffectiveTimeHigh: time when the problem stop become a concern to clinician -  there will always be a low (implicit)
observationEffectiveTimeLow: problem onset - can be a date in the past
observationEffectiveTimeHigh: problem resolve date
  ```
#### Medication
```yaml
consumableCodeSystemName: the code system name for medication - medication has separate <translation> blocks for name (NDC) and code (SNOMED CT) in test CDAs, maybe need experimentation?
consumableCodeSystemValue: template id of code system name
```
#### Allergy
```yaml
allergySectionCodeName: either "Propensity to adverse reaction" or "Allergy to substance" - they both appear in test CDAs
allergySectionCodeValue: the SNOMED code for the allergy section code
```

## Contributing
See [contributing](CONTRIBUTING.md)

## Licence
