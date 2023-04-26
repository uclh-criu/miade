# <img src="assets/miade-logo.png" width="40%">

[![Build Status](https://github.com/uclh-criu/miade/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/uclh-criu/miade/actions/workflows/ci.yml?query=Tests)

A set of tools for extracting formattable data from clinical notes stored in electronic health record systems.
For the reference server implementation, see: [miade-server](https://github.com/uclh-criu/miade-server).

Built with Cogstack's [MedCAT](https://github.com/CogStack/MedCAT) package.

## Contents

1. [Contributors](#Contributors)
2. [Installing](#Installing)
3. [Testing](#Testing)
4. [Contributing](#Contributing)
5. [Licence](#Licence)


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

## Contributing
See [contributing](CONTRIBUTING.md)

## Licence
