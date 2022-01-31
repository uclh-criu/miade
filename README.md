# MIADE

A set of tools for extracting formattable data from clinical notes stored in electronic health record systems.

For the reference server implementation, see: [nlp-engine-server](https://github.com/uclh-criu/nlp-engine-server).

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

```bash
pip install -e ./src
```
The `-e` flag sets the install to auto-update, useful when developing. Remove for production.

## Testing

This project uses pytest tests, which can be run with:
```bash
pytest ./tests/*
```
> Remember, if using a virtual environment, to install pytest within your environment itself, otherwise you will be using the system python install of pytest, which will use system python and will not find your modules.

## Contributing

## Licence
