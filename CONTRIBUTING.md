# Contributing to MiADE

Thank you for considering contributing to MiADE!

## Code of Conduct

This project and everyone participating in it is governed by the MiADE Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to miade@uclh.net.

## I don't want to read this whole thing I just have a question!!!

INSERT LINK TO SOME SORT OF FORUM

## What should I know before I get started?

### Dependencies

You can find a list of dependencies in our [pyproject.toml](https://github.com/uclh-criu/miade/blob/master/pyproject.toml) file. MiADE is compatible with Python 3.8 and above.

To install the project with the dev dependencies, run:

```bash
pip install -e .[dev]
```
The `-e` flag sets the install to auto-update, useful when developing.

### Testing

MiADE uses [pytest](https://docs.pytest.org/en/8.2.x/), which can be run with:

```bash
pytest ./tests/*
```
> Remember, if using a virtual environment, to install pytest within your environment itself, otherwise you will be using the system python install of pytest, which will use system python and will not find your modules.

### Formatting and Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run:

```bash
ruff format 
ruff check --fix
```

## How Can I Contribute?

### Reporting Bugs

#### Before Submitting A Bug Report

* Check the documentation for tips on how to fix the issue on your own.
* Determine which repository the problem should be reported in - MiADE wraps around [MedCAT](https://github.com/CogStack/MedCAT/tree/master?tab=readme-ov-file), so if you encounter an issue related to MedCAT models, it is better to report it to these folks!
* Check if the issue has already been reported. If it has **and the issue is still open**, add a comment to the existing issue instead of opening a new one.

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub issues](https://github.com/uclh-criu/miade/issues). Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include links to files or GitHub projects, or copy/pasteable snippets, which you use in those examples.


### Your First Code Contribution

Unsure where to begin contributing to MiADE? You can start by looking through these `beginner` and `help-wanted` issues:

* [Good first issues](https://github.com/uclh-criu/miade/issues?q=is:open+is:issue+label:%22good+first+issue%22) - issues which should only require a few lines of code, and a test or two.
* [Help wanted issues](https://github.com/uclh-criu/miade/issues?q=is:open+is:issue+label:%22help+wanted%22) - issues which should be a bit more involved than `beginner` issues.

### Pull Requests

The process described here has several goals:

- Maintain MiADE's quality
- Fix problems that are important to users
- Engage the community in working toward the best possible version of MiADE
- Enable a sustainable system for MiADE's maintainers to review contributions

Please follow these steps to have your contribution considered by the maintainers:

1. Follow all instructions in [the template](https://github.com/uclh-criu/miade/blob/documentation/.github/PULL_REQUEST_TEMPLATE.md)
2. Follow the [styleguides](#styleguides)
3. After you submit your pull request, verify that all tests are passing

## Styleguides

We use [Google Python style docstring](https://google.github.io/styleguide/pyguide.html).

### Versioning
Versioning is performed through git tags, which should follow the [semantic versioning](https://semver.org/) approach prefixed with a "v".
E.g.: `v0.1.2-alpha`


Thank you for reading through the contributing guide and for your interest in making MiADE better. We look forward to your contributions!