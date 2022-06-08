from setuptools import setup, find_packages

setup(
    name="miade",
    version="0.2",
    description="",
    packages=find_packages(),
    include_package_data=True,
    package_data={"miade": ["configs/*.yml"]},
    setup_requires=[
        "wheel",
    ],
    install_requires=[
        "medcat==1.2.8",
        "spacy==3.1.0",
        "typing==3.7.4.3",
        "typer",
        "pathlib",
    ],
    scripts=["scripts/miade"],
)
