from setuptools import setup, find_packages

setup(
    name="miade",
    version="0.1",
    description="",
    packages=find_packages(),
    include_package_data=True,
    package_data={"miade": ["data/*.csv"]},
    setup_requires=[
        "wheel",
    ],
    install_requires=[
        "medcat==1.2.8",
        "spacy==3.1.0",
        "typing==3.7.4.3",
        "pandas",
        "typer",
        "pathlib"
    ],
    scripts=[
        "scripts/miade"
    ]
)
