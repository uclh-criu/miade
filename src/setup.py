from setuptools import setup, find_packages

setup(
    name="miade",
    version="0.1",
    description="",
    packages=find_packages(),
    setup_requires=[
        "wheel",
    ],
    install_requires=[
        "medcat==1.2.6",
        "spacy==3.1.0",
        "typing==3.7.4.3",
    ],
)
