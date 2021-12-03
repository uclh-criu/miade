from setuptools import setup, find_packages

setup(
    name="nlp_engine_core",
    version="0.1",
    description="",
    packages=find_packages(),
    setup_requires=[
        "wheel",
    ],
    install_requires=["medcat", "pathlib"],
)
