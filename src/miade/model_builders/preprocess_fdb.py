"""The database used was already largely prepared."""

from pathlib import Path
from pandas import DataFrame, read_csv


def preprocess_fdb(csv_path: Path) -> DataFrame:
    df = read_csv(csv_path)
    df["name"].apply(lambda x: x.lower())
    df["ontologies"] = "FDB"
    return df
