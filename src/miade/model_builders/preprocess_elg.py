from pandas import DataFrame, read_csv, isnull
from pathlib import Path


def preprocess_elg(filepath: Path) -> DataFrame:
    df = read_csv(filepath)

    df = df[df.RECORD_STATE_NAME != "Deleted"]

    df["cui"] = df.ALLERGEN_ID
    df["name"] = df.ALLERGEN_NAME.str.lower()
    df["ontologies"] = "ELG"
    df["name_status"] = "P"

    return df[["cui", "name", "ontologies", "name_status"]]
