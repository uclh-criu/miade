import os
from pathlib import Path
from typing import List, Optional

from shutil import rmtree
import pandas as pd
from medcat.cdb import CDB
from medcat.config import Config
from medcat.cdb_maker import CDBMaker

from .preprocess_snomeduk import Snomed
from .preprocess_fdb import preprocess_fdb
from .preprocess_elg import preprocess_elg


class CDBBuilder(object):
    """Builds cdb from snomed data"""

    def __init__(
        self,
        snomed_data_path: Optional[Path],
        fdb_data_path: Optional[Path],
        elg_data_path: Optional[Path],
        snomed_subset_path: Optional[Path],
        temp_dir: Path,
        snomed_exclusions_path: Optional[Path] = None,
        config: Optional[Config] = None,
        model: str = "en_core_web_md",
    ):
        self.temp_dir = temp_dir
        self.fdb_data_path = fdb_data_path
        self.elg_data_path = elg_data_path
        self.snomed_subset_path = snomed_subset_path
        self.snomed_exclusions_path = snomed_exclusions_path
        if config is not None:
            self.config = config
        else:
            self.config = Config()
        self.config.general["spacy_model"] = model

        if snomed_data_path:
            self.snomed = Snomed(str(snomed_data_path))
        self.maker = CDBMaker(self.config)

        self.temp_dir.mkdir()

    def __del__(self):
        print(self.temp_dir)
        rmtree(self.temp_dir)

    def preprocess_snomed(self, output_dir: Path = Path.cwd()) -> Path:
        print("Exporting preprocessed SNOMED to csv...")

        if self.snomed_subset_path is not None:
            snomed_subset = pd.read_csv(self.snomed_subset_path, header=0)
            if "cui" not in snomed_subset.columns.values:
                snomed_subset.rename(columns={"conceptId": "cui"}, inplace=True)
            snomed_subset["cui"] = snomed_subset.cui.apply(lambda x: f"SNO-{x}")
        else:
            snomed_subset = None

        if self.snomed_exclusions_path is not None:
            snomed_exclusions = pd.read_csv(
                self.snomed_exclusions_path, sep="\n", header=None
            )
            snomed_exclusions.columns = ["cui"]
            snomed_exclusions["cui"] = snomed_exclusions.cui.apply(lambda x: f"SNO-{x}")
        else:
            snomed_exclusions = None

        output_file = output_dir / Path("preprocessed_snomed.csv")
        df = self.snomed.to_concept_df(
            subset_list=snomed_subset, exclusion_list=snomed_exclusions
        )
        df.to_csv(output_file, index=False)
        return output_file

    def preprocess_fdb(self, output_dir: Path = Path.cwd()) -> Path:
        output_file = output_dir / Path("preprocessed_fdb.csv")
        preprocess_fdb(self.fdb_data_path).to_csv(output_file, index=False)
        return output_file

    def preprocess_elg(self, output_dir: Path = Path.cwd()) -> Path:
        output_file = output_dir / Path("preprocessed_elg.csv")
        preprocess_elg(self.elg_data_path).to_csv(output_file, index=False)
        return output_file

    def preprocess(self):
        self.vocab_files = []
        if self.snomed:
            self.vocab_files.append(self.preprocess_snomed(self.temp_dir))
        if self.fdb_data_path:
            self.vocab_files.append(self.preprocess_fdb(self.temp_dir))
        if self.elg_data_path:
            self.vocab_files.append(self.preprocess_elg(self.temp_dir))

    def create_cdb(self) -> CDB:
        cdb = self.maker.prepare_csvs(self.vocab_files, full_build=True)
        return cdb
