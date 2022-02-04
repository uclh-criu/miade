import os
from pathlib import Path
from typing import List, Optional

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
        snomed_data_path: Path,
        fdb_data_path: Path,
        elg_data_path: Path,
        config: Optional[Config] = None,
        model: str = "en_core_web_md",
    ):
        self.fdb_data_path = fdb_data_path
        self.elg_data_path = elg_data_path
        if config is not None:
            self.config = config
        else:
            self.config = Config()
        self.config.general["spacy_model"] = model

        self.snomed = Snomed(str(snomed_data_path))
        self.maker = CDBMaker(self.config)

    def preprocess_snomed(self, output_dir: Path = Path.cwd()) -> None:
        print("Exporting preprocessed SNOMED to csv...")
        df = self.snomed.to_concept_df()
        df.to_csv(output_dir / Path("preprocessed_snomed.csv"), index=False)

    def preprocess_fdb(self, output_dir: Path = Path.cwd()) -> None:
        preprocess_fdb(self.fdb_data_path).to_csv(
            output_dir / Path("preprocessed_fdb.csv"), index=False
        )

    def preprocess_elg(self, output_dir: Path = Path.cwd()) -> None:
        preprocess_elg(self.elg_data_path).to_csv(
            output_dir / Path("preprocessed_elg.csv"), index=False
        )

    def create_cdb(self, csv_paths: List[str]) -> CDB:
        cdb = self.maker.prepare_csvs(csv_paths, full_build=True)
        return cdb
