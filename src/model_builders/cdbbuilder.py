import os
from pathlib import Path
from zipfile import ZipFile
from typing import List

from medcat.config import Config
from medcat.cdb_maker import CDBMaker
from preprocess_snomed_uk import Snomed


class CDBBuilder(object):
    """Builds cdb from snomed data"""
    def __init__(self, data_path: Path, config: Config = None, model: str = 'en_core_sci_md'):
        if config is not None:
            self.config = config
        else:
            self.config = Config()
        self.config.general['spacy_model'] = model

        print(f"Unzipping{data_path}...")
        self.data_path = os.path.join('snomed' + data_path.stem)
        ZipFile(data_path).extractall(self.data_path)

        self.snomed = Snomed(self.data_path)
        self.maker = CDBMaker(self.config)

    def preprocess_snomed(self) -> None:
        print("Exporting preprocessed SNOMED to csv...")
        df = self.snomed.to_concept_df()
        df.to_csv('preprocessed_snomed.csv', index=False)

    def create_cdb(self, csv_path: List[str]) -> None:
        cdb = self.maker.prepare_csvs(csv_path, full_build=True)
        cdb.save('SNOMED_cdb.dat')

