from pathlib import Path
from typing import List, Optional

from medcat.config import Config
from medcat.cdb_maker import CDBMaker

from .preprocess_snomeduk import Snomed


class CDBBuilder(object):
    """Builds cdb from snomed data"""
    def __init__(self, data_path: Path, config: Optional[Config] = None, model: str = 'en_core_web_md'):
        if config is not None:
            self.config = config
        else:
            self.config = Config()
        self.config.general['spacy_model'] = model

        self.snomed = Snomed(str(data_path))
        self.maker = CDBMaker(self.config)

    def preprocess_snomed(self) -> None:
        print("Exporting preprocessed SNOMED to csv...")
        df = self.snomed.to_concept_df()
        df.to_csv('preprocessed_snomed.csv', index=False)

    def create_cdb(self, csv_path: List[str]) -> None:
        cdb = self.maker.prepare_csvs(csv_path, full_build=True)
        cdb.save('cdb.dat')

