"""This module is essentially the same as the MedCAT util preprocess_snomed.py
    with a few minor changes adapted to reading snomed UK folder paths"""

import os
import re
import hashlib
import pandas as pd


def parse_file(filename, first_row_header=True, columns=None):
    with open(filename, encoding="utf-8") as f:
        entities = [[n.strip() for n in line.split("\t")] for line in f]
        return pd.DataFrame(entities[1:], columns=entities[0] if first_row_header else columns)


class Snomed:
    """
    Pre-process SNOMED CT release files:
    Args:
        data_path:
            Path to the unzipped SNOMED CT folder
    """

    def __init__(
        self,
        data_path,
    ):
        self.data_path = data_path
        self.release = data_path.split("_")[-1][0:8]

    def to_concept_df(self, subset_list=None, exclusion_list=None):
        """
        :param: subset_list
        :param: exclusion_list
        :return: SNOMED CT concept DataFrame ready for MEDCAT CDB creation
        """
        snomed_releases = []
        paths = []
        if "Snapshot" in os.listdir(self.data_path):
            paths.append(self.data_path)
            snomed_releases.append(self.release)
        else:
            for folder in os.listdir(self.data_path):
                if "SnomedCT" in folder:
                    paths.append(os.path.join(self.data_path, folder))
                    snomed_releases.append(folder.split("_")[-1][0:8])
        if len(paths) == 0:
            raise FileNotFoundError("Incorrect path to SNOMED CT directory")

        df2merge = []
        for i, snomed_release in enumerate(snomed_releases):
            contents_path = os.path.join(paths[i], "Snapshot", "Terminology")
            uk_code = None
            snomed_v = None
            for f in os.listdir(contents_path):
                m = re.search(r"sct2_Concept_(.*)Snapshot_(.*)_\d*.txt", f)
                if m:
                    uk_code = m.group(1)
                    snomed_v = m.group(2)

            if uk_code is None or snomed_v is None:
                raise FileNotFoundError("Could not find file matching pattern")

            int_terms = parse_file(f"{contents_path}/sct2_Concept_{uk_code}Snapshot_{snomed_v}_{snomed_release}.txt")
            active_terms = int_terms[int_terms.active == "1"]
            del int_terms

            int_desc = parse_file(
                f"{contents_path}/sct2_Description_{uk_code}Snapshot-en_{snomed_v}_{snomed_release}.txt"
            )
            active_descs = int_desc[int_desc.active == "1"]
            del int_desc

            _ = pd.merge(
                active_terms,
                active_descs,
                left_on=["id"],
                right_on=["conceptId"],
                how="inner",
            )
            del active_terms
            del active_descs

            active_with_primary_desc = _[_["typeId"] == "900000000000003001"]  # active description
            active_with_synonym_desc = _[_["typeId"] == "900000000000013009"]  # active synonym
            del _
            active_with_all_desc = pd.concat([active_with_primary_desc, active_with_synonym_desc])

            active_snomed_df = active_with_all_desc[["id_x", "term", "typeId"]]
            del active_with_all_desc

            active_snomed_df.rename(
                columns={"id_x": "cui", "term": "name", "typeId": "name_status"},
                inplace=True,
            )
            active_snomed_df["ontologies"] = "SNOMED-CT"
            active_snomed_df["name_status"] = active_snomed_df["name_status"].replace(
                ["900000000000003001", "900000000000013009"], ["P", "A"]
            )
            active_snomed_df.reset_index(drop=True, inplace=True)

            temp_df = active_snomed_df[active_snomed_df["name_status"] == "P"][["cui", "name"]]
            temp_df["description_type_ids"] = temp_df["name"].str.extract(r"\((\w+\s?.?\s?\w+.?\w+.?\w+.?)\)$")
            active_snomed_df = pd.merge(
                active_snomed_df,
                temp_df.loc[:, ["cui", "description_type_ids"]],
                on="cui",
                how="left",
            )
            del temp_df

            # Hash semantic tag to get a 8 digit type_id code
            # need to drop Nans in the dataframe if there are any
            active_snomed_df["type_ids"] = (
                active_snomed_df["description_type_ids"]
                .dropna()
                .apply(lambda x: int(hashlib.sha256(x.encode("utf-8")).hexdigest(), 16) % 10**8)
            )
            df2merge.append(active_snomed_df)

        df = pd.concat(df2merge).reset_index(drop=True)

        if subset_list is not None:
            df = df.merge(subset_list, how="inner", on="cui")

        if exclusion_list is not None:
            df = df[~df["cui"].isin(exclusion_list.cui)]

        return df.reset_index(drop=True)

    def list_all_relationships(self):
        """
        SNOMED CT provides a rich set of inter-relationships between concepts.
        :return: List of all SNOMED CT relationships
        """
        snomed_releases = []
        paths = []
        if "Snapshot" in os.listdir(self.data_path):
            paths.append(self.data_path)
            snomed_releases.append(self.release)
        else:
            for folder in os.listdir(self.data_path):
                if "SnomedCT" in folder:
                    paths.append(os.path.join(self.data_path, folder))
                    snomed_releases.append(folder.split("_")[-1][0:8])
        if len(paths) == 0:
            raise FileNotFoundError("Incorrect path to SNOMED CT directory")

        all_rela = []
        for i, snomed_release in enumerate(snomed_releases):
            contents_path = os.path.join(paths[i], "Snapshot", "Terminology")
            uk_code = None
            snomed_v = None
            for f in os.listdir(contents_path):
                m = re.search(r"sct2_Relationship_(.*)Snapshot_(.*)_\d*.txt", f)
                if m:
                    uk_code = m.group(1)
                    snomed_v = m.group(2)

            if uk_code is None or snomed_v is None:
                raise FileNotFoundError("Could not find file matching pattern")

            int_relat = parse_file(
                f"{contents_path}/sct2_Relationship_{uk_code}Snapshot_{snomed_v}_{snomed_release}.txt"
            )
            active_relat = int_relat[int_relat.active == "1"]
            del int_relat

            all_rela.extend([relationship for relationship in active_relat["typeId"].unique()])

        return all_rela
