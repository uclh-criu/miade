import os
import json
import datetime
import logging

import typer
import yaml
import numpy as np
import pandas as pd

from pathlib import Path
from shutil import rmtree
from typing import Optional, List
from pydantic import BaseModel

from tokenizers import ByteLevelBPETokenizer
from gensim.models import Word2Vec
from medcat.cat import CAT
from medcat.meta_cat import MetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBPE

from miade.model_builders import CDBBuilder
from miade.utils.miade_cat import MiADE_CAT
from miade.utils.miade_meta_cat import MiADE_MetaCAT

log = logging.getLogger("miade")


def create_metacats(
    tokenizer_path: Path,
    category_names: List[str],
    output: Optional[Path] = typer.Argument(Path.cwd()),
):
    log.info(f"Loading tokenizer from {tokenizer_path}/...")
    tokenizer = TokenizerWrapperBPE.load(str(tokenizer_path))
    log.info(f"Loading embeddings from embeddings.npy...")
    embeddings = np.load(str(os.path.join(tokenizer_path, "embeddings.npy")))

    assert len(embeddings) == tokenizer.get_size(), (
        f"Tokenizer and embeddings not the same size {len(embeddings)}, "
        f"{tokenizer.get_size()}"
    )

    metacat = MetaCAT(tokenizer=tokenizer, embeddings=embeddings)
    for category in category_names:
        metacat.config.general["description"] = f"MiADE blank {category} MetaCAT model"
        metacat.config.general["category_name"] = category
        metacat.save(str(os.path.join(output, f"meta_{category}")))
        log.info(f"Saved meta_{category} at {output}")


def train_metacat(
    model_path: Path,
    annotation_path: Path,
    synthetic_data_path: Optional[Path] = None,
    nepochs: int = 50,
    cntx_left: int = 20,
    cntx_right: int = 15,
    description: str = None,
):
    mc = MiADE_MetaCAT.load(str(model_path))

    if description is None:
        description = f"MiADE meta-annotations model {model_path.stem} trained on {annotation_path.stem}"

    mc.config.general["description"] = description
    mc.config.general["category_name"] = model_path.stem.split("_")[
        -1
    ]  # meta folder name should be e.g. meta_presence
    mc.config.general["cntx_left"] = cntx_left
    mc.config.general["cntx_right"] = cntx_right
    mc.config.train["nepochs"] = nepochs

    log.info(
        f"Starting MetaCAT training for {mc.config.general['category_name']} for {nepochs} epoch(s) "
        f"with annotation file {annotation_path}"
    )
    report = mc.train(
        json_path=str(annotation_path),
        synthetic_csv_path=str(synthetic_data_path),
        save_dir_path=str(model_path),
    )
    training_stats = {mc.config.general["category_name"]: report}

    report_save_name = os.path.join(model_path, "training_report.json")
    with open(report_save_name, "w") as f:
        json.dump(training_stats, f)

    log.info(f"Saved training report at {report_save_name}")
