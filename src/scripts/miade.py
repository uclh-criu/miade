#!/usr/bin/env python
import os
import json
import datetime
import logging
import re
from typing_extensions import Dict

import typer
import yaml
import tomllib
import numpy as np
import pandas as pd
import requests

from pathlib import Path, PosixPath
from shutil import rmtree
from typing import Optional, List
from pydantic import BaseModel

from tokenizers import ByteLevelBPETokenizer
from gensim.models import Word2Vec
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.meta_cat import MetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBPE

from miade.model_builders import CDBBuilder
from miade.utils.miade_cat import MiADE_CAT
from miade.utils.miade_meta_cat import MiADE_MetaCAT

log = logging.getLogger("miade")
log.setLevel("INFO")

app = typer.Typer()


def _sanitise_filename(filename: Path) -> Path:
    """
    Sanitizes a filename by removing or replacing invalid characters.
    """
    # Replace invalid characters with underscores
    clean_name = re.sub(r'[<>:"/\\|?*]', "_", str(filename.name))
    # Remove leading/trailing spaces and dots
    clean_name = clean_name.strip(". ")
    return Path(clean_name)


def _ensure_unique_path(path: Path) -> Path:
    """
    Ensures a file path is unique by adding a number suffix if necessary.
    """
    if not path.exists():
        return path

    base = path.stem
    extension = path.suffix
    counter = 1

    while True:
        new_path = path.parent / f"{base}_{counter}{extension}"
        if not new_path.exists():
            return new_path
        counter += 1


class YAMLConfig(BaseModel):
    @classmethod
    def from_yaml_file(cls, config_filepath: Path):
        with config_filepath.open("r") as stream:
            config_dict = yaml.safe_load(stream)
            return cls(**config_dict)


class CLI_Config(YAMLConfig):
    snomed_data_path: Optional[Path] = None
    fdb_data_path: Optional[Path] = None
    elg_data_path: Optional[Path] = None
    snomed_subset_path: Optional[Path] = None
    snomed_exclusions_path: Optional[Path] = None
    medcat_config_file: Optional[Path] = None
    training_data_path: Path
    output_dir: Path


class URL(BaseModel):
    path: str

    def download(self, dir: Path) -> Path:
        log = logging.getLogger(__name__)
        log.info(f"Downloading {self.path} to {dir}")

        # Make initial request with stream=True to examine headers
        response = requests.get(self.path, stream=True, allow_redirects=True)
        response.raise_for_status()

        # Try to get filename from Content-Disposition header
        if "Content-Disposition" in response.headers:
            content_disposition = response.headers["Content-Disposition"]
            matches = re.findall(r"filename[^;=\n]*=([\w\.\"\']*[\w\.])", content_disposition)
            if matches:
                # Clean up the filename by removing quotes
                filename = matches[0].strip("\"'")
                filename = Path(filename)

        # Try to get filename from URL path
        else:
            url_path = self.path.split("/")[-1].split("?")[0]
            if url_path and "." in url_path:  # Basic check if it looks like a filename
                filename = Path(url_path)

            # Generate filename based on Content-Type
            else:
                content_type = response.headers.get("Content-Type", "").split(";")[0]
                ext_map = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "application/pdf": ".pdf",
                    "text/plain": ".txt",
                    "application/json": ".json",
                    # Add more mappings as needed
                    "": ".bin",  # Default extension for unknown types
                }
                ext = ext_map.get(content_type, ".bin")
                # Generate a filename using timestamp or handle based on your needs
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = Path(f"download_{timestamp}{ext}")

        # Ensure the filename is safe and unique
        safe_filename = dir / _sanitise_filename(filename)
        safe_filename = _ensure_unique_path(safe_filename)

        # Download the file in chunks to handle large files efficiently
        with open(safe_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)

        log.info(f"Download completed: {safe_filename}")
        return safe_filename


class Location(BaseModel):
    location: Path | URL

    @classmethod
    def from_dict(cls, config_dict: Dict):
        assert len(config_dict.keys()) == 1  # must be exactly one source

        path = config_dict.get("path")
        if path:
            return cls(location=Path(path))

        url = config_dict.get("url")
        if url:
            return cls(location=URL(path=url))

    def get_or_download(self, dir: Path) -> Path:
        filepath = self.location
        if type(self.location) == URL:
            filepath = self.location.download(dir)
        return filepath


class Source(BaseModel):
    location: Optional[Location] = None
    data: Optional[Location] = None

    @classmethod
    def from_dict(cls, config_dict: Dict):
        location = config_dict.get("location")
        if location:
            return cls(location=Location.from_dict(location))

        data = config_dict.get("data")
        if data:
            return cls(data=Location.from_dict(data))


class MakeConfig(BaseModel):
    model: Optional[Source]
    vocab: Optional[Source]
    cdb: Optional[Source]
    meta_models: Dict[str, Source]

    def __str__(self) -> str:
        return (
            f"""model: {self.model}\nCDB:\n\t{self.cdb}\nvocab:\n\t{self.vocab}\nmeta-models:\n\t{self.meta_models}"""
        )

    @classmethod
    def from_yaml_string(cls, s: str):
        config_dict = yaml.safe_load(s)

        model = config_dict.get("model")
        if model:
            model = Source.from_dict(model)

        cdb_config = config_dict.get("cdb")
        cdb = None
        if cdb_config:
            cdb = Source.from_dict(cdb_config)

        vocab_config = config_dict.get("vocab")
        vocab = None
        if vocab_config:
            vocab = Source.from_dict(vocab_config)

        meta_config = config_dict.get("meta-models")
        meta_models = {}
        if meta_config:
            meta_models: Dict[str, Source] = {
                name: Source.from_dict(config) for d in meta_config for name, config in d.items() if d
            }
        return cls(
            model=model,
            cdb=cdb,
            vocab=vocab,
            meta_models=meta_models,
        )

    @classmethod
    def from_yaml_file(cls, config_filepath: Path):
        with config_filepath.open("r") as file:
            return cls.from_yaml_string(file.read())


@app.command()
def make(config_filepath: Path, temp_dir: Path = Path("./.temp")):
    """
    Build a model from a specification file.
    """

    log.info(f"Making temporary directory: {temp_dir}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    config = MakeConfig.from_yaml_file(config_filepath)
    # print(config)

    if config.model:
        log.info(f"Loading model pack from {config.model.location}")
        model = CAT.load_model_pack(config.model.location.get_or_download(temp_dir))
    else:
        log.info(f"Creating new model pack")
        model = CAT()

    if config.cdb:
        path = config.cdb.source
        model.cdb = CDB.load(path=path)


@app.command()
def build_model_pack(
    cdb_data_path: Path,
    vocab_path: Path,
    description: Optional[str] = None,
    ontology: Optional[str] = None,
    tag: Optional[str] = None,
    output: Optional[Path] = typer.Argument(Path.cwd()),
    temp: Optional[Path] = typer.Argument(Path.cwd() / Path(".temp")),
):
    # builds from cdb already in format and vocab from another medcat model
    log.info(f"Building CDB from {str(cdb_data_path)}...")
    cdb_builder = CDBBuilder(temp_dir=temp, custom_data_paths=[cdb_data_path])
    cdb_builder.preprocess()
    cdb = cdb_builder.create_cdb()
    log.info(f"CDB name2cui check: {list(cdb.cui2names.items())[:10]}")

    log.info(f"Creating model pack with vocab from {str(vocab_path)}...")
    vocab_cat = CAT.load_model_pack(str(vocab_path))
    cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab_cat.vocab)

    if description is None:
        log.info("Automatically populating description field of model card...")
        split_tag = ""
        if tag is not None:
            split_tag = " ".join(tag.split("_")) + " "
        description = (
            f"MiADE {split_tag}untrained model built using cdb from "
            f"{cdb_data_path.stem}{cdb_data_path.suffix} and vocab "
            f"from model {vocab_path.stem}"
        )

    cat.config.version["location"] = str(output)
    cat.config.version["description"] = description
    cat.config.version["ontology"] = ontology

    current_date = datetime.datetime.now().strftime("%b_%Y")
    name = f"miade_{tag}_blank_modelpack_{current_date}" if tag is not None else f"miade_blank_modelpack_{current_date}"

    cat.create_model_pack(str(output), name)
    log.info(f"Saved model pack at {output}/{name}_{cat.config.version['id']}")


@app.command()
def train(
    model: Path,
    data: Path,
    checkpoint: int = 5000,
    description: Optional[str] = None,
    tag: Optional[str] = None,
    train_partial: Optional[int] = None,
    output: Optional[Path] = typer.Argument(Path.cwd()),
):
    if data.suffix == ".csv":
        log.info(f"Loading text column of csv file {data}...")
        df = pd.read_csv(data)
        training_data = df.text.to_list()
    else:
        log.info(f"Loading text file {data}...")
        with data.open("r") as d:
            training_data = [line.strip() for line in d]
    log.info(f"Training data length: {len(training_data)}")
    log.info(f"Data check: {training_data[0][:100]}")

    if train_partial:
        log.info(f"Partial training first {train_partial} documents")
        training_data = training_data[:train_partial]

    cat = CAT.load_model_pack(str(model))

    if checkpoint:
        log.info(f"Checkpoint steps configured to {checkpoint}")
        cat.config.general["checkpoint"]["steps"] = checkpoint
        cat.config.general["checkpoint"]["output_dir"] = os.path.join(Path.cwd(), "checkpoints")

    cat.train(training_data)

    if description is None:
        log.info("Automatically populating description field of model card...")
        split_tag = ""
        if tag is not None:
            split_tag = " ".join(tag.split("_")) + " "
        description = f"MiADE {split_tag}unsupervised trained model trained on text dataset {data.stem}{data.suffix}"

    cat.config.version["description"] = description
    cat.config.version["location"] = str(output)

    current_date = datetime.datetime.now().strftime("%b_%Y")
    name = (
        f"miade_{tag}_unsupervised_trained_modelpack_{current_date}"
        if tag is not None
        else f"miade_unsupervised_trained_modelpack_{current_date}"
    )

    cat.create_model_pack(str(output), name)
    log.info(f"Saved model pack at {output}/{name}_{cat.config.version['id']}")


@app.command()
def train_supervised(
    model: Path,
    annotations_path: Path,
    synthetic_data_path: Optional[Path] = None,
    nepochs: int = 1,
    use_filters: bool = False,
    print_stats: bool = True,
    train_from_false_positives: bool = True,
    is_resumed: bool = False,
    description: Optional[str] = None,
    tag: Optional[str] = None,
    output: Optional[Path] = typer.Argument(Path.cwd()),
):
    cat = MiADE_CAT.load_model_pack(str(model))

    log.info(f"Starting {nepochs} epoch(s) supervised training with {annotations_path}")
    fp, fn, tp, p, r, f1, cui_counts, examples = cat.train_supervised(
        data_path=str(annotations_path),
        synthetic_data_path=synthetic_data_path,
        nepochs=nepochs,
        use_filters=use_filters,
        print_stats=print_stats,
        train_from_false_positives=train_from_false_positives,
        is_resumed=is_resumed,
    )

    # populate the description field in versioning
    if description is None:
        log.info("Automatically populating description field of model card...")
        split_tag = ""
        if tag is not None:
            split_tag = " ".join(tag.split("_")) + " "
        description = f"MiADE {split_tag}supervised trained model with annotations file {annotations_path.stem}"

    cat.config.version["description"] = description
    cat.config.version["location"] = str(output)

    current_date = datetime.datetime.now().strftime("%b_%Y")
    name = (
        f"miade_{tag}_supervised_trained_modelpack_{current_date}"
        if tag is not None
        else f"miade_supervised_trained_modelpack_{current_date}"
    )

    cat.create_model_pack(str(output), name)

    # dump the training stats into a json file for reference(they are very long)
    model_id = cat.config.version["id"]
    training_stats = {
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "p": p,
        "r": r,
        "f1": f1,
        "cui_counts": cui_counts,
        "examples": examples,
    }

    stats_save_name = os.path.join(output, f"supervised_training_stats_{model_id}.json")
    with open(stats_save_name, "w") as f:
        json.dump(training_stats, f)

    log.info(f"Saved training stats at {stats_save_name}")


@app.command()
def create_bbpe_tokenizer(
    train_data: Path,
    name: Optional[str] = "bbpe_tokenizer",
    output: Optional[Path] = typer.Argument(Path.cwd()),
):
    # Create, train on text and save the tokenizer
    log.info(f"Creating BPE tokenizer and start training on {train_data}...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(str(train_data))
    tokenizer.add_tokens(["<PAD>"])

    save_path = os.path.join(output, name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    tokenizer.save_model(str(save_path), "bbpe")
    log.info(f"Saved tokenizer model files vocab.json and merges.txt to {save_path}")

    log.info(f"Started tokenizing text from {train_data}...")
    data = []
    step = 0
    with open(train_data) as f:
        for line in f:
            if step % 5000 == 0:
                log.info(f"{step} DONE")
            data.append(tokenizer.encode(line).tokens)
            step += 1

    log.info("Started training word2vec model with tokenized text...")
    w2v = Word2Vec(data, vector_size=300, min_count=1)

    log.info(f"Creating embeddings matrix, vocab size {tokenizer.get_vocab_size()}")
    embeddings = []
    step = 0
    for i in range(tokenizer.get_vocab_size()):
        word = tokenizer.id_to_token(i)
        if step % 5000 == 0:
            log.info(f"{step} DONE")
        if word in w2v.wv:
            embeddings.append(w2v.wv[word])
        else:
            embeddings.append(np.random.rand(300))
        step += 1

    log.info(f"Embedding length: {len(embeddings)}")

    embeddings_save_name = os.path.join(save_path, "embeddings.npy")
    np.save(open(str(embeddings_save_name), "wb"), np.array(embeddings))
    log.info(f"Saved embeddings at {embeddings_save_name}")


@app.command()
def create_metacats(
    tokenizer_path: Path,
    category_names: List[str],
    output: Optional[Path] = typer.Argument(Path.cwd()),
):
    log.info(f"Loading tokenizer from {tokenizer_path}/...")
    tokenizer = TokenizerWrapperBPE.load(str(tokenizer_path))
    log.info("Loading embeddings from embeddings.npy...")
    embeddings = np.load(str(os.path.join(tokenizer_path, "embeddings.npy")))

    assert len(embeddings) == tokenizer.get_size(), (
        f"Tokenizer and embeddings not the same size {len(embeddings)}, " f"{tokenizer.get_size()}"
    )

    metacat = MetaCAT(tokenizer=tokenizer, embeddings=embeddings)
    for category in category_names:
        metacat.config.general["description"] = f"MiADE blank {category} MetaCAT model"
        metacat.config.general["category_name"] = category
        metacat.save(str(os.path.join(output, f"meta_{category}")))
        log.info(f"Saved meta_{category} at {output}")


@app.command()
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
    mc.config.general["category_name"] = model_path.stem.split("_")[-1]  # meta folder name should be e.g. meta_presence
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


@app.command()
def add_metacat_models(
    model: Path,
    meta_cats_path: List[Path],
    description: str = None,
    output: Optional[Path] = typer.Argument(Path.cwd()),
):
    cat = CAT.load_model_pack(str(model))

    meta_cats = []
    categories = []
    stats = {}
    for metacat_path in meta_cats_path:
        mc = MetaCAT.load(str(metacat_path))
        meta_cats.append(mc)
        categories.append(mc.config.general["category_name"])
        log.info(f"Loaded MetaCAT model {categories[-1]}")
        # get training stats if there are any
        report_path = os.path.join(metacat_path, "training_report.json")
        if os.path.exists(report_path):
            log.info(f"Found training_report.json from {str(metacat_path)}")
            with open(report_path) as f:
                report = json.load(f)
            stats[categories[-1]] = report

    log.info(f"Creating CAT with MetaCAT models {categories}...")
    cat_w_meta = CAT(cdb=cat.cdb, vocab=cat.vocab, config=cat.config, meta_cats=meta_cats)

    if description is None:
        log.info("Automatically populating description field of model card...")
        description = cat.config.version["description"] + " | Packaged with MetaCAT model(s) " + ", ".join(categories)
    cat.config.version["description"] = description

    for category in categories:
        cat.config.version["performance"]["meta"] = stats.get(category)

    save_name = model.stem.rsplit("_", 1)[0] + "_w_meta"
    cat_w_meta.create_model_pack(str(output), save_name)
    log.info(f"Saved model pack at {output}/{save_name}_{cat.config.version['id']}")


@app.command()
def rename_model_pack(
    model: Path,
    new_name: str,
    remove_old: bool = True,
    description: Optional[str] = None,
    location: Optional[str] = None,
    ontology: Optional[str] = None,
    performance: Optional[str] = None,
    output: Optional[Path] = typer.Argument(Path.cwd()),
):
    cat = CAT.load_model_pack(str(model))
    if description is not None:
        log.info("Adding description to model card...")
        cat.config.version["description"] = description
    if location is not None:
        log.info("Adding location to model card...")
        cat.config.version["location"] = location
    if ontology is not None:
        log.info("Adding ontology to model card...")
        cat.config.version["ontology"] = ontology
    if performance is not None:
        log.info("Adding performance to model card...")
        cat.config.version["performance"] = performance

    cat.create_model_pack(str(output), new_name)
    log.info(f"Saved model pack at {output}/{new_name}_{cat.config.version['id']}")

    # remove old model
    if remove_old:
        log.info(f"Removing old model {str(model)}")
        os.remove(str(model) + ".zip")
        rmtree(model)


if __name__ == "__main__":
    app()
