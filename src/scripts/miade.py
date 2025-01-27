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
import polars as pl
import requests

from pathlib import Path, PosixPath
from shutil import rmtree
from typing import Optional, List
from pydantic import BaseModel, validator

from tokenizers import ByteLevelBPETokenizer
from gensim.models import Word2Vec
from medcat.cat import CAT
from medcat.config import Config
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.meta_cat import MetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBPE
from medcat.config_meta_cat import ConfigMetaCAT

from miade.model_builders import CDBBuilder
from miade.utils.miade_cat import MiADE_CAT
from miade.utils.miade_meta_cat import MiADE_MetaCAT

log = logging.getLogger("miade")
log.setLevel("INFO")

app = typer.Typer()


def _add_zip(a, b):
    return {key: a.get(key, 0) + b.get(key, 0) for key in set(list(a.keys()) + list(b.keys()))}


def _get_label_counts_for_annotation_export(export, category):
    counts = {}
    for label in [
        ann["meta_anns"][category]["value"]
        for doc in export["projects"][0]["documents"]
        for ann in doc["annotations"]
        if (ann["validated"] == True and ann["deleted"] == False and ann["alternative"] == False and ann["meta_anns"].get(category))
    ]:
        if counts.get(label):
            counts[label] += 1
        else:
            counts[label] = 1
    return counts


def _get_label_counts_for_synthetic_data(data, category):
    counts = {}
    for label in data[category]:
        if counts.get(label):
            counts[label] += 1
        else:
            counts[label] = 1
    return counts


def _prep_synthetic_training_data(
    synth_df,
    name,
):
    return synth_df[["text", name]]


def _sanitise_filename(filename: Path) -> Path:
    """
    Sanitizes a filename by removing or replacing invalid characters.
    """
    # Replace invalid characters with underscores
    clean_name = re.sub(r'[<>:"/\\|?*]', "_", str(filename.name))
    # Remove leading/trailing spaces and dots
    clean_name = clean_name.strip(". ")
    return Path(clean_name)


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

        # Download the file in chunks to handle large files efficiently
        with open(safe_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)

        log.info(f"Download completed: {safe_filename}")
        return safe_filename


class Location(BaseModel):
    path: Optional[Path]
    url: Optional[URL]

    @validator("path", "url")
    def validate_choice(cls, v, values):
        if (v is not None) == ((values.get("path") is not None) + (values.get("url") is not None)) % 2:
            raise ValueError("Only one of path or url can be set")
        return v

    def __str__(self):
        if self.url:
            return f"url: {self.url}"
        else:
            return f"path: {self.path}"

    def get_or_download(self, dir: Path) -> Path:
        if self.url:
            filepath = self.url.download(dir)
        elif self.path:
            filepath = self.path
        else:
            raise Exception("attempted to get a location with no url or path")
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


class SupervisedTrainingConfig(BaseModel):
    annotation_data: Location
    synthetic_data: Optional[Location] = None
    nepochs: int = 1
    use_filters: bool = False
    train_from_false_positives: bool = True


class MetaModelConfig(BaseModel):
    annotations: Location
    synthetic_data: Optional[Location]
    config: ConfigMetaCAT = ConfigMetaCAT()
    balance_weights: bool = False


class MetaModelsConfig(BaseModel):
    tokenizer: Source
    models: Dict[str, MetaModelConfig]


class MakeConfig(BaseModel):
    tag: Optional[str]
    description: Optional[str]
    ontology: Optional[str]
    model: Optional[Source]
    medcat_model_config: Optional[Dict]
    vocab: Optional[Source]
    cdb: Optional[Source]
    context_embedding_training_data: Optional[Location]
    supervised_training: Optional[SupervisedTrainingConfig]
    meta_models: Optional[MetaModelsConfig]


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


def _create_bbpe_tokenizer(
    train_data: Path,
) -> tuple[ByteLevelBPETokenizer, List]:
    log.info(f"Creating BPE tokenizer and start training on {train_data}...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(str(train_data))
    tokenizer.add_tokens(["<PAD>"])

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
    return tokenizer, embeddings


@app.command()
def create_bbpe_tokenizer(
    train_data: Path,
    name: str = "bbpe_tokenizer",
    output: Path = typer.Argument(Path.cwd()),
):
    # Create, train on text and save the tokenizer
    tokenizer, embeddings = _create_bbpe_tokenizer(train_data)

    save_path = output / Path(name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    tokenizer.save_model(str(save_path), "bbpe")
    log.info(f"Saved tokenizer model files vocab.json and merges.txt to {save_path}")

    embeddings_save_name = os.path.join(save_path, "embeddings.npy")
    np.save(open(str(embeddings_save_name), "wb"), np.array(embeddings))
    log.info(f"Saved embeddings at {embeddings_save_name}")


def _create_metacats(
    tokenizer: ByteLevelBPETokenizer,
    embeddings: List[np.ndarray],
    category_names: List[str],
) -> List[TokenizerWrapperBPE]:
    tokenizer = TokenizerWrapperBPE(tokenizer)
    assert len(embeddings) == tokenizer.get_size(), (
        f"Tokenizer and embeddings not the same size {len(embeddings)}, {tokenizer.get_size()}"
    )

    metacats = []
    for category in category_names:
        metacat = MetaCAT(tokenizer=tokenizer, embeddings=embeddings)
        metacat.config.general["description"] = f"MiADE blank {category} MetaCAT model"
        metacat.config.general["category_name"] = category
        metacats.append(metacat)

    return metacats


@app.command()
def create_metacats(
    tokenizer_path: Path,
    category_names: List[str],
    output: Path = typer.Argument(Path.cwd()),
):
    log.info(f"Loading tokenizer from {tokenizer_path}/...")
    tokenizer = TokenizerWrapperBPE.load(str(tokenizer_path))
    log.info("Loading embeddings from embeddings.npy...")
    embeddings = np.load(str(os.path.join(tokenizer_path, "embeddings.npy")))

    assert len(embeddings) == tokenizer.get_size(), (
        f"Tokenizer and embeddings not the same size {len(embeddings)}, {tokenizer.get_size()}"
    )

    metacat = MetaCAT(tokenizer=tokenizer, embeddings=embeddings)
    for category in category_names:
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


@app.command()
def make(config_filepath: Path, temp_dir: Path = Path("./.temp"), output: Path = Path.cwd()):
    """
    Build a model from a specification file.
    """
    log.info(f"Making temporary directory: {temp_dir}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    config = MakeConfig(**yaml.safe_load(config_filepath.open()))
    log.info(f"making: {config}")

    # Create MedCAT config object
    medcat_config = Config()
    if config.medcat_model_config:
        medcat_config = Config.from_dict(config.medcat_model_config)

    # Load vocab
    vocab = None
    if config.vocab:
        if config.vocab.location:
            log.info(f"Loading vocab: {config.vocab.location}")
            vocab = Vocab.load(str(config.vocab.location.get_or_download(temp_dir)))
        elif config.vocab.data:
            raise Exception("vocab generation not yet implemented")  #  TODO

    # Load CDB
    cdb = None
    if config.cdb:
        if config.cdb.location:
            log.info(f"Loading cdb: {config.cdb.location}")
            cdb = CDB.load(str(config.cdb.location.get_or_download(temp_dir)))
        elif config.cdb.data:
            log.info(f"Building CDB from {config.cdb.data}")

            cdb_temp_dir = temp_dir / Path("cdb")

            cdb_builder = CDBBuilder(
                temp_dir=cdb_temp_dir, custom_data_paths=[config.cdb.data.get_or_download(temp_dir)]
            )
            cdb_builder.preprocess()
            cdb = cdb_builder.create_cdb()
            del cdb_builder

    # Load model
    if config.model:
        if config.model.location:
            log.info(f"Loading model pack from {config.model.location}")
            model = MiADE_CAT.load_model_pack(str(config.model.location.get_or_download(temp_dir)))
            if vocab:
                log.info("Replacing vocab")
                model.vocab = vocab
            if cdb:
                log.info("Replacing cdb")
                model.cdb = cdb
    else:
        if vocab and cdb:
            log.info("Creating new model pack")
            model = MiADE_CAT(cdb=cdb, config=cdb.config, vocab=vocab)
        else:
            log.error("Both a CDB and vocab are required to make a new model.")
            raise Exception("Both a CDB and vocab are required to make a new model.")

    # Train context embeddings
    if config.context_embedding_training_data:
        context_data = pl.read_csv(config.context_embedding_training_data.get_or_download(temp_dir))["text"].to_list()
        log.info(f"Training context embeddings from: {config.context_embedding_training_data}")
        log.info(f"Data sample: {context_data[:10]}")
        model.config.general["checkpoint"]["output_dir"] = str(temp_dir / Path("cdb"))
        model.train(context_data)

    # Supervised model training
    if config.supervised_training:
        supervised_config_dict = config.supervised_training.__dict__
        supervised_config_dict["print_stats"] = True
        supervised_config_dict["data_path"] = config.supervised_training.annotation_data.get_or_download(temp_dir)
        del supervised_config_dict["annotation_data"]
        if config.supervised_training.synthetic_data:
            supervised_config_dict["synthetic_data_path"] = config.supervised_training.synthetic_data.get_or_download(
                temp_dir
            )
        del supervised_config_dict["synthetic_data"]

        log.info(f"Starting supervised training with {supervised_config_dict['data_path']}")
        fp, fn, tp, p, r, f1, cui_counts, examples = model.train_supervised(**supervised_config_dict)

        model_id = model.config.version["id"]
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

    if config.meta_models:
        log.debug("Building meta models")
        if config.meta_models.tokenizer.location:
            raise Exception("tokenizer loading not yet implemented")  # TODO load tokeniser
        else:
            tokenizer, embeddings = _create_bbpe_tokenizer(config.meta_models.tokenizer.data.get_or_download(temp_dir))
            wrapped_tokenizer = TokenizerWrapperBPE(tokenizer)

        meta_models = []
        meta_model_categories = []
        stats = {}
        for meta_model_name, meta_spec in config.meta_models.models.items():
            if meta_spec.config.general.category_name is None:
                meta_spec.config.general.category_name = meta_model_name

            meta_model_categories.append(meta_spec.config.general.category_name)

            meta_model = MiADE_MetaCAT(tokenizer=wrapped_tokenizer, embeddings=embeddings, config=meta_spec.config)

            annotations_path = meta_spec.annotations.get_or_download(temp_dir)

            if meta_spec.synthetic_data:
                synthetic_csv_path = meta_spec.synthetic_data.get_or_download(temp_dir)
                synthetic_data = pl.read_csv(synthetic_csv_path)[["text", meta_model_name]]
                prepped_synthetic_data_path = temp_dir / Path(f"prepped_synthetic_data_{meta_model_name}")
                synthetic_data.write_csv(prepped_synthetic_data_path)
                synthetic_label_counts = _get_label_counts_for_synthetic_data(synthetic_data, meta_model_name)
            else:
                prepped_synthetic_data_path = None
                synthetic_label_counts = {}

            label_counts = _add_zip(
                _get_label_counts_for_annotation_export(json.load(open(annotations_path)), meta_model_name),
                synthetic_label_counts,
            )

            labels = list(label_counts.keys())
            total_label_count = sum(label_counts.values())

            if meta_spec.balance_weights:
                meta_model.config.train["class_weights"] = []
                i = 0
                for label in labels:
                    meta_model.config.train["class_weights"].append(1 - (label_counts[label] / total_label_count))
                    meta_model.config.general["category_value2id"][label] = i
                    i += 1

            log.info(
                f"Starting MetaCAT training for {meta_spec.config.general['category_name']} for {meta_spec.config.train.nepochs} epoch(s) "
                f"with annotation file {str(annotations_path)}"
            )

            meta_model_save_path = temp_dir / Path(meta_model_name)
            report = meta_model.train(
                json_path=str(annotations_path),
                synthetic_csv_path=str(prepped_synthetic_data_path) if prepped_synthetic_data_path else None,
                save_dir_path=str(meta_model_save_path),
            )
            log.debug(report)
            log.info(meta_model.config.general["category_value2id"])
            stats[meta_spec.config.general.category_name] = report
            meta_models.append(MetaCAT.load(str(meta_model_save_path)))

        model = CAT(cdb=model.cdb, vocab=model.vocab, config=model.config, meta_cats=meta_models)
        for category in meta_model_categories:
            model.config.version["performance"]["meta"] = stats.get(category)

    model.config.version["location"] = str(output)
    if config.description:
        model.config.version["description"] = config.description
    if config.ontology:
        model.config.version["ontology"] = config.ontology

    tag = config.tag + "_" if config.tag else ""
    current_date = datetime.datetime.now().strftime("%Y_%m_%d")
    name = f"miade_{tag}modelpack_{current_date}"

    model.create_model_pack(str(output), name)
    log.info(f"Saved model pack at {output}/{name}_{model.config.version['id']}")

    rmtree(temp_dir)


if __name__ == "__main__":
    app()
