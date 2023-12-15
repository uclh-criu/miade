import json
import logging
import pandas as pd
from copy import deepcopy

from typing import List, Tuple, Optional, Dict, Set, Union

from spacy import Defaults
from tqdm.autonotebook import trange
from spacy.tokens import Span, Doc

from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.config import Config
from medcat.meta_cat import MetaCAT
from medcat.ner.transformers_ner import TransformersNER
from medcat.vocab import Vocab
from medcat.utils.matutils import intersect_nonempty_set
from medcat.utils.data_utils import make_mc_train_test, get_false_positives
from medcat.utils.checkpoint import Checkpoint
from medcat.utils.helpers import tkns_from_doc
from medcat.utils.filters import get_project_filters

logger = logging.getLogger("cat")


class MiADE_CAT(CAT):
    """Experimental - overriding medcat write out function - more control over spacy pipeline: add negex results"""

    def __init__(self,
                 cdb: CDB,
                 vocab: Union[Vocab, None] = None,
                 config: Optional[Config] = None,
                 meta_cats: List[MetaCAT] = [],
                 addl_ner: Union[TransformersNER, List[TransformersNER]] = []) -> None:
        if config.preprocessing.stopwords is not None:
            Defaults.stop_words = config.preprocessing.stopwords
        super.__init__(cdb=cdb, vocab=vocab, config=config, meta_cats=meta_cats, addl_ner=addl_ner)

    def _doc_to_out(
        self,
        doc: Doc,
        only_cui: bool,
        addl_info: List[str],
        out_with_text: bool = False,
    ) -> Dict:
        out: Dict = {"entities": {}, "tokens": []}
        cnf_annotation_output = getattr(self.config, "annotation_output", {})
        if doc is not None:
            out_ent: Dict = {}
            if self.config.general.get("show_nested_entities", False):
                _ents = []
                for _ent in doc._.ents:
                    entity = Span(doc, _ent["start"], _ent["end"], label=_ent["label"])
                    entity._.cui = _ent["cui"]
                    entity._.detected_name = _ent["detected_name"]
                    entity._.context_similarity = _ent["context_similarity"]
                    entity._.id = _ent["id"]
                    if "meta_anns" in _ent:
                        entity._.meta_anns = _ent["meta_anns"]
                    _ents.append(entity)
            else:
                _ents = doc.ents

            if cnf_annotation_output.get("lowercase_context", True):
                doc_tokens = [tkn.text_with_ws.lower() for tkn in list(doc)]
            else:
                doc_tokens = [tkn.text_with_ws for tkn in list(doc)]

            if cnf_annotation_output.get("doc_extended_info", False):
                # Add tokens if extended info
                out["tokens"] = doc_tokens

            context_left = cnf_annotation_output.get("context_left", -1)
            context_right = cnf_annotation_output.get("context_right", -1)
            doc_extended_info = cnf_annotation_output.get("doc_extended_info", False)

            for _, ent in enumerate(_ents):
                cui = str(ent._.cui)
                if not only_cui:
                    out_ent["pretty_name"] = self.cdb.get_name(cui)
                    out_ent["cui"] = cui
                    out_ent["type_ids"] = list(self.cdb.cui2type_ids.get(cui, ""))
                    out_ent["types"] = [self.cdb.addl_info["type_id2name"].get(tui, "") for tui in out_ent["type_ids"]]
                    out_ent["source_value"] = ent.text
                    out_ent["detected_name"] = str(ent._.detected_name)
                    out_ent["acc"] = float(ent._.context_similarity)
                    out_ent["context_similarity"] = float(ent._.context_similarity)
                    out_ent["start"] = ent.start_char
                    out_ent["end"] = ent.end_char
                    for addl in addl_info:
                        tmp = self.cdb.addl_info.get(addl, {}).get(cui, [])
                        out_ent[addl.split("2")[-1]] = list(tmp) if type(tmp) == set else tmp
                    out_ent["id"] = ent._.id
                    out_ent["meta_anns"] = {}

                    if doc_extended_info:
                        out_ent["start_tkn"] = ent.start
                        out_ent["end_tkn"] = ent.end

                    if context_left > 0 and context_right > 0:
                        out_ent["context_left"] = doc_tokens[max(ent.start - context_left, 0) : ent.start]
                        out_ent["context_right"] = doc_tokens[ent.end : min(ent.end + context_right, len(doc_tokens))]
                        out_ent["context_center"] = doc_tokens[ent.start : ent.end]

                    if hasattr(ent._, "meta_anns") and ent._.meta_anns:
                        out_ent["meta_anns"] = ent._.meta_anns

                    if hasattr(ent._, "negex"):
                        out_ent["negex"] = ent._.negex

                    out["entities"][out_ent["id"]] = dict(out_ent)
                else:
                    out["entities"][ent._.id] = cui

            if cnf_annotation_output.get("include_text_in_output", False) or out_with_text:
                out["text"] = doc.text
        return out

    def train_supervised(
        self,
        data_path: str,
        synthetic_data_path: Optional[str] = None,
        reset_cui_count: bool = False,
        nepochs: int = 1,
        print_stats: int = 0,
        use_filters: bool = False,
        terminate_last: bool = False,
        use_overlaps: bool = False,
        use_cui_doc_limit: bool = False,
        test_size: int = 0,
        devalue_others: bool = False,
        use_groups: bool = False,
        never_terminate: bool = False,
        train_from_false_positives: bool = False,
        extra_cui_filter: Optional[Set] = None,
        checkpoint: Optional[Checkpoint] = None,
        is_resumed: bool = False,
    ) -> Tuple:
        checkpoint = self._init_ckpts(is_resumed, checkpoint)

        # Backup filters
        _filters = deepcopy(self.config.linking["filters"])
        filters = self.config.linking["filters"]

        fp = fn = tp = p = r = f1 = examples = {}
        with open(data_path) as f:
            data = json.load(f)
        cui_counts = {}

        if test_size == 0:
            logger.info("Running without a test set, or train==test")
            test_set = data
            train_set = data
        else:
            train_set, test_set, _, _ = make_mc_train_test(data, self.cdb, test_size=test_size)

        if print_stats > 0:
            fp, fn, tp, p, r, f1, cui_counts, examples = self._print_stats(
                test_set,
                use_project_filters=use_filters,
                use_cui_doc_limit=use_cui_doc_limit,
                use_overlaps=use_overlaps,
                use_groups=use_groups,
                extra_cui_filter=extra_cui_filter,
            )
        if reset_cui_count:
            # Get all CUIs
            cuis = []
            for project in train_set["projects"]:
                for doc in project["documents"]:
                    doc_annotations = self._get_doc_annotations(doc)
                    for ann in doc_annotations:
                        cuis.append(ann["cui"])
            for cui in set(cuis):
                if cui in self.cdb.cui2count_train:
                    self.cdb.cui2count_train[cui] = 10

        # Remove entities that were terminated
        if not never_terminate:
            for project in train_set["projects"]:
                for doc in project["documents"]:
                    doc_annotations = self._get_doc_annotations(doc)
                    for ann in doc_annotations:
                        if ann.get("killed", False):
                            self.unlink_concept_name(ann["cui"], ann["value"])

        latest_trained_step = checkpoint.count if checkpoint is not None else 0
        current_epoch, current_project, current_document = self._get_training_start(train_set, latest_trained_step)

        for epoch in trange(
            current_epoch,
            nepochs,
            initial=current_epoch,
            total=nepochs,
            desc="Epoch",
            leave=False,
        ):
            # Print acc before training
            for idx_project in trange(
                current_project,
                len(train_set["projects"]),
                initial=current_project,
                total=len(train_set["projects"]),
                desc="Project",
                leave=False,
            ):
                project = train_set["projects"][idx_project]

                # Set filters in case we are using the train_from_fp
                filters["cuis"] = set()
                if isinstance(extra_cui_filter, set):
                    filters["cuis"] = extra_cui_filter

                if use_filters:
                    project_filter = get_project_filters(
                        cuis=project.get("cuis", None),
                        type_ids=project.get("tuis", None),
                        cdb=self.cdb,
                        project=project,
                    )

                    if project_filter:
                        filters["cuis"] = intersect_nonempty_set(project_filter, filters["cuis"])

                for idx_doc in trange(
                    current_document,
                    len(project["documents"]),
                    initial=current_document,
                    total=len(project["documents"]),
                    desc="Document",
                    leave=False,
                ):
                    doc = project["documents"][idx_doc]
                    spacy_doc: Doc = self(doc["text"])

                    # Compatibility with old output where annotations are a list
                    doc_annotations = self._get_doc_annotations(doc)
                    for ann in doc_annotations:
                        if not ann.get("killed", False):
                            cui = ann["cui"]
                            start = ann["start"]
                            end = ann["end"]
                            spacy_entity = tkns_from_doc(spacy_doc=spacy_doc, start=start, end=end)
                            deleted = ann.get("deleted", False)
                            self.add_and_train_concept(
                                cui=cui,
                                name=ann["value"],
                                spacy_doc=spacy_doc,
                                spacy_entity=spacy_entity,
                                negative=deleted,
                                devalue_others=devalue_others,
                            )
                    if train_from_false_positives:
                        fps: List[Span] = get_false_positives(doc, spacy_doc)

                        for fp in fps:
                            fp_: Span = fp
                            self.add_and_train_concept(
                                cui=fp_._.cui,
                                name=fp_.text,
                                spacy_doc=spacy_doc,
                                spacy_entity=fp_,
                                negative=True,
                                do_add_concept=False,
                            )

                    latest_trained_step += 1
                    if (
                        checkpoint is not None
                        and checkpoint.steps is not None
                        and latest_trained_step % checkpoint.steps == 0
                    ):
                        checkpoint.save(self.cdb, latest_trained_step)

            if synthetic_data_path is not None:
                synth_data = pd.read_csv(synthetic_data_path)
                logger.info(
                    f"Training with additional {len(synth_data)} synthetic data points from {synthetic_data_path}"
                )
                for i in range(len(synth_data)):
                    spacy_doc: Doc = self(synth_data.text.values[i])
                    cui = synth_data.cui.values[i]
                    name = synth_data.name.values[i]
                    start = synth_data.start.values[i]
                    end = synth_data.end.values[i]
                    spacy_entity = tkns_from_doc(spacy_doc=spacy_doc, start=start, end=end)
                    self.add_and_train_concept(
                        cui=cui,
                        name=name,
                        spacy_doc=spacy_doc,
                        spacy_entity=spacy_entity,
                        negative=False,
                        devalue_others=devalue_others,
                    )

            if terminate_last and not never_terminate:
                # Remove entities that were terminated, but after all training is done
                for project in train_set["projects"]:
                    for doc in project["documents"]:
                        doc_annotations = self._get_doc_annotations(doc)
                        for ann in doc_annotations:
                            if ann.get("killed", False):
                                self.unlink_concept_name(ann["cui"], ann["value"])

            if print_stats > 0 and (epoch + 1) % print_stats == 0:
                fp, fn, tp, p, r, f1, cui_counts, examples = self._print_stats(
                    test_set,
                    epoch=epoch + 1,
                    use_project_filters=use_filters,
                    use_cui_doc_limit=use_cui_doc_limit,
                    use_overlaps=use_overlaps,
                    use_groups=use_groups,
                    extra_cui_filter=extra_cui_filter,
                )

        # Set the filters again
        self.config.linking["filters"] = _filters

        return fp, fn, tp, p, r, f1, cui_counts, examples
