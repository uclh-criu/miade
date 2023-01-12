from spacy.tokens import Span, Doc
from medcat.cat import CAT

from typing import List, Dict


class MiADE_CAT(CAT):
    """Experimental - overriding medcat write out function - more control over spacy pipeline: add negex results"""

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
                    out_ent["types"] = [
                        self.cdb.addl_info["type_id2name"].get(tui, "")
                        for tui in out_ent["type_ids"]
                    ]
                    out_ent["source_value"] = ent.text
                    out_ent["detected_name"] = str(ent._.detected_name)
                    out_ent["acc"] = float(ent._.context_similarity)
                    out_ent["context_similarity"] = float(ent._.context_similarity)
                    out_ent["start"] = ent.start_char
                    out_ent["end"] = ent.end_char
                    for addl in addl_info:
                        tmp = self.cdb.addl_info.get(addl, {}).get(cui, [])
                        out_ent[addl.split("2")[-1]] = (
                            list(tmp) if type(tmp) == set else tmp
                        )
                    out_ent["id"] = ent._.id
                    out_ent["meta_anns"] = {}

                    if doc_extended_info:
                        out_ent["start_tkn"] = ent.start
                        out_ent["end_tkn"] = ent.end

                    if context_left > 0 and context_right > 0:
                        out_ent["context_left"] = doc_tokens[
                            max(ent.start - context_left, 0) : ent.start
                        ]
                        out_ent["context_right"] = doc_tokens[
                            ent.end : min(ent.end + context_right, len(doc_tokens))
                        ]
                        out_ent["context_center"] = doc_tokens[ent.start : ent.end]

                    if hasattr(ent._, "meta_anns") and ent._.meta_anns:
                        out_ent["meta_anns"] = ent._.meta_anns

                    if hasattr(ent._, "negex"):
                        out_ent["negex"] = ent._.negex

                    out["entities"][out_ent["id"]] = dict(out_ent)
                else:
                    out["entities"][ent._.id] = cui

            if (
                cnf_annotation_output.get("include_text_in_output", False)
                or out_with_text
            ):
                out["text"] = doc.text
        return out
