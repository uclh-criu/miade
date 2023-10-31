import pandas as pd

from medcat.meta_cat import MetaCAT
from medcat.config_meta_cat import ConfigMetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBPE

from typing import Optional


def load_documents(data):
    documents = {}
    for i in range(0,len(data['projects'][0]['documents'])):
        documents[data['projects'][0]['documents'][i]['id']] = data['projects'][0]['documents'][i]['text']
    return documents


def load_annotations(data):
    annotations = []
    for i in range(0,len(data['projects'][0]['documents'])):
        document_id = data['projects'][0]['documents'][i]['id']
        annotations.extend([Annotation.from_dict(ann, document_id) for ann in data['projects'][0]['documents'][i]['annotations']])
    return annotations


def get_valid_annotations(data):
    annotations = []
    for ann in data:
        if not ann.deleted and not ann.killed and not ann.irrelevant:
            annotations.append(ann)
    return annotations


def get_probs_meta_classes_data(documents, annotations, ):
    r_labels = []
    p_labels = []
    l_labels = []
    cuis = []
    names = []
    texts = []
    tokens = []
    for ann in annotations:
        r_labels.append(ann.meta_relevance)
        p_labels.append(ann.meta_presence)
        l_labels.append(ann.meta_laterality)

        cuis.append(ann.cui)
        names.append(ann.value.lower())

        document = documents[ann.document_id].lower()
        _start = max(0, ann.start - 70)
        _end = min(len(document), ann.end + 1 + 70)
        texts.append(document[_start:_end])

        # doc_text = tokenizer(document)
        # ind = 0
        # for ind, pair in enumerate(doc_text['offset_mapping']):
        #     if ann.start >= pair[0] and ann.start < pair[1]:
        #         break
        # t_start = max(0, ind - 15)
        # t_end = min(len(doc_text['input_ids']), ind + 1 + 10)
        # tkns = doc_text['tokens'][t_start:t_end]
        # tokens.append(tkns)

    df = pd.DataFrame({"text": texts,
                       "cui": cuis,
                       "name": names,
                       # "tokens": tokens,
                       "relevance": r_labels,
                       "presence": p_labels,
                       "laterality (generic)": l_labels, })
    return df


def get_meds_meta_classes_data(documents, annotations, ):
    substance_labels = []
    allergy_labels = []
    severity_labels = []
    reaction_labels = []
    cuis = []
    names = []
    texts = []
    tokens = []
    for ann in annotations:
        substance_labels.append(ann.meta_substance_cat)
        allergy_labels.append(ann.meta_allergy_type)
        severity_labels.append(ann.meta_severity)
        reaction_labels.append(ann.meta_reaction_pos)

        cuis.append(ann.cui)
        names.append(ann.value.lower())

        document = documents[ann.document_id].lower()
        _start = max(0, ann.start - 70)
        _end = min(len(document), ann.end + 1 + 70)
        texts.append(document[_start:_end])

        # doc_text = tokenizer(document)
        # ind = 0
        # for ind, pair in enumerate(doc_text['offset_mapping']):
        #     if ann.start >= pair[0] and ann.start < pair[1]:
        #         break
        # t_start = max(0, ind - 15)
        # t_end = min(len(doc_text['input_ids']), ind + 1 + 10)
        # tkns = doc_text['tokens'][t_start:t_end]
        # tokens.append(tkns)

    df = pd.DataFrame({"text": texts,
                       "cui": cuis,
                       "name": names,
                       # "tokens": tokens,
                       "substance_category": substance_labels,
                       "allergy_type": allergy_labels,
                       "severity": severity_labels,
                       "reaction_pos": reaction_labels})
    return df




class Annotation:
    def __init__(
            self,
            alternative,
            id,
            document_id,
            cui,
            value,
            deleted,
            start,
            end,
            irrelevant,
            killed,
            manually_created,
            meta_laterality,
            meta_presence,
            meta_relevance,
            meta_allergy_type,
            meta_substance_cat,
            meta_severity,
            meta_reaction_pos,
            dictionary
    ):
        self.alternative = alternative
        self.id = id
        self.value = value
        self.document_id = document_id
        self.cui = cui
        self.deleted = deleted
        self.start = start
        self.end = end
        self.irrelevant = irrelevant
        self.killed = killed
        self.manually_created = manually_created
        self.meta_laterality = meta_laterality
        self.meta_presence = meta_presence
        self.meta_relevance = meta_relevance
        self.meta_allergy_type = meta_allergy_type
        self.meta_substance_cat = meta_substance_cat
        self.meta_severity = meta_severity
        self.meta_reaction_pos = meta_reaction_pos
        self.dict: Optional[dict] = dictionary

    @classmethod
    def from_dict(cls, d, document_id):
        meta_laterality = None
        meta_presence = None
        meta_relevance = None

        meta_allergy_type = None
        meta_substance_cat = None
        meta_severity = None
        meta_reaction_pos = None

        meta_anns = d.get("meta_anns")
        if meta_anns is not None:
            meta_ann_l = meta_anns.get('laterality (generic)')
            if meta_ann_l is not None:
                meta_laterality = meta_ann_l['value']
            meta_ann_r = meta_anns.get('relevance')
            if meta_ann_r is not None:
                meta_relevance = meta_ann_r['value']
            meta_ann_p = meta_anns.get('presence')
            if meta_ann_p is not None:
                meta_presence = meta_ann_p['value']

            meta_ann_allergy = meta_anns.get('allergy_type')
            if meta_ann_allergy is not None:
                meta_allergy_type = meta_ann_allergy['value']
            meta_ann_substance = meta_anns.get('substance_category')
            if meta_ann_substance is not None:
                meta_substance_cat = meta_ann_substance['value']
            meta_ann_severity = meta_anns.get('severity')
            if meta_ann_severity is not None:
                meta_severity = meta_ann_severity['value']
            meta_ann_reaction = meta_anns.get('reaction_pos')
            if meta_ann_reaction is not None:
                meta_reaction_pos = meta_ann_reaction['value']
        return cls(
            alternative=d['alternative'],
            id=d['id'],
            document_id=document_id,
            cui=d['cui'],
            value=d['value'],
            deleted=d['deleted'],
            start=d['start'],
            end=d['end'],
            irrelevant=d['irrelevant'],
            killed=d['killed'],
            manually_created=d['manually_created'],
            meta_laterality=meta_laterality,
            meta_presence=meta_presence,
            meta_relevance=meta_relevance,
            meta_allergy_type=meta_allergy_type,
            meta_substance_cat=meta_substance_cat,
            meta_severity=meta_severity,
            meta_reaction_pos=meta_reaction_pos,
            dictionary=d,
        )

    def __str__(self):
        return f"""
---
              id: {self.id}
     document_id: {self.document_id}
             cui: {self.cui}
           value: {self.value}
           start: {self.start}
             end: {self.end}

         deleted: {self.deleted}
      irrelevant: {self.irrelevant}
          killed: {self.killed}
manually created: {self.manually_created}

      laterality: {self.meta_laterality}
        presence: {self.meta_presence}
       relevance: {self.meta_relevance}

       substance category: {self.meta_substance_cat}
       allergy type: {self.meta_allergy_type}
       severity: {self.meta_severity}
       reaction pos: {self.reaction_pos}
---
        """

    def __eq__(self, other):
        return (
                self.alternative == other.alternative
                and
                self.cui == other.cui
                and
                self.document_id == other.document_id
                and
                self.deleted == other.deleted
                and
                self.start == other.start
                and
                self.end == other.end
                and
                self.irrelevant == other.irrelevant
                and
                self.killed == other.killed
                and
                self.manually_created == other.manually_created
                and
                self.meta_laterality == other.meta_laterality
                and
                self.meta_presence == other.meta_presence
                and
                self.meta_relevance == other.meta_relevance
                and
                self.meta_substance_cat == other.meta_substance_cat
                and
                self.meta_allergy_type == other.meta_allergy_type
                and
                self.meta_severity == other.meta_severity
                and
                self.meta_reaction_pos == other.meta_reaction_pos

        )

    def is_same_model_annotation(self, other):
        return (
                self.cui == other.cui
                and
                self.start == other.start
                and
                self.end == other.end
        )
