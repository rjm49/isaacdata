from collections import Counter, defaultdict
from datetime import timedelta, datetime, MINYEAR

import pandas as pd

def concept_extract(q=None):
    base = "../../../isaac_data_files/"
    df = pd.read_csv(base+"hwgen1.csv", index_col=0, header=0)

    concepts_all = set()
    dconcepts_all = set()

    idx = df.index if q is None else q #here select whether to get concepts just for one qn or for all qns

    for q in idx:
        concepts_raw = df.loc[q, "related_concepts"]
        concepts = eval(df.loc[q, "related_concepts"]) if not pd.isna(concepts_raw) else []
        dconcepts_raw = df.loc[q, "detailed_concept_sections"]
        dconcepts = eval(df.loc[q, "detailed_concept_sections"]) if not pd.isna(dconcepts_raw) else []
        concepts_all.update(concepts)
        dconcepts_all.update(dconcepts)
    # print(list(concepts_all))
    # for ix,c in enumerate(concepts_all):
    #     print(ix, c)
    #
    # for ix,c in enumerate(dconcepts_all):
    #     print(ix, c)

    return concepts_all

def page_to_concept_map():
    base = "../../../isaac_data_files/"
    df = pd.read_csv(base + "hwgen1.csv", index_col=0, header=0)

    page_to_concept_map = {}

    idx = df.index  # here select whether to get concepts just for one qn or for all qns

    for q in idx:
        concepts_raw = df.loc[q, "related_concepts"]
        concepts = eval(df.loc[q, "related_concepts"]) if not pd.isna(concepts_raw) else []
        dconcepts_raw = df.loc[q, "detailed_concept_sections"]
        dconcepts = eval(df.loc[q, "detailed_concept_sections"]) if not pd.isna(dconcepts_raw) else []
        qpage = q.split("|")[0]
        if qpage not in page_to_concept_map:
            page_to_concept_map[qpage] = concepts

    return page_to_concept_map