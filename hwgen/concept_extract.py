from collections import Counter, defaultdict
from datetime import timedelta, datetime, MINYEAR

import pandas as pd

base = "../../../isaac_data_files/"
df = pd.read_csv(base+"hwgen1.csv", index_col=0, header=0)

concepts_all = set()
dconcepts_all = set()

idx = df.index
for q in df.index:
    concepts_raw = df.loc[q, "related_concepts"]
    concepts = eval(df.loc[q, "related_concepts"]) if not pd.isna(concepts_raw) else []
    dconcepts_raw = df.loc[q, "detailed_concept_sections"]
    dconcepts = eval(df.loc[q, "detailed_concept_sections"]) if not pd.isna(dconcepts_raw) else []
    concepts_all.update(concepts)
    dconcepts_all.update(dconcepts)
print(list(concepts_all))
for ix,c in enumerate(concepts_all):
    print(ix, c)

for ix,c in enumerate(dconcepts_all):
    print(ix, c)