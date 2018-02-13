from collections import Counter
from datetime import timedelta, datetime, MINYEAR

import pandas as pd
import random
import json

from backfit.BackfitUtils import init_objects

base = "../../../isaac_data_files/"
df = pd.read_csv(base+"hwgen1.csv", index_col=0, header=0)

base = "../../../isaac_data_files/"
#choose a teacher for whose class to create homework
teacher_df = pd.read_csv(base+"groups.csv")
groupmem_df = pd.read_csv(base+"group_memberships.csv")

idx = df.index
# concepts = set()
# for qid in idx:
#     rawval = df.loc[qid, "related_concepts"]
#     if not pd.isna(rawval):
#         rcs = eval(rawval)
#         concepts.update(rcs)

# for c in sorted(concepts):
#     print(c)
# print(len(concepts))

n_users = 1000
cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users,
                                                                               path="../../../isaac_data_files/",
                                                                               seed=666)

run_ct = 0
max_ts = datetime(1,1,MINYEAR)
print("Generating files for {} users...".format(len(users)))


while True:
    r = random.randint(0, teacher_df.shape[0])
    group = teacher_df.iloc[r, :]
    t_id = group["owner_id"]
    g_id = group["id"]
    members = groupmem_df[groupmem_df["group_id"]==g_id]
    users = members["user_id"]
    print(g_id, t_id, list(users))

    for u in users:
        try:
            runs = open("../../../isaac_data_files/by_runs/{}.txt".format(u)).readlines()
        except FileNotFoundError:
            continue
        for run_ix, run in enumerate(runs):
            ts, q, n_atts, n_pass = eval(run)
            if pd.to_datetime(ts) > max_ts:
                max_ts = pd.to_datetime(ts)

    ts_cutoff = max_ts - timedelta(days=7)

    all_concepts = Counter()
    all_cats = Counter()
    for u in users:
        try:
            runs = open("../../../isaac_data_files/by_runs/{}.txt".format(u)).readlines()
        except FileNotFoundError:
            continue
        for run_ix, run in enumerate(runs):
            run_ct += 1
            ts, q, n_atts, n_pass = eval(run)
            if(pd.to_datetime(ts) < ts_cutoff or q not in df.index):
                continue

            cat = cat_lookup[q.replace("|","~")]
            concepts_raw = df.loc[q, "related_concepts"]
            concepts = eval(df.loc[q, "related_concepts"]) if not pd.isna(concepts_raw) else []
            #print(ts, u, q, concepts)
            all_cats[cat]+=1
            all_concepts.update(concepts)

    print(ts_cutoff)
    print(all_cats.most_common(10))
    print(all_concepts.most_common(10))
    print(len(all_concepts))

    input("prompt")
