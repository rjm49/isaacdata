from collections import Counter
from datetime import timedelta, datetime, MINYEAR

import pandas as pd
import random
import json

from backfit.BackfitUtils import init_objects
from irt.irt_engine import IRTEngine

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

    recent_concepts = Counter()
    recent_cats = Counter()
    irts = {}
    subj_irts = {}
    for u in users:
        irts[u]={}
        subj_irts[u]={}
        try:
            runs = open("../../../isaac_data_files/by_runs/{}.txt".format(u)).readlines()
        except FileNotFoundError:
            continue
        for run_ix, run in enumerate(runs):
            run_ct += 1
            ts, q, n_atts, n_pass = eval(run)

            qt = q.replace("|","~")
            cat = cat_lookup[qt]
            lev = levels[qt]+1
            concepts_raw = df.loc[q, "related_concepts"]
            concepts = eval(df.loc[q, "related_concepts"]) if not pd.isna(concepts_raw) else []
            #print(ts, u, q, concepts)

            if(pd.to_datetime(ts) >= ts_cutoff or q not in df.index):
                recent_cats[cat]+=1
                recent_concepts.update(concepts)

            for c in concepts:
                if c not in irts[u]:
                    # print("new irt engine for", u,c)
                    irts[u][c] = IRTEngine()
                else:
                    irt = irts[u][c]
                    irt.curr_theta = irt.update(lev, (n_pass>0))
                    irts[u][c] = irt
                    # print(u, c,"history =", irt.history)
                    # print("irt update", u, c, irt.curr_theta)

            if cat not in subj_irts[u]:
                subj_irts[u][cat] = IRTEngine()
            else:
                subj_irts[u][cat].curr_theta = subj_irts[u][cat].update(lev, (n_pass>0))
                #print("irt cat update", u, cat, subj_irts[u][cat].curr_theta)


    print("GROUP", g_id, "SUBJECT MIXTURE SINCE", ts_cutoff)
    print(recent_cats.most_common(10))
    print(recent_concepts.most_common(10))

    for u in users:
        print("- - - - - - - -")
        print("PUPIL ",u)
        print("Subject level abilities:")
        for s in subj_irts[u]:
            print(s,"=",subj_irts[u][s].curr_theta)
        if irts[u]:
            print("\nConcept level abilities:")
        for c in irts[u]:
            print(c,"=",irts[u][c].curr_theta)
        print("- - - - - - - -")

    input("prompt")
