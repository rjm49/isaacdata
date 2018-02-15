from collections import Counter, defaultdict
from datetime import timedelta, datetime, MINYEAR

import pandas as pd
import random
import json

from matplotlib.gridspec import GridSpec

from backfit.BackfitUtils import init_objects
from irt.irt_engine import IRTEngine
from matplotlib import pyplot as plt

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
print("Generating files for {} users...".format(len(users)))


while True:
    max_ts = datetime(1, 1, MINYEAR)
    r = random.randint(0, teacher_df.shape[0])
    group = teacher_df.iloc[r, :]
    t_id = group["owner_id"]
    g_id = group["id"]
    members = groupmem_df[groupmem_df["group_id"]==g_id]
    users = members["user_id"]

    if(len(users) < 10):
        continue

    for u in users:
        try:
            runs = open("../../../isaac_data_files/by_runs/{}.txt".format(u)).readlines()
        except FileNotFoundError:
            continue
        for run_ix, run in enumerate(runs):
            ts, q, n_atts, n_pass = eval(run)
            if pd.to_datetime(ts) > max_ts:
                max_ts = pd.to_datetime(ts)

    ts_cutoff = (max_ts - timedelta(days=7)) if (max_ts > datetime(1,1,MINYEAR)) else max_ts

    classs_concepts = set()
    recent_concepts = Counter()
    recent_cats = Counter()
    irts = {}
    subj_irts = {}
    concept_levels = defaultdict(list)
    cat_levels = defaultdict(list)
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

            classs_concepts.update(concepts)
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

    if(recent_concepts):
        con_index, con_data = zip(*recent_concepts.most_common(10))
        recent_concepts_df = pd.DataFrame(columns=["count"], index=con_index)
        recent_concepts_df["count"]=con_data

    if(recent_cats):
        cat_index, cat_data = zip(*recent_cats.most_common(10))
        recent_cats_df = pd.DataFrame(columns=["count"], index=cat_index)
        recent_cats_df["count"]=cat_data

    avg_concept_levels_df = pd.DataFrame(0, columns=["level"], index=classs_concepts)
    avg_concept_counts_df = pd.DataFrame(0, columns=["level"], index=classs_concepts)

    for u in users:
        print("- - - - - - - -")
        print("PUPIL ",u)
        print("Subject level abilities:")
        for s in subj_irts[u]:
            theta = subj_irts[u][s].curr_theta
            print(s,"=",theta)
            cat_levels[s].append(theta)
        if irts[u]:
            print("\nConcept level abilities:")
        for c in irts[u]:
            theta = irts[u][c].curr_theta
            print(c,"=",theta)
            concept_levels[c].append(theta)
            avg_concept_levels_df.loc[c,"level"]+=theta
            avg_concept_counts_df.loc[c,"level"]+=1
        print("- - - - - - - -")

    avg_concept_levels_df = avg_concept_levels_df / avg_concept_counts_df
    avg_concept_levels_df.fillna(value=0, inplace=True)

    plt.style.use('ggplot')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    gs = GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[0, 1])
    ax4 = plt.subplot(gs[1, 1])

    #plt.suptitle("CLASS "+str(g_id))
    plt.suptitle("CLASS {} : ({} Students) : {} to {}".format(g_id, len(users), ts_cutoff.date(), max_ts.date()))

    if(recent_concepts):
        ax1.axis("equal")
        ax1.set_title("Concepts studied this week")
        ax1.pie(recent_concepts_df, labels=recent_concepts_df.index)
        #recent_concepts_df.transpose().plot(kind="barh", width=1.0, stacked=True, ax=ax1 )
    else:
        ax1.axis("off")
    if(recent_cats):
        ax2.axis("equal")
        ax2.set_title("Topics studied this week")
        ax2.pie(recent_cats_df, labels=recent_cats_df.index)
        #recent_cats_df.transpose().plot(kind="barh", width=1.0, stacked=True, ax=ax2)
    else:
        ax2.axis("off")
    #ax3.bar(x=avg_concept_levels_df.index, height=avg_concept_levels_df["level"])
    print(list(concept_levels.values()))
    print(list(concept_levels.keys()))
    ax3.set_title("Student skill by concept")
    if(concept_levels):
        ax3.boxplot(list(concept_levels.values()), labels=list(concept_levels.keys()))
        plt.sca(ax3)
        plt.xticks(rotation='vertical')
    ax4.set_title("Student skill by topic")
    if(cat_levels):
        ax4.boxplot(list(cat_levels.values()), labels=list(cat_levels.keys()))
        plt.sca(ax4)
        plt.xticks(rotation='vertical')

    plt.show()
