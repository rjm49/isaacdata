import pandas as pd
import json

from backfit.BackfitUtils import init_objects

base = "../../../isaac_data_files/"
df = pd.read_csv(base+"hwgen1.csv", index_col=0, header=0)

idx = df.index
concepts = set()
for qid in idx:
    rawval = df.loc[qid, "related_concepts"]
    if not pd.isna(rawval):
        rcs = eval(rawval)
        concepts.update(rcs)

for c in sorted(concepts):
    print(c)
print(len(concepts))

n_
cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users,
                                                                               path="../../../isaac_data_files/",
                                                                               seed=666)

runs = open(base+"by_runs/{}.txt".format(u)).readlines()
        u_run_ct = len(runs)
        all_zero_level = True
        for run_ix,run in enumerate(runs):
            run_ct+=1
            ts, q, n_atts, n_pass = eval(run)
            if u_start_ts is None:
                u_start_ts = pd.to_datetime(ts)
            u_end_ts = pd.to_datetime(ts)
            #fout.write(",".join(map(str,(run))) + "\n")
            # fout.write(str(run)+"\n")
            # continue
            qt = q.replace("|","~")
            lev = levels[qt]
            pass
            if(lev!=0):
                all_zero_level=False
            else:
                pass
            print(all_types)
            # atix = all_types.index( str( atypes.loc[qt,7] ) )
            # print(atix)