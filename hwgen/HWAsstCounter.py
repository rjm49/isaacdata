from collections import Counter


import pandas
from backfit.BackfitUtils import init_objects
from hwgen.concept_extract import concept_extract, page_to_concept_map

base = "../../../isaac_data_files/"

#get assignments ....
asst_df = pandas.read_csv(base + "gb_assignments.csv")
sprofs = pandas.read_csv(base + "student_profiling/users_all.csv")

def make_role_map():
    map = {}
    roledf = pandas.read_csv(base + "role_changes.csv", index_col="id", header=0)
    event_details_list = roledf[["event_details","timestamp"]]
    for e,ts in event_details_list.itertuples(index=False):
        emap = eval(e)
        if emap['newRole']=="TEACHER":
            uid = emap['userId']
            if uid not in map:
                map[uid]=pandas.to_datetime(ts)
    return map


asst_ct = Counter()
ts_map = {}
user_ids = asst_df["user_id"] #hopefully isolates the owners of the assignments
tss = asst_df["timestamp"]

#for each assignment
for uid,ts in zip(user_ids, tss):
    ts = pandas.to_datetime(ts)
    print(uid)
    if not uid.isdigit():
        continue
    uid = int(uid)
    if not uid in sprofs.index:
        continue
    asst_ct[uid]+=1 # increment this teachers assignment count
    if uid not in ts_map or ts > ts_map[uid]:
        ts_map[uid] = ts

role_changes_map = make_role_map()

fout = open(base+"teachers_hw.csv","w")
fout.write("id,name,num_assignments,last_assignment,role_changed\n")
n = len(asst_ct)
for uid, cnt in asst_ct.most_common(n):
    trow = sprofs.loc[int(uid), ["given_name", "family_name"]]
    role_changed = role_changes_map[uid].date() if uid in role_changes_map else ""
    if str(trow["given_name"])!="nan":
        name = str(trow["given_name"]) + " " + str(trow["family_name"])
    else:
        name = trow["family_name"]
    fout.write("{},{},{},{},{}\n".format(uid, name, cnt, ts_map[uid].date(), role_changed))
fout.close()