import os
import pickle

import numpy
import pandas
from numpy import save
from sklearn.externals import joblib
from hwgen.common import init_objects, get_meta_data, get_user_data, get_student_list, make_db_call
from hwgen.concept_extract import concept_extract, page_to_concept_map
from hwgen.profiler import profile_student, get_attempts_from_db, profile_students

base = "../../../isaac_data_files/"

#need to build a softmax classifier to recommend questions...
#this would have qn output nodes

n_users = -1
cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs, _, _ = init_objects(n_users)

# def make_gb_question_map():
#     gbd_df = pandas.read_csv(base + "gameboards.txt", sep="~")
#     map = {}
#     for gb_id, item in zip(gbd_df["id"], gbd_df["questions"]):
#         if str is not type(item):
#             continue
#         # print(gb_id)
#         # print(item)
#         item = item[1:-1]
#         item = item.split(",")
#         map[gb_id] = item
#     return map

def make_gb_question_map():
    query = "select id, questions from gameboards"
    raw_df = make_db_call(query)
    map = {}
    for r in raw_df.iterrows():
        gb_id = r[1]["id"]
        qs = r[1]["questions"] # TODO must we eval()?
        map[gb_id] = qs
    return map

# grp_df = pandas.read_csv(base + "group_memberships.csv")
# def get_students_in_group(gr_id):
#     return list(grp_df[grp_df["group_id"] == gr_id]["user_id"])


hwdf = get_meta_data()
concepts_all = set()
hwdf.index = hwdf["question_id"]
hwdf["related_concepts"] = hwdf["related_concepts"].map(str)
for concepts_raw in hwdf["related_concepts"]:
    print(concepts_raw)
    concepts = eval(concepts_raw)
    if concepts is not None:
        concepts_all.update(concepts)
concepts_all = list(concepts_all)



asst_fname = base+"assignments.pkl"


def make_data(ass_n):
    user_cache = {}
    asses = []
    ass_df = pandas.read_csv(base + "gb_assignments.csv")
    # ass_df = ass_df.iloc[27000:, :]
    # sprofs = pandas.read_csv(base + "student_profiling/users_all.csv")
    # sprofs["date_of_birth"] = pandas.to_datetime(sprofs["date_of_birth"])
    gb_qmap = make_gb_question_map()
    ass_ct =0

    ass_df["timestamp"] = pandas.to_datetime(ass_df["timestamp"])
    ass_df = ass_df[ass_df.event_details!="{}"]
    ass_df["event_details"] = ass_df["event_details"].str.replace("0L,", "0,")

    profile_df = get_user_data("*")
    profile_df["date_of_birth"] = pandas.to_datetime(profile_df["date_of_birth"])

    # group_list = set()
    # for ass in ass_df.iterrows():
    #     if 0 < ass_n < ass_ct:
    #         break
    #     event_details = eval(ass[1]['event_details'])
    #     gr_id = event_details["groupId"]
    #     group_list.add(gr_id)

    # all_students = get_student_list(group_list)

    for ass in ass_df.iterrows():
        if 0 < ass_n < ass_ct:
            break
        print("assct {} of {} ({} users cached)".format(ass_ct, ass_n, len(user_cache)))
        ts = ass[1]['timestamp']
        # print(ts)
        event_details = eval(ass[1]['event_details'])
        gb_id = event_details["gameboardId"]
        if gb_id not in gb_qmap:
            print("gb id unknown")
            continue
        this_qns = gb_qmap[gb_id]

        ass_ct += 1

        gr_id = event_details["groupId"]
        students = get_student_list(gr_id)

        if students.empty:
            continue

        students = list(students["user_id"])
        profile_df = get_user_data(list(students))
        # print("get group attempts")
        # attempts_df = get_attempts_from_db(students)
        # print("got group attempts")
        attempts_df = None

        profiles = profile_students(students, profile_df, ts, concepts_all, hwdf, user_cache, attempts_df)

        qarray = [0]*len(all_qids)
        for qid in this_qns:
            if qid in all_qids:
                ix = all_qids.index(qid)
                qarray[ix] = 1

        ass_entry = (ts, gb_id, gr_id, this_qns, profiles) #, qarray)
        asses.append(ass_entry)
        print("...{} students".format(len(profiles)))
    print("dumping")
    # joblib.dump(asses, asst_fname)
    with open(asst_fname, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(asses, f, pickle.HIGHEST_PROTOCOL)
    print("dumped")
    return asses

os.nice(3)

ass_n = 1000 # the number of SUCCESSFUL (i.e. complete) assignments to process # incomplete assts are skipped and do not count
data_gen = True

asses=None
model=None
if __name__=="__main__":
    if data_gen:
        asses = make_data(ass_n)