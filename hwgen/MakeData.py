import os
from collections import OrderedDict

import pandas
import pickle
from hwgen.common import init_objects, get_meta_data, get_user_data, get_student_list, make_db_call, get_all_assignments
from hwgen.concept_extract import concept_extract, page_to_concept_map
from hwgen.profiler import profile_student, get_attempts_from_db, profile_students
import zlib

base = "../../../isaac_data_files/"
FORCE_OVERWRITE = False

#need to build a softmax classifier to recommend questions...
#this would have qn output nodes

n_users = -1
cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(n_users)

# grp_df = pandas.read_csv(base + "group_memberships.csv")
# def get_students_in_group(gr_id):
#     return list(grp_df[grp_df["group_id"] == gr_id]["user_id"])


hwdf = get_meta_data()
concepts_all = set()
hwdf.index = hwdf["question_id"]
hwdf["related_concepts"] = hwdf["related_concepts"].map(str)
for concepts_raw in hwdf["related_concepts"]:
    print(concepts_raw)
    if(concepts_raw != "nan"):
        concepts = eval(concepts_raw)
        if concepts is not None:
            concepts_all.update(concepts)
concepts_all = list(concepts_all)



asst_fname = base+"assignments.pkl"


def make_data(ass_n, pickle_at, APPEND=True):
    user_cache = {}

    ass_df = get_all_assignments()
    # ass_df = ass_df.iloc[27000:, :]
    # sprofs = pandas.read_csv(base + "student_profiling/users_all.csv")
    # sprofs["date_of_birth"] = pandas.to_datetime(sprofs["date_of_birth"])
    gb_qmap = make_gb_question_map()
    ass_ct =0

    ass_df["creation_date"] = pandas.to_datetime(ass_df["creation_date"])
    #ass_df = ass_df[ass_df.event_details!="{}"]
    #ass_df["event_details"] = ass_df["event_details"].str.replace("0L,", "0,")

    profile_df = get_user_data("*")
    profile_df["date_of_birth"] = pandas.to_datetime(profile_df["date_of_birth"])

    ct=0

    if APPEND:
        print("APPEND mode")
        #recycle old pap
        f = open(asst_fname, 'rb')
        asses = pickle.load(f)
        f.close()
        tracking = open("tracking.dat","w+")
        print("loaded {} existing assignments".format(len(asses)))

    else:
        f = open(asst_fname, 'wb')
        f.truncate(0)
        f.close()
        tracking = open("tracking.dat","w")
        print("FRESH mode")
        #bake it fresh
        asses = OrderedDict()

    start_at = len(asses)
    number_to_do = ass_n - start_at
    if number_to_do <= 0:
        print("We already have {}>{} samples".format(start_at, ass_n))
        exit(1)

    #if ass_n is -1 then this overrides the trimming of the assts
    ass_df = ass_df.iloc[start_at:,:] if (ass_n>0) else ass_df

    for ass in ass_df.iterrows():
        id = ass[1]["id"]
        if id in asses and False == FORCE_OVERWRITE:
            # print("this assignment has already been processed, skipping!")
            continue

        print("assct {} of {} ({} users cached)".format(ass_ct, ass_n, len(user_cache)))
        ts = ass[1]['creation_date']
        # print(ts)
        # event_details = eval(ass[1]['event_details'])
        gb_id = ass[1]["gameboard_id"]
        if gb_id not in gb_qmap:
            print("gb id unknown")
            continue

        this_concepts = set()
        raw_qns = gb_qmap[gb_id]
        this_levels = []
        this_qns = raw_qns
        if type(raw_qns) is str:
            this_qns = eval(raw_qns) #TODO make sure this works hitting the database as well
        for q in this_qns:
            if "|" in q:
                q = q.split("|")[0]
            this_levels.append( lev_page_lookup[q] )
            cs = concept_extract(q)
            this_concepts.update(cs)


        gr_id = ass[1]["group_id"]
        students = get_student_list([gr_id])

        if students.empty:
            print(gr_id, "no students")
            continue
        else:
            print(gr_id, "students!")

        students = list(students["user_id"])
        profile_df = get_user_data(list(students))
        # print("get group attempts")
        # attempts_df = get_attempts_from_db(students)
        # print("got group attempts")

        profiles = profile_students(students, profile_df, ts, concepts_all, hwdf, user_cache, attempts_df=None)
        print(len(profiles),len(students))
        assert len(profiles)==len(students)
        assert len(profiles)>0
        # if len(profiles)==0:
        #     print("no profiles")
        #     continue
        print("compressing_profiles")
        c_profiles = zlib.compress(pickle.dumps(profiles))
        print("compressed")

        ass_entry = (ts, gb_id, gr_id, this_qns, this_concepts, this_levels, students, c_profiles)
        tracking.write(str(ass_entry[0:7]+(len(profiles),)))
        tracking.write("\n")
        # asses.append(ass_entry)
        asses[id] = ass_entry
        ass_ct += 1

        print("...{} students".format(len(profiles)))
        # ct+=1
        # afile.write(str(ass_entry)+"\n")
        # if ct > 100:
        #     afile.flush()
        #     ct=0
        print("ass_ct",ass_ct)
        print("pickle at", pickle_at)
        print("%", (ass_ct % pickle_at))
        if (ass_ct == number_to_do) or (ass_ct % pickle_at) == 0:
            f = open(asst_fname, 'wb')
            pickle.dump(asses, f)
            f.flush()
            print("***SAVED (hallelujah)")

        if ass_ct==number_to_do:
            print("we have hit maximum ass limit")
            break
    # print("taking massive dump")
    # # afile.write("]\n")
    # # afile.close()
    # # joblib.dump(asses, asst_fname)
    # # with gzip.open(asst_fname, 'w') as f:
    # #     #_pickle.dump(asses, f)
    # #     f.write(_pickle.dumps(asses))
    # with open(asst_fname, 'wb') as f:
    #     pickle.dump(asses, f)
    f.close()
    print("We now have {} assignments on disc".format(len(asses)))
    return
    tracking.close()

os.nice(3)

ass_n = 100000 # the number of SUCCESSFUL (i.e. complete) assignments to process # incomplete assts are skipped and do not count
pickle_at = 200
data_gen = True

asses=None
model=None
if __name__=="__main__":
    if data_gen:
        make_data(ass_n, pickle_at, APPEND=True)