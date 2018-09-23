import gc
import os
import pickle
from collections import Counter

import numpy
import pandas
from keras.models import load_model
from sklearn.externals import joblib

from hwgen.common import init_objects, get_meta_data, get_all_assignments, get_student_list, make_gb_question_map
from hwgen.deep.ttb_evaluate import evaluate3, class_evaluation, evaluate_by_bucket
from hwgen.deep.ttb_scorecards import create_student_scorecards
from hwgen.deep.ttb_utils import filter_assignments, build_SXUA, train_deep_model

use_saved = True
do_train = False
do_testing = True
create_scorecards = True

base = "../../../isaac_data_files/"

n_users = -1
cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(
    n_users)

hwdf = get_meta_data()
concepts_all = set()
hwdf.index = hwdf["question_id"]
hwdf["related_concepts"] = hwdf["related_concepts"].map(str)
for concepts_raw in hwdf["related_concepts"]:
    print(concepts_raw)
    if concepts_raw != "nan":
        concepts = eval(concepts_raw)
        if concepts is not None:
            concepts_all.update(concepts)
concepts_all = list(concepts_all)

asst_fname = base + "assignments.pkl"
# con_page_lookup = page_to_concept_map()


numpy.set_printoptions(threshold=numpy.nan)

concept_map = {}
topic_map = {}
concept_list = []
page_list = []
meta_df = pandas.DataFrame.from_csv(base + "book_question_meta.csv")
for thing in meta_df.iterrows():
    thing = thing[1]
    k = thing["URL:"].split("/")[-1]
    page_list.append(k)
    sft = "/".join((thing["Subject"], thing["Field"], thing["Topic"]))
    # concepts = thing["Related Concepts"].split(",")
    # concept_map[k] = concepts
    topic_map[k] = sft
    for c in concepts:
        if c not in concept_list:
            concept_list.append(c)

pid_override = list(topic_map.keys())


if __name__ == "__main__":
    # tracemalloc.start()
    print("Initialising deep learning HWGen....")
    os.nice(3)

    model = None
    sc = None
    fs = None

    USE_CACHED_ASSGTS = True
    SAVE_CACHED_ASSGTS = True
    cache_fname = base + "cached_assgts.csv"
    if USE_CACHED_ASSGTS:
        assignments = pandas.DataFrame.from_csv(cache_fname)
    else:
        assignments = get_all_assignments()
        assignments = filter_assignments(assignments, book_only=True)
        if SAVE_CACHED_ASSGTS:
            assignments.to_csv(cache_fname)
    # Now filter and split
    assignments = assignments[assignments["include"] == True]
    assignments["creation_date"] = pandas.to_datetime(assignments["creation_date"])
    assignments["creation_date"] = assignments["creation_date"].dt.floor("D")

    BUILD_SXUA = False
    if BUILD_SXUA:
        SXUA = build_SXUA(base, assignments, all_qids=all_qids, all_page_ids=all_page_ids, pid_override=pid_override)
        f = open(base + "SXUA.comp.pkl", 'wb')
        pickle.dump(SXUA, f)
        f.close()
        print("compressed and SAVED")

    else:
        print("loading SXUA")
        f = open(base + "SXUA.comp.pkl", 'rb')
        SXUA = pickle.load(f)
        f.close()
    print("loaded")

    gc.collect()
    print("gc'd")

    # assignments = assignments[assignments["owner_user_id"]==7062]
    # assignments = assignments[assignments["owner_user_id"]==6026]

    POST_FILTER=True
    if POST_FILTER:
        gb_qmap = make_gb_question_map()
        assignments["include"] = False
        for rix,row in enumerate(assignments.iterrows()):
            # print(row)
            aid = row[1]["id"]
            ts = row[1]["creation_date"]
            gr_id = row[1]["group_id"]
            gb_id = row[1]["gameboard_id"]
            student_ids = list(get_student_list(gr_id)["user_id"])
            # print(student_ids)
            # student_data = get_user_data(student_ids)
            hexes = list(gb_qmap[gb_id])
            # print(hexes)
            if len(student_ids)<=75:
                assignments.iloc[rix,6] = True

        print(assignments.shape[0])
        assignments = assignments[assignments["include"]==True]
        print(assignments.shape[0])
        print("post filtered")


    COUNT_TEACHERS=True
    if COUNT_TEACHERS:
        ct = Counter()
        for t in numpy.unique(assignments["owner_user_id"]):
            t_assignments = assignments[assignments["owner_user_id"] == t]
            ct[t] = t_assignments.shape[0]
        print(ct.most_common(20))
        print("teachers counted")


    t_train = None
    t_test = None

    asslimit = 30000
    assct = 0
    teacherct = 0
    for (t, tct) in list(ct.most_common(len(ct))):
        t_assignments = assignments[assignments["owner_user_id"] == t]
        this_ass = t_assignments.shape[0]
        if this_ass < 10:
            print("not enough assignments",t,tct,this_ass)
            del ct[t]
        else:
            teacherct += 1
            assct += this_ass
        if assct > asslimit:
            break
    print(teacherct, assct)

    teacherN = teacherct
    test_insts = 0
    target_test_insts = 10 #teacherN // 10

    for_test = False
    for (t, tct) in ct.most_common(teacherN):
        t_assignments = assignments[assignments["owner_user_id"] == t]
        print(t_assignments.shape[0], "new training assts")
        # sel_n = min(t_assignments.shape[0], assct // teacherN)
        # t_assignments = t_assignments.iloc[0:sel_n, :]

        # nass = t_assignments.shape[0]
        # if nass <2:
        #     this_split = 0
        # else:
        # this_split = 1
        # print("this split", this_split)
        # temp_t_tr = t_assignments.iloc[0:-(split//teacherN),:]
        # temp_t_tt = t_assignments.iloc[-(split//teacherN):, :]
        # end = t_assignments.shape[0] - this_split
        # temp_t_tr = t_assignments.iloc[0:end,:]
        # temp_t_tt = t_assignments.iloc[end:, :]

        if for_test and test_insts<target_test_insts:
            for_test = False
            test_insts += 1
            if t_test is None:
                t_test = t_assignments
            else:
                t_test = pandas.concat([t_test, t_assignments])
        else:
            for_test = True
            if t_train is None:
                t_train = t_assignments
            else:
                t_train = pandas.concat([t_train, t_assignments])

        # print("training dates:", temp_t_tr["creation_date"].min(), temp_t_tr["creation_date"].max())
        # print("testing dates:", temp_t_tt["creation_date"].min(), temp_t_tt["creation_date"].max())
        # if t_train is None:
        #     t_train = temp_t_tr
        #     t_test = temp_t_tt
        #     print("created t_train {} and t_test {}".format(len(t_train),len(t_test)))
        # else:
        #     t_train = pandas.concat([t_train, temp_t_tr])
        #     t_test = pandas.concat([t_test, temp_t_tt])
        #     print("extended t_train {} and t_test {}".format(len(t_train),len(t_test)))
        # if len(t_train) + len(t_test) >= totass:
        #     print("exceeded totass")
        #     break
    tr = t_train
    tt = t_test


    gc.collect()
    print("Split complete!")
    print("{} {}".format(len(tt), len(tr)))

    n_macroepochs =100
    n_epochs = 100

    if do_train:
        print("training")
        model, fs, sc = train_deep_model(tr, SXUA, n_macroepochs, n_epochs, load_saved_tr=use_saved, all_page_ids=all_page_ids, pid_override=pid_override)
        print("...deleted original X,y")
        model.save(base + 'hwg_model.hd5')
        joblib.dump(fs, base + 'hwg_fs.pkl')
        joblib.dump(sc, base + 'hwg_mlb.pkl')
        # joblib.dump((sscaler,levscaler,volscaler), base + 'hwg_scaler.pkl')

    if model is None:
        model = load_model(base + "hwg_model.hd5")
        fs = joblib.load(base + 'hwg_fs.pkl')
        sc = joblib.load(base + 'hwg_mlb.pkl')

    numpy.set_printoptions(precision=4)
    if do_testing:
        print("testing")
        evaluate3(tt, SXUA, model, sc,fs, load_saved_data=use_saved, pid_override=pid_override)
        # input("now class")
        class_ev_lookup = class_evaluation(tt, SXUA, model, sc, fs, load_saved_data=use_saved, pid_override=pid_override)
        # evaluate_phybook_loss(tt, SXUA, model, sc, load_saved_data=use_saved)  # , sscaler,levscaler,volscaler)
        # input("DEEP testing done")
        print("m testing")
        evaluate_by_bucket(tt, SXUA, model, sc,fs, load_saved_data=use_saved, group_data=class_ev_lookup, pid_override=pid_override)

    if create_scorecards:
        create_student_scorecards(tt, SXUA, model, sc,fs, load_saved_data=use_saved, pid_override=pid_override)
