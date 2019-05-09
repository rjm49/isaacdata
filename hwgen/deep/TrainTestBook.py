import gc
import os
import pickle
from collections import Counter

import numpy
import pandas
from keras.models import load_model
from sklearn.externals import joblib

from hwgen.common import init_objects, get_meta_data, get_all_assignments, get_student_list, make_gb_question_map, \
    get_all_attempts
from hwgen.deep.ttb_evaluate import evaluate3, class_evaluation, evaluate_by_bucket
from hwgen.deep.ttb_scorecards import create_student_scorecards
from hwgen.deep.ttb_utils import build_SXUA, train_deep_model, create_assignment_summary, \
    augment_data, build_start_dates
from hwgen.profiler import get_attempts_from_db

print("started")

use_saved = True
do_train = True
do_testing = True
create_scorecards = True

base = "../../../isaac_data_files/"

# n_users = -1
# print("initing objects")
# cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(
#     n_users)



print("initied")

hwdf = get_meta_data()
concepts_all = set()
hwdf.index = hwdf["question_id"]
hwdf.loc[:,"related_concepts"] = hwdf["related_concepts"].map(str)
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

# pid_override = list(topic_map.keys())


if __name__ == "__main__":

    df = get_all_attempts()
    print(df[0:10])
    exit()

    # tracemalloc.start()
    print("Initialising deep learning HWGen....")
    os.nice(3)

    model = None
    sc = None
    xmask = None

    USE_CACHED_ASSGTS = True
    SAVE_CACHED_ASSGTS = True
    cache_fname = base + "cached_assgts.csv"
    if USE_CACHED_ASSGTS:
        print("using cached assets")
        ass_summ = pandas.read_csv(base + "ass_summ.csv")
        ass_summ.loc[:, "creation_date"] = pandas.to_datetime(ass_summ["creation_date"]).dt.floor("D")
    else:
        print("not using cached assets ... creating...")
        ass = get_all_assignments()
        #ass = filter_assignments(ass, book_only=False)
        ass_summ = create_assignment_summary(ass)
        if SAVE_CACHED_ASSGTS:
            ass.to_csv(cache_fname)
            ass_summ.to_csv(base + "ass_summ.csv")

    # assignments.loc[:, "creation_date"] = pandas.to_datetime(assignments["creation_date"]).dt.floor("D")
    # pid_override = set()
    # for h in ass_summ["hexes"]: # go through all hexes ever assigned...
    #     pid_override.update(eval(h))
    # pid_override = sorted(list(pid_override))

    tr = ass_summ[0:100]
    tt = ass_summ[100:110]

    pid_override = set()
    qids_in_play = set()
    ss = set()
    for aix in ass_summ.index:
        ss.update(eval(ass_summ.loc[aix,"students"]))
        # hxz = [h for h in eval(ass_summ.loc[aix,"hexes"]) if (h.startswith("ch_") or h.startswith("ch-i"))]
        # ass_summ.loc[aix, "hexes"] = str(hxz)

    for s in ss:
        attz = get_attempts_from_db(s)
        qids_in_play.update([q for q in list(attz["question_id"])])# if (q.startswith("ch_") or q.startswith("ch-i"))] )
        pid_override.update([s.split("|")[0].replace("-","_") for s in list(attz["question_id"]) if (s.startswith("ch_") or s.startswith("ch-i"))])
    all_qids = sorted(qids_in_play)
    pid_override = sorted(pid_override)
    all_page_ids = pid_override

    print(all_qids)
    print("Qids in play len", len(all_qids))

    print(pid_override)
    print("Pids in play len", len(pid_override))

    BUILD_SXUA = True
    if BUILD_SXUA:
        start_dates = build_start_dates(ass_summ)
        ass_summ = ass_summ[ass_summ["has_book_hexes"] == True]  # discard activity that has nowt to do with the book
        for psi in list(start_dates.keys())[0:10]:
            print(psi, start_dates[psi])
        print("building SXUA")
        SXUA = build_SXUA(base, ass_summ, all_qids=all_qids, all_page_ids=all_page_ids, pid_override=all_page_ids, start_dates=start_dates)
        f = open(base + "SXUA.comp.pkl", 'wb')
        pickle.dump(SXUA, f)
        f.close()
        print("compressed and SAVED")
    else:
        print("loading SXUA")
        f = open(base + "SXUA.comp.pkl", 'rb')
        SXUA = pickle.load(f)
        f.close()

    # ass_summ = ass_summ[0:8000]
    ass_summ = ass_summ[ass_summ["has_book_hexes"]==True] # discard activity that has nowt to do with the book
    ass_summ = ass_summ[ass_summ["num_hexes"]==1] # discard activity that has nowt to do with the book
    print("ass summ pre filtered, shape {}".format(ass_summ.shape))
    ass_summ = ass_summ[ass_summ["include"]==True]
    print("ass summ post filtered, shape {}".format(ass_summ.shape))


    COUNT_TEACHERS=False
    if COUNT_TEACHERS:
        t_test = None
        t_train = None
        ct = Counter()
        for t in numpy.unique(ass_summ["owner_user_id"]):
            t_assignments = ass_summ[ass_summ["owner_user_id"] == t]
            ct[t] = ass_summ.shape[0]
        print(ct.most_common(20))
        print("teachers counted")

        assct = 0
        teacherct = 0
        for (t, tct) in list(ct.most_common(len(ct))):
            t_assignments = ass_summ[ass_summ["owner_user_id"] == t]
            this_t_assts = t_assignments.shape[0]
            if this_t_assts < 5:
                print("not enough assignments",t,tct,this_t_assts)
                del ct[t]
            else:
                teacherct += 1
        print(teacherct, assct)

        teacherN = teacherct
        test_insts = 0
        target_test = 6000 #teacherN // 10

        t_test = []
        t_train = []
        put_in_test_set = False
        n_train = 0
        n_test = 0
        for (t, tct) in ct.most_common(teacherN):
            num_students = 0
            t_assignments = ass_summ[ass_summ["owner_user_id"] == t]
            print(t_assignments.shape[0], "new training assts")

            if put_in_test_set and n_test<target_test:
                # test_insts += 1
                aug = augment_data(t_assignments, SXUA, pid_override=pid_override, filter=True)
                num_students = len(aug[0])
                n_test += num_students
                if len(t_test) == 0:
                    t_test = t_assignments
                else:
                    t_test = pandas.concat([t_test, t_assignments])
                put_in_test_set = False
            else:
                aug = augment_data(t_assignments, SXUA, pid_override=pid_override, filter=False)
                num_students = len(aug[0])
                n_train += num_students
                if len(t_train) == 0:
                    t_train = t_assignments
                else:
                    t_train = pandas.concat([t_train, t_assignments])
                put_in_test_set = True

            if n_train >= 50000:
                break

        tr = t_train
        tt = t_test
        print("numbers of (student,asst) pairs:", n_train, n_test)
        tr.to_csv(base + "tr_ttb.csv")
        tt.to_csv(base + "tt_ttb.csv")
    else:
        tr = pandas.read_csv(base + "tr_ttb.csv")
        tt = pandas.read_csv(base + "tt_ttb.csv")

    ass_summ = pandas.concat([tr,tt])

    print("len t_train, t_test",len(tr), len(tt))

    # pid_override = all_page_ids
    # print(pid_override)
    # print("{} pids available".format(len(pid_override)))


    # Now filter and split
    # assignments = assignments[assignments["include"] == True]

    # assignments["creation_date"] = pandas.to_datetime(assignments["creation_date"])
    # assignments["creation_date"] = assignments["creation_date"].dt.floor("D")
    # ass_summ = ass_summ[0:1000]


    print("Split complete!")
    print("{} {}".format(len(tt), len(tr)))

    n_macroepochs =100
    n_epochs = 100

    if do_train:
        print("training with tr shape={}".format(tr.shape))
        aug = augment_data(tr, SXUA, pid_override=pid_override, filter=False)
        model = train_deep_model(aug)
        print("...deleted original X,y")
        model.save(base + 'hwg_model.hd5')
        # joblib.dump(xmask, base + 'hwg_xmask.pkl')
        # joblib.dump(sc, base + 'hwg_mlb.pkl')
        # joblib.dump((sscaler,levscaler,volscaler), base + 'hwg_scaler.pkl')
        print("saved model and exit")

    numpy.set_printoptions(precision=4)

    model = load_model(base + "hwg_model.hd5")
    aug = None
    if do_testing or create_scorecards:
        aug = augment_data(tt, SXUA, pid_override=pid_override, filter=True)
        if do_testing:
            print("testing")
            evaluate3(aug, model, pid_override)
            input("now class")
            class_ev_lookup = class_evaluation(aug, model, pid_override)
            # evaluate_phybook_loss(tt, SXUA, model, sc, load_saved_data=use_saved)  # , sscaler,levscaler,volscaler)
            # input("DEEP testing done")
            print("m testing")
            evaluate_by_bucket(aug, model, pid_override, class_ev_lookup)
        if create_scorecards:
            create_student_scorecards(aug, model, all_qids=all_qids, all_page_ids=pid_override)
