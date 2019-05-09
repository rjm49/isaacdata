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
from hwgen.deep.ttb_utils import filter_assignments, build_SXUA, train_deep_model, create_assignment_summary, \
    augment_data
from hwgen.profiler import get_attempts_from_db

print("started")

use_saved = False
do_train = False
do_testing = True
create_scorecards = False

base = "../../../isaac_data_files/"

n_users = -1
print("initing objects")
cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(
    n_users)

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
    # tracemalloc.start()
    print("Initialising deep learning HWGen....")
    os.nice(3)

    print("loading SXUA")
    f = open(base + "SXUA.comp.pkl", 'rb')
    SXUA = pickle.load(f)
    f.close()

    tr = pandas.DataFrame.from_csv(base+"tr_ttb.csv")
    tt = pandas.DataFrame.from_csv(base+"tt_ttb.csv")

    ass_summ = pandas.concat([tr,tt])
    pid_override = set()
    qids_in_play = set()
    ss = set()
    for aix in ass_summ.index:
        ss.update(eval(ass_summ.loc[aix,"students"]))
        # hxz = [h for h in eval(ass_summ.loc[aix,"hexes"]) if (h.startswith("ch_") or h.startswith("ch-i"))]
        # ass_summ.loc[aix, "hexes"] = str(hxz)

    for s in ss:
        attz = get_attempts_from_db(s)
        qids_in_play.update([q for q in list(attz["question_id"]) if (q.startswith("ch_") or q.startswith("ch-i"))] )
        pid_override.update([s.split("|")[0] for s in list(attz["question_id"]) if (s.startswith("ch_") or s.startswith("ch-i"))])
    all_qids = sorted(qids_in_play)
    pid_override = sorted(pid_override)
    all_page_ids = pid_override

    print(all_qids)
    print("Qids in play len", len(all_qids))

    print(pid_override)
    print("Pids in play len", len(pid_override))

    numpy.set_printoptions(precision=4)

    model = load_model(base + "hwg_model.hd5")
    aug = None
    if do_testing:
        print("testing")
        aug = augment_data(tt, SXUA, all_page_ids=all_page_ids, pid_override=pid_override, filter=True)
        evaluate3(aug, model, pid_override)

        input("now class")
        class_ev_lookup = class_evaluation(aug, model, pid_override)
        # evaluate_phybook_loss(tt, SXUA, model, sc, load_saved_data=use_saved)  # , sscaler,levscaler,volscaler)
        # input("DEEP testing done")
        print("m testing")
        evaluate_by_bucket(tt, aug, model, pid_override)

    if create_scorecards:
        if not aug:
            aug = augment_data(tt, SXUA, all_page_ids=all_page_ids, pid_override=pid_override, filter=True)
        create_student_scorecards(aug, model, all_qids=all_qids, all_page_ids=pid_override, pid_override=pid_override, cat_page_lookup=cat_page_lookup)
