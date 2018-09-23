import numpy
import sys
# sys.path.append(".")
# sys.path.append("..")
from collections import Counter

import pandas

from hwgen.common import get_all_assignments, init_objects
from keras.models import load_model
from sklearn.externals import joblib

from hwgen.deep.preproc import build_SXUA, populate_student_cache, filter_assignments
from hwgen.hwgen_mentoring_utils import make_mentoring_model, train_deep_model, create_student_scorecards, evaluate3

base = "/home/rjm49/isaac_data_files/"
n_users = -1
cats, cat_lookup, all_qids, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(
    n_users)

print(len(all_page_ids))
for ix in range(20):
    print(all_page_ids[ix])

# for pid in all_page_ids:
#     print(pid)

target_group_ids = [7680, 7681, 7682]


# assignments = assignments[assignments["group_id"].isin(target_group_ids)]
pid_override = [pid for pid in all_page_ids if (pid.startswith("ch_") or pid.startswith("ch-i"))]

assignments = get_all_assignments()
assignments["creation_date"] = pandas.to_datetime(assignments["creation_date"])
# assignments["creation_date"] = assignments["creation_date"].dt.date
tx_list = list(numpy.unique(assignments["owner_user_id"]))
tr_tx_list = []
tt_tx_list = []
teacher_ct = Counter()
for t in tx_list:
    t_assignments = assignments[assignments["owner_user_id"] == t]
    teacher_ct[t] = t_assignments.shape[0]
print(teacher_ct.most_common(20))
print("teachers counted")
tt_len = 9
print("target tt len is",tt_len)
tx_ct = 0
for tx,c in teacher_ct.most_common():
    if c >= 10:
        if tx_ct % 2 == 1 and len(tt_tx_list) < tt_len:
            tt_tx_list.append(tx)
        else:
            tr_tx_list.append(tx)
        tx_ct+=1
print("{} teachers for training, {} for test".format(len(tr_tx_list), len(tt_tx_list)))

LOAD_DISC_ASSIGNMENTS =False
tt_raw_fname = "tt_raw.csv"
tr_raw_fname = "tr_raw.csv"
# ass_fname = "filtered_assts.pkl"
if LOAD_DISC_ASSIGNMENTS:
    # assignments = joblib.load(base+ass_fname)
    tt = joblib.load(base+tt_raw_fname)
    tr = joblib.load(base+tr_raw_fname)
else:
    # joblib.dump(assignments, (base + ass_fname))

    # tr = assignments[~assignments["group_id"].isin(target_group_ids)]
    # tr = tr.sample(n=10000, replace=False)
    # tt = assignments[assignments["group_id"].isin(target_group_ids)]

    tt = assignments[assignments["owner_user_id"].isin(tt_tx_list)]
    tr = assignments[assignments["owner_user_id"].isin(tr_tx_list)]
    print("pre filter:")
    print("tr {}".format(tr.shape))
    print("tt {}".format(tt.shape))

    tr = filter_assignments(tr, mode="book_only", max_n=10000, top_teachers_first=True, shuffle_rows=True)
    tt = filter_assignments(tt, mode="book_only", max_n=500, top_teachers_first=True, shuffle_rows=False)

    joblib.dump(tt,base+tt_raw_fname)
    joblib.dump(tr,base+tr_raw_fname)
    print("post filter:")
print("tr {}".format(tr.shape))
print("tt {}".format(tt.shape))

LOAD_PSI_CACHE = False
if LOAD_PSI_CACHE:
    populate_student_cache(assignments)
    exit()

BUILD_SXUA = False
sxua_file = base + "SXUA.comp.pkl"
if BUILD_SXUA:
    SXUA = build_SXUA(assignments, base, all_qids, all_page_ids)
    joblib.dump(SXUA, sxua_file)
else:
    SXUA = joblib.load(sxua_file)

do_train=True
fs = None
if do_train:
    print("training")
    model_genr = make_mentoring_model
    model, _, sc = train_deep_model(tr, SXUA, all_qids, all_page_ids, pid_override, 10, 10, load_saved_tr=False, filter_by_length=True, model_generator=model_genr)
    print("...deleted original X,y")
    model.save(base + 'hwg_model.hd5')
    # joblib.dump(fs, base + 'hwg_fs.pkl')
    joblib.dump(sc, base + 'hwg_mlb.pkl')
    # joblib.dump((sscaler,levscaler,volscaler), base + 'hwg_scaler.pkl')
else:
    model = load_model(base + "hwg_model.hd5")
    # fs = joblib.load(base + 'hwg_fs.pkl')
    sc = joblib.load(base + 'hwg_mlb.pkl')

do_test = True
if do_test:
    print("tt shape", tt.shape)
    evaluate3(tt,SXUA, model, sc,None, load_saved_data=False, pid_map=all_page_ids, sugg_map=pid_override)
    exit() #TODO remove for production
    create_student_scorecards(tt, SXUA, model, sc, fs=None, qid_map=all_qids, pid_map=all_page_ids, sugg_map=pid_override)
