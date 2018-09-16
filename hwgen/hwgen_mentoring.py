import sys
# sys.path.append(".")
# sys.path.append("..")
import pandas

from hwgen.common import get_all_assignments, init_objects
from keras.models import load_model
from sklearn.externals import joblib
from hwgen.deep.TrainTestBook import create_student_scorecards, train_deep_model
from hwgen.deep.preproc import build_SXUA, populate_student_cache, filter_assignments

base = "/home/rjm49/isaac_data_files/"
n_users = -1
cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(
    n_users)

print(len(all_page_ids))
for ix in range(20):
    print(all_page_ids[ix])

# for pid in all_page_ids:
#     print(pid)

target_group_ids = [7680, 7681, 7682]
assignments = get_all_assignments()
assignments["creation_date"] = pandas.to_datetime(assignments["creation_date"])

LOAD_PSI_CACHE = False
if LOAD_PSI_CACHE:
    populate_student_cache(assignments)
    exit()

# assignments = assignments[assignments["group_id"].isin(target_group_ids)]

N = len(assignments)
LOAD_PREVIOUS_ASSIGNMENTS = True
ass_fname = "filtered_assts.pkl"
if LOAD_PREVIOUS_ASSIGNMENTS:
    assignments = joblib.load(base+ass_fname)
else:
    print("pre filter", assignments.shape)
    assignments = filter_assignments(assignments, mode="non_book_only")
    assignments = assignments[assignments["include"] == True]
    print("post filter", assignments.shape)
    joblib.dump(assignments, (base + ass_fname))

tr = assignments[~assignments["group_id"].isin(target_group_ids)]
tt = assignments[assignments["group_id"].isin(target_group_ids)]

print(tr.shape, tt.shape)

BUILD_SXUA = False
sxua_file = base + "SXUA.comp.pkl"
if BUILD_SXUA:
    SXUA = build_SXUA(assignments, base, all_qids, all_page_ids)
    joblib.dump(SXUA, sxua_file)
else:
    SXUA = joblib.load(sxua_file)

do_train=False
fs = None
if do_train:
    print("training")
    model, _, sc = train_deep_model(tr, SXUA, all_qids, all_page_ids, all_page_ids, 10, 10, load_saved_tr=False, filter_by_length=True)
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
    create_student_scorecards(tt, SXUA, model, sc, None, qid_map=all_qids, pid_map=all_page_ids, sugg_map=all_page_ids)
