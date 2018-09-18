import sys
# sys.path.append(".")
# sys.path.append("..")
import pandas

from hwgen.common import get_all_assignments, init_objects
from keras.models import load_model
from sklearn.externals import joblib
from hwgen.deep.preproc import build_SXUA, populate_student_cache, filter_assignments
from hwgen.hwgen_mentoring_utils import make_mentoring_model, train_deep_model, create_student_scorecards

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


# assignments = assignments[assignments["group_id"].isin(target_group_ids)]

LOAD_PREVIOUS_ASSIGNMENTS =False
ass_fname = "filtered_assts.pkl"
if LOAD_PREVIOUS_ASSIGNMENTS:
    assignments = joblib.load(base+ass_fname)
else:
    assignments = get_all_assignments()
    assignments.loc[:,"creation_date"] = pandas.to_datetime(assignments.loc[:,"creation_date"])
    joblib.dump(assignments, (base + ass_fname))

tr = assignments[~assignments["group_id"].isin(target_group_ids)]
# tr = tr.sample(n=10000, replace=False)
tt = assignments[assignments["group_id"].isin(target_group_ids)]

print("pre filter", assignments.shape)
tr = filter_assignments(tr, mode="all", max_n=1000)
print("tr filtered")
# tr = tr[tr["include"] == True]
print("post filter", tr.shape)

LOAD_PSI_CACHE = False
if LOAD_PSI_CACHE:
    populate_student_cache(assignments)
    exit()

print(tr.shape, tt.shape)

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
    model, _, sc = train_deep_model(tr, SXUA, all_qids, all_page_ids, all_page_ids, 10, 10, load_saved_tr=False, filter_by_length=True, model_generator=model_genr)
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
