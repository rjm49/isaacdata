import os
import pandas
from keras.models import load_model
from sklearn.externals import joblib

from hwgen.common import init_objects, get_meta_data, make_db_call, get_user_data, get_group_list, get_student_list
from hwgen.profiler import profile_students

base = "../../../isaac_data_files/"

n_users = -1
cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs, _, _, = init_objects(n_users)


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


#get assignments ....




os.nice(3)

ass_n = 10000 # the number of SUCCESSFUL (i.e. complete) assignments to process # incomplete assts are skipped and do not count
data_gen = True
do_train = True
do_testing = True



asses=None
model=None




if __name__=="__main__":
    teachers_df = pandas.DataFrame.from_csv("teachers.dat", header=0)
    teacher_ids = teachers_df.index

    model = load_model(base+"hwg_model.hd5")
    mlb = joblib.load(base+'hwg_mlb.pkl')
    scaler = joblib.load(base+'hwg_scaler.pkl')

    up_to_ts = pandas.datetime.now()
    fout = open("predictions.out","w")
    for t in teacher_ids:
        user_cache = {}
        class_list = get_group_list(t).iloc[:,0]
        print("groups:", class_list)
        for c in class_list:
            print("get student lsit for =>",c)
            students = get_student_list(c)
            students = list(students["user_id"])
            print("students:",students)
            if not students:
                continue
            # students = list(students)
            profile_df = get_user_data(students)
            print("profiles:",profile_df)
            profiles = profile_students(students, profile_df, up_to_ts, concepts_all, hwdf, user_cache, attempts_df=None)
            X = []
            for u in students:
                if u in profiles:
                    X.append(profiles[u])
                else:
                    print("why {} no here?".format(u))
            print(X)
            if not X:
                continue
            X = scaler.transform(X)
            predictions = model.predict(X)
            print(predictions)
            fout.write("TEACHER {} / CLASS {}\n".format(t,c))
            for s,p in zip(students,predictions):
                fout.write("{}\t{}\n".format(s,p))
    fout.close()