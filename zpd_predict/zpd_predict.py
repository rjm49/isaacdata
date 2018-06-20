import numpy
import pandas

from hwgen.common import get_user_data, init_objects
from hwgen.hwgengen2 import hwgengen2, gen_semi_static
from hwgen.profiler import get_attempts_from_db
from utils.utils import extract_runs_w_timestamp
from zpd_predictor import ZPDPredictor

cats, cat_lookup, all_qids, _, _, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(-1)
reverse_qid_dict = {}
for ix,q in enumerate(all_qids):
    reverse_qid_dict[q]=ix

def gen_experience(psi, ts_list):
    raw_attempts = get_attempts_from_db(psi)
    X_list = []
    X = numpy.zeros(len(all_qids))
    for ts in sorted(ts_list):
        attempts = raw_attempts[(raw_attempts["timestamp"] < ts)]
        hits = attempts["question_id"]
        for qid in list(hits):
            try:
                qix = reverse_qid_dict[qid]
            except:
                print("UNK Qn ", qid)
                continue
            X[qix] = 1
        X_list.append(numpy.copy(X))
        raw_attempts = raw_attempts[(raw_attempts["timestamp"] >= ts)]
    return X_list

def gen_qhist(psi, ts):
    raw_attempts = get_attempts_from_db(psi)
    attempts = raw_attempts[(raw_attempts["timestamp"] < ts)]
    l1 = list(attempts["question_id"])
    l2 = list(attempts["timestamp"])
    qhist = list( zip(l1,l2) )
    return qhist

def gen_success(psi,ts_list):
    raw_attempts = get_attempts_from_db(psi)
    U_list = []
    U = numpy.zeros(len(all_qids))
    for ts in sorted(ts_list):
        attempts = raw_attempts[(raw_attempts["timestamp"] < ts)]
        hits = attempts[(attempts["correct"] == True)]
        hits = hits["question_id"]
        for qid in list(hits):
            try:
                qix = reverse_qid_dict[qid]
            except:
                print("UNK Qn ", qid)
                continue
            attct = len(attempts[attempts["question_id"]==qid])
            U[qix] = (1.0/attct)
        U_list.append(numpy.copy(U))
        raw_attempts = raw_attempts[(raw_attempts["timestamp"] >= ts)]
    return U_list

def encode_q_vectors(attempts):
    qids = list(attempts["question_id"])
    Qvec_list = []
    for qid in qids:
        Qvec = numpy.zeros(len(all_qids))
        if qid in all_qids:
            qix = all_qids.index(qid)
            Qvec[qix]=1
        Qvec_list.append(Qvec)
    return Qvec_list

def main():
    user_df = get_user_data("*")
    user_list = list(pandas.unique(user_df[(user_df["role"]=="STUDENT")]["id"]))
    zpdp = ZPDPredictor()
    S_list = []
    X_list = []
    U_list = []
    Qv_list = []
    pass_list = []
    atts_list = []

    i_users = 0
    n_users = 50
    for u in user_list[0:n_users]:
        i_users += 1
        print(i_users)
        attempts = get_attempts_from_db(u)
        ts_list = list(attempts["timestamp"])
        # S_list = gen_semi_static()
        tX = gen_experience(u, ts_list)
        X_list += tX
        U_list += gen_success(u, ts_list)
        Qv_list += encode_q_vectors(attempts)
        S_list += [numpy.zeros(1) for t in tX]
        pass_list += list(attempts["correct"]==True)
        atts_list += [numpy.ones(1) for t in tX]
        if (len(X_list) > 10000) or (i_users==n_users):
            S_list = numpy.array(S_list)
            X_list = numpy.array(X_list)
            U_list = numpy.array(U_list)
            Qv_list = numpy.array(Qv_list)
            pass_list = numpy.array(pass_list)
            atts_list = numpy.array(atts_list)
            zpdp.train((S_list,X_list,U_list),Qv_list,pass_list, atts_list)
            S_list = []
            X_list = []
            U_list = []
            Qv_list = []
            pass_list = []
            atts_list = []

    for u in user_list[n_users:2*n_users]:
        attempts = get_attempts_from_db(u)
        ts_list = list(attempts["timestamp"])
        delta_x = gen_experience(u, ts_list)
        X_list += delta_x
        U_list += gen_success(u, ts_list)
        Qv_list += encode_q_vectors(attempts)
        S_list += [numpy.zeros(1) for t in delta_x]
        pass_list += list(attempts["correct"] == True)
    S_list = numpy.array(S_list)
    X_list = numpy.array(X_list)
    U_list = numpy.array(U_list)
    Qv_list = numpy.array(Qv_list)
    pass_list = numpy.array(pass_list)
    metrics = zpdp.pass_model.evaluate([S_list,X_list,U_list,Qv_list], pass_list)
    print(metrics)

if __name__=="__main__":
    main()