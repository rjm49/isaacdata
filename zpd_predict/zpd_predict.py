import numpy
import pandas

from hwgen.common import get_user_data, init_objects
from hwgen.hwgengen2 import hwgengen2, gen_semi_static, gen_success, gen_experience
from hwgen.profiler import get_attempts_from_db
from utils.utils import extract_runs_w_timestamp
from zpd_predictor import ZPDPredictor

cats, cat_lookup, all_qids, _, _, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(-1)

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

    i=0
    for u in user_list[0:1000]:
        i += 1
        attempts = get_attempts_from_db(u)
        ts_list = list(attempts["timestamp"])
        # S_list = gen_semi_static()
        S_list += [numpy.zeros(1) for t in ts_list]
        X_list += gen_experience(u, ts_list)
        U_list += gen_success(u, ts_list)
        Qv_list += encode_q_vectors(attempts)
        pass_list += list(attempts["correct"]==True)
        atts_list += [numpy.ones(1) for t in ts_list]
        if (len(S_list) > 10000) or (i==1000):
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

if __name__=="__main__":
    main()