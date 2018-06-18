import numpy
import pandas

from hwgen.common import get_user_data, init_objects
from hwgen.deep.TrainTestBook import load_tr_tt
from hwgen.hwgengen2 import hwgengen2, gen_semi_static, gen_success, gen_experience
from hwgen.profiler import get_attempts_from_db
from utils.utils import extract_runs_w_timestamp
from zpd_predict.ZPDPredictor import ZPDPredictor

cats, cat_lookup, all_qids, _, _, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(-1)

def encode_q_vectors(attempts):
    qids = list(attempts["question_id"])
    Qvec_list = []
    for qid in qids:
        Qvec = numpy.zeros(len(all_qids))
        qix = all_qids.index(qid)
        Qvec_list.append(Qvec)
    return Qvec_list

def main():
    user_df = get_user_data("*")
    user_list = list(pandas.unique(user_df["id"]))
    zpdp = ZPDPredictor()
    for u in user_list:
        attempts = get_attempts_from_db(u)
        ts_list = list(attempts["timestamp"])
        # S_list = gen_semi_static()
        S_list = [numpy.zeros(1) for t in ts_list]
        X_list = gen_experience(u, ts_list)
        U_list = gen_success(u, ts_list)
        pass_list = attempts["correct"]==True
        #hits = (attempts["correct"] == True)

        zpdp.train((S_list,X_list,U_list),Q_vector,pass_list, )

if __name__=="__main__":
    main()