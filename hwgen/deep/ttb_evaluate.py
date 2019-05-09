from math import sqrt
from random import choice

import numpy
from collections import Counter
from statistics import mean, median, stdev, variance

import pandas
from sklearn.externals import joblib

from hwgen.common import get_q_names, get_student_list, get_user_data, get_all_assignments, make_gb_question_map
from matplotlib import pyplot as plt

from hwgen.deep.ttb_utils import augment_data

gb_qmap = make_gb_question_map()




def evaluate3(aug, model, pid_override):
    # if load_saved_data:
    #     aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load(
    #         "tt.data")
    # else:
    aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, _, gr_id_list, ts_list, maxdops = aug
    # try:
    #     s_list = sc.transform(s_list)
    # except ValueError as ve:
    #     print(ve)
    #     print("Don't forget to check your flags... maybe you have do_train==False and have changed S ...")

    # arr=pandas.DataFrame(s_raw_list)
    # arr.to_csv("s_raw.csv")

    if len(x_list) == 0:
        print("wtf, x list is zero in eval3")
        exit(1)

    # print(x_list.shape)
    # x_list = x_list[:, xmask]
    ct = 0

    if maxdops:
        dops = [ sr[1] for sr in s_list if sr[1] <= maxdops ]
    else:
        dops = [ sr[1] for sr in s_list ]
    dopdelta = max(dops)
    print(dopdelta)

    y_trues = []
    delta_dict = {}
    exact_ct = Counter()
    strat_list = ["hwgen","step","lin","random"]

    model.summary()
    for sl,xl,ul,al,hxtt,hxtd in zip(s_list,x_list,u_list,a_list,hexes_to_try_list, hexes_tried_list):
        # print("--- {}".format((sl.shape, xl.shape, ul.shape, al.shape)))
        predictions = model.predict([sl.reshape(1, -1), xl.reshape(1, -1), ul.reshape(1, -1), al.reshape(1, -1)])
        # predictions = model.predict([sl, xl])#, ul, al])

        y_hats = list(reversed(numpy.argsort(predictions)[0]))
        for candix in y_hats:
            if pid_override[candix] not in hxtd:
                hwgen_y_hat = candix
                break

        options = [p for p in pid_override if p not in hxtd]
        hts = [h for h in hxtd if (h in pid_override)]

        p_hat = pid_override[hwgen_y_hat]
        random_p_hat = choice(options)
        random_y_hat = pid_override.index(random_p_hat)
        step_y_hat = 0 if not hts else min(len(pid_override) - 1, pid_override.index(hts[-1]) + 1)
        #lin_y_hat = int((len(pid_override) - 2) * (sr[1] / dopdelta))
        lin_y_hat = int((72) * (sl[1] / dopdelta))

        # y_trues = []
        # for p_true in sorted(hxtt):
        # p_true = sorted(hxtt)[(len(hxtt)-1) //2]
        #     y_trues.append(pid_override.index(p_true))
        # y_true = median(y_trues)
        p_true = sorted(hxtt)[0]
        y_true = pid_override.index(p_true)

        y_trues.append(y_true)

        for strat, y_hat in zip(strat_list, [hwgen_y_hat, step_y_hat, lin_y_hat, random_y_hat]):
            if strat not in delta_dict:
                delta_dict[strat] = []
            delta_dict[strat].append(((y_hat-y_true)))
            if(y_true == y_hat):
                exact_ct[strat]+=1

        ct += 1

    mean_y_true = numpy.mean(y_trues)
    var_y_true = numpy.var(y_trues)
    print("mean y is {}".format(mean_y_true))
    print("var(mean) is {}".format(var_y_true))

    for strat in strat_list:
        delta_list = delta_dict[strat]
        sq_list = [ d*d for d in delta_list ]
        mu = numpy.mean(delta_list)
        abs_mu = mean([abs(d) for d in delta_list])
        medi = numpy.median(delta_list)
        s2 = variance(delta_list)
        stdev = numpy.std(delta_list)
        print("{}: mean={} med={} std={}, s2={}".format(strat, mu, medi, stdev, s2))
        print("mae = {}".format(abs_mu))
        acc = exact_ct[strat]/ct
        print("Exact = {} of {} = {}".format(exact_ct[strat], ct, acc))
        print("MSE = {}".format(numpy.mean(sq_list)))
        rmse = sqrt(numpy.mean(sq_list))
        print("RMSE = {}".format(rmse))
        R2 = (var_y_true - s2) / var_y_true
        print("R2 = {}\n".format(R2))
    return (rmse, acc)
        
def evaluate_by_bucket(aug, model, pid_override, class_lookup):
    all_assts = get_all_assignments()
    aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list, max_dop = aug

    dops = [ s[1] for s in s_list ]
    min_dop = min(dops)
    max_dop = max(dops)
    # min_dop= min(s_list[:,1])
    # max_dop = max(s_list[:,1])

    collist = ["assignment", "group", "teacher", "student", "bucket", "actdop", "y_true", "random", "linear", "canonical", "hwgen"]
    apc_df = pandas.DataFrame(columns=collist)

    apc_df["assignment"] = aid_list
    apc_df["group"] = gr_id_list
    apc_df["student"] = psi_list
    apc_df["actdop"] = dops

    zippit = zip(aid_list,gr_id_list,psi_list,s_list, dops, x_list,u_list,a_list,hexes_to_try_list, hexes_tried_list, ts_list)
    for ix, (aid,grid,psi,sl,actdop,xl,ul,al,hxtt, hxtd, aidts) in enumerate(zippit):
        if ix % 50 == 0:
            print(aid)
    #     # xl = top_n_of_X(xl,fs)
        predictions = model.predict([sl.reshape(1,-1), xl.reshape(1,-1), ul.reshape(1,-1), al.reshape(1,-1)])
        # for j,p in enumerate(predictions):
        #     print(j,p)
        y_hwgen = numpy.argsort(predictions[0])[-1]
        apc_df.loc[ix,"hwgen"] = y_hwgen

        y_lin = (len(pid_override)-1)*actdop/(max_dop-min_dop)
        apc_df.loc[ix, "linear"] = y_lin

        y_true = [pid_override.index(hx) for hx in hxtt]
        apc_df.loc[ix, "y_true"] = y_true[0]

        choices = [pid for pid in pid_override if pid not in hxtd]
        ch = choice(choices)
        y_rand = pid_override.index(ch)

        apc_df.loc[ix, "random"] = y_rand

        last = max([pid_override.index(hx) for hx in hxtd]) if len(hxtd) else -1
        y_canon = min(last+1, len(pid_override)-1)
        apc_df.loc[ix, "canonical"] = y_canon

        (max_sum_ix, max_vote_ix) = class_lookup[aid]
        apc_df.loc[ix, "smax_av"] = max_sum_ix
        apc_df.loc[ix, "vote"] = max_vote_ix

    apc_df.to_csv("apc38_df.csv")


def class_evaluation(aug, model, pid_override):

    aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list, maxdops = aug


    # for row in tt.iterrows():
    lookup = {}
    ts_grid_lookup = {}
    for aid,s,s_raw,x,u,a,y,psi,grid,ts,hxtt in zip(aid_list, s_list, s_raw_list, x_list, u_list, a_list, y_list, psi_list, gr_id_list, ts_list, hexes_to_try_list):
        if aid not in lookup:
            lookup[aid] = ([],[],[],[],[],[],[],[])
            ts_grid_lookup[aid] = (ts,grid)
        sl,sr,xl,ul,al,yl,psil,hxl = lookup[aid]
        sl.append(s)
        sr.append(s_raw)
        xl.append(x)
        ul.append(u)
        al.append(a)
        yl.append(y)
        hxl.append(hxtt)
        psil.append(psi)
        lookup[aid] = (sl,sr,xl,ul,al,yl,psil,hxl)

    lkk = list(lookup.keys())

    result_lkup = {}

    sum_exact_match = 0
    vote_exact_match = 0
    sum_deltas = []
    vote_deltas = []
    sum_abs_deltas = []
    vote_abs_deltas = []
    for aid in lkk:
        m_list = []
        s_list = []
        x_list = []
        ts, gr_id = ts_grid_lookup[aid]
        sl, srl, xl, ul, al, yl, psil, hxl = lookup[aid]


        xl = numpy.array(xl)#[:,fs]
        sl = numpy.array(sl)
        ul = numpy.array(ul)
        al = numpy.array(al)

        print(sl)

        hxtt = hxl[0]

        # x_arr = top_n_of_X(x_arr,fs)

        print("shapez=", sl.shape, xl.shape, ul.shape, al.shape)

        y_preds = model.predict([sl, xl, ul, al])


        N = y_preds.shape[0]
        print(N)
        sum_preds = numpy.sum(y_preds, axis=0)
        print("sum of sums", numpy.sum(sum_preds))
        sum_preds = sum_preds / N
        max_sum_ix = sum_preds.argmax()
        max_sum_prob = sum_preds.max()

        vote_ct = Counter()
        for yp in y_preds:
            yp_max_ix : int = numpy.argmax(yp)
            label = pid_override[yp_max_ix]
            vote_ct[label]+=1

        max_vote_lab = vote_ct.most_common(1)[0][0]
        max_vote_ix = pid_override.index(max_vote_lab)
        max_sum_lab = pid_override[max_sum_ix]
        print("max sum lab =", max_sum_lab, max_sum_prob)
        print("votes counted:",vote_ct.most_common(5))
        print("most voted =", max_vote_lab)

        p_true = sorted(hxtt)[0]
        y_true = pid_override.index(p_true)
        sum_abs_delta = abs(y_true - max_sum_ix)
        vote_abs_delta = abs(y_true - max_vote_ix)
        sum_delta = (y_true - max_sum_ix)
        vote_delta = (y_true - max_vote_ix)
        if max_sum_lab == p_true:
            sum_exact_match += 1
        if max_vote_lab == p_true:
            vote_exact_match += 1
        print(p_true, max_vote_lab, max_sum_lab)
        sum_abs_deltas.append(sum_abs_delta)
        vote_abs_deltas.append(vote_abs_delta)
        sum_deltas.append(sum_delta)
        vote_deltas.append(vote_delta)

        result_lkup[aid] = (max_sum_ix, max_vote_ix)

    print("sum mean delta: {}".format(mean(sum_deltas)))
    print("vote mean delta: {}".format(mean(vote_deltas)))
    print("sum mean delta: {}".format(mean(sum_abs_deltas)))
    print("vote mean delta: {}".format(mean(vote_abs_deltas)))
    print("sum MSE: {}".format(mean([v**2 for v in sum_deltas])))
    print("vote MSE: {}".format(mean([v**2 for v in vote_deltas])))
    n_aids = len(lkk)
    print("sum exacts {}/{} = {}".format(sum_exact_match, n_aids, sum_exact_match/n_aids))
    print("vote exacts {}/{} = {}".format(vote_exact_match, n_aids, vote_exact_match/n_aids))
    input("wait>>>")
    return result_lkup
