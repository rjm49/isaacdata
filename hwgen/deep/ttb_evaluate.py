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

# def evaluate2(tt, sxua, model, sc,fs, load_saved_data=False):
#     if load_saved_data:
#         aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load(
#             "tt.data")
#     else:
#         aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(
#             tt, sxua)
#     print(s_list.shape)
#     try:
#         s_list = sc.transform(s_list)
#     except ValueError as ve:
#         print(ve)
#         print("Don't forget to check your flags... maybe you have do_train==False and have changed S ...")
#
#
#     x_list = x_list[:, fs]
#     deltas = []
#     signed_deltas = []
#     exacts = 0
#     for sl,xl,ul,al,hxtt,hxtd in zip(s_list,x_list,u_list,a_list,hexes_to_try_list, hexes_tried_list):
#         predictions = model.predict([sl.reshape(1, -1), xl.reshape(1, -1), ul.reshape(1, -1), al.reshape(1, -1)])
#         y_hats = list(reversed(numpy.argsort(predictions)[0]))
#         for candix in y_hats:
#             if pid_override[candix] not in hxtd:
#                 y_hat = candix
#                 break
#
#         p_hat = pid_override[y_hat]
#         # y_trues = []
#         # for p_true in sorted(hxtt):
#         # p_true = sorted(hxtt)[(len(hxtt)-1) //2]
#         #     y_trues.append(pid_override.index(p_true))
#         # y_true = median(y_trues)
#         p_true = sorted(hxtt)[0]
#         y_true = pid_override.index(p_true)
#         if(y_true == y_hat):
#             exacts+=1
#         # p_true = pid_override[int(y_true)]
#         print(y_hat, y_true, p_hat, p_true)
#         deltas.append( abs((y_hat-y_true)))
#         signed_deltas.append( (y_hat - y_true))
#     mu = numpy.mean(deltas)
#     signed_mu = numpy.mean(signed_deltas)
#     sigma = numpy.std(deltas) if len(deltas)>1 else 0.0
#     print("Mean of {} difference from teachers, std={}".format(mu,sigma))
#     print("Signed delta is {}".format(signed_mu))
#     print("{} exact matches out of {} = {}".format(exacts, len(s_list), (exacts/len(s_list))))
#     # exit()

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

    #     teacher_id = tt.loc[tt[id==aid], "owner_user_id"]
    #
    #     for b in buckets:
    #         if abs((bucket_step*b) - actdop)<=bucket_width:
    #             bucket_counter[b] += 1
    #             ct += 1
    #             y_hats = []
    #             y_hats_raw = list(reversed(numpy.argsort(predictions)[0]))
    #             yix = 0
    #             while len(y_hats)<len(hxtt):
    #                 cand = pid_override[y_hats_raw[yix]]
    #                 yix+=1
    #                 if cand not in hxtd:
    #                     y_hats.append(pid_override.index(cand))
    #
    #             # p_hat = pid_override[y_hat]
    #
    #             #RANDOM SELECTION
    #             options = [p for p in pid_override if p not in hxtd]
    #
    #             #N+1
    #             hts = [h for h in hxtd if (h in pid_override)]
    #             # n1_p_hat = [h for h in sorted(hxtt) if (h in pid_override)][0]
    #             # n1_y_hat = pid_override.index(n1_p_hat)
    #
    #             ix = 0
    #             for p_true, y_hat in zip(sorted([hxtt[0]]), [y_hats[0]]):
    #                 random_p_hat = choice(options)
    #                 random_y_hat = pid_override.index(random_p_hat)
    #                 n1_y_hat = 0 if not hts else min(len(pid_override) - 1, pid_override.index(hts[-1]) + ix)
    #                 lin_y_hat = int((len(pid_override)-2)*(bucket_step*b / dopdelta))
    #                 ix+=1
    #
    #                 # y_trues=[]
    #                 # for p_true in sorted(hxtt):
    #                 #     y_trues.append(pid_override.index(p_true))
    #                 # y_true = mean(y_trues)
    #                 y_true = pid_override.index(sorted(hxtt)[0])
    #
    #                 # y_true = pid_override.index(p_true)
    #                 if (y_true == y_hat):
    #                     exacts += 1
    #                 if (y_true == n1_y_hat):
    #                     stepexacts += 1
    #                 if (y_true == lin_y_hat):
    #                     linexacts += 1
    #                 if (y_true == random_y_hat):
    #                     randexacts += 1
    #
    #                 #print(y_hat, y_true, p_hat, p_true)delta = abs(y_hat - y_true)
    #
    #
    #                 if PLOT_MODE=="ABS":
    #                     delta = abs(y_hat - y_true)
    #                     signed_delta = abs(y_hat - y_true)
    #                     rand_delta = abs(random_y_hat - y_true)
    #                     n1_delta = abs(n1_y_hat - y_true)
    #                     lin_delta = abs(lin_y_hat - y_true)
    #                 elif PLOT_MODE=="DELTA":
    #                     delta = (y_hat - y_true)
    #                     signed_delta = (y_hat - y_true)
    #                     rand_delta = (random_y_hat - y_true)
    #                     n1_delta = (n1_y_hat - y_true)
    #                     lin_delta = (lin_y_hat - y_true)
    #                 elif PLOT_MODE=="RAW":
    #                     delta = y_hat
    #                     signed_delta = y_hat
    #                     rand_delta = random_y_hat
    #                     n1_delta = n1_y_hat
    #                     lin_delta = lin_y_hat
    #                 else:
    #                     raise ValueError("{} is an invalid plotting mode!".format(PLOT_MODE))
    #
    #                 replist = [aid,grid,teacher_id, psi, bucket_step*b, actdop, y_true, random_y_hat, lin_y_hat, n1_y_hat, y_hat]
    #                 # if group_data is not None:
    #                 #     (gr_sum_ix, gr_vote_ix) = group_data[aid]
    #                 #     if b not in glookup:
    #                 #         glookup[b] = [ (gr_sum_ix - y_true, gr_vote_ix - y_true) ]
    #                 #     else:
    #                 #         deltas = glookup[b]
    #                 #         deltas.append( (gr_sum_ix - y_true, gr_vote_ix - y_true) )
    #                 #     replist += [gr_sum_ix, gr_vote_ix]
    #
    #                 if not b in mlookup:
    #                     mlookup[b] = [ (delta, signed_delta, rand_delta, n1_delta, lin_delta) ]
    #
    #                 else:
    #                     deltas = mlookup[b]
    #                     deltas.append( (delta, signed_delta, rand_delta, n1_delta, lin_delta) )
    #                     mlookup[b] = deltas
    #
    #                 apc_df.loc[report_ix] = replist
    #                 report_ix += 1
    #
    # for sm in [exacts,stepexacts,linexacts,randexacts]:
    #     print(sm, sm/ct)
    #
    #
    # bucketx = []
    # buckety = []
    # for b in sorted(list(bucket_counter.keys())):
    #     bucketx.append((max(bucket_width, 1) * b))
    #     buckety.append(bucket_counter[b])
    # plt.plot(bucketx, buckety)
    # plt.show()
    #
    # # apc_df.index = apc_df["assignment"]
    # # apc_df.drop("assignment", inplace=True)
    # apc_df.to_csv("apc38_df.csv")
    # y_del_vals = []
    # y_randels = []
    # y_n1dels = []
    # y_lindels = []
    # y_actuals = []
    # for b in mlookup:
    #     dels, sdels, randels, n1dels, lin_dels = zip(*mlookup[b])
    #     n_samples = len(dels)
    #     totdels = sum(dels)
    #     totsdels = sum(sdels)
    #     # mu = totdels / n_samples
    #     # smu = totsdels / n_samples
    #     mu = mean(dels)
    #     smu = mean(sdels)
    #     smed = median(sdels)
    #     sig = stdev(sdels) if len(sdels)>1 else 0.0
    #     print(b, n_samples, mu, smu, smed, sig)
    #     y_del_vals.append(mean(sdels))
    #     y_randels.append(mean(randels))
    #     y_n1dels.append(mean(n1dels))
    #     y_lindels.append(mean(lin_dels))
    #     y_actuals.append(0)
    #
    # xvals = sorted(mlookup.keys())
    # for yvals in y_actuals, y_del_vals, y_randels, y_n1dels, y_lindels:
    #     plt.scatter(xvals, yvals)
    #     new_buckets = numpy.linspace(xvals[0], xvals[-1], 50)
    #     # smooth = spline(xvals, yvals, new_buckets)
    #     # s = InterpolatedUnivariateSpline(xvals, yvals)
    #     # ynew = s(new_buckets)
    #     z = numpy.polyfit(xvals, yvals, 2)
    #     p = numpy.poly1d(z)
    #     plt.plot(new_buckets, p(new_buckets))
    #     # plt.plot(xvals,yvals)
    # labels = ['actual', 'hwgen', 'random', 'canonical', 'linear']
    #
    # # if group_data is not None:
    # #     g_y_sumvals = []
    # #     g_y_votevals = []
    # #     for b in glookup:
    # #         sumdels, votedels = zip(*glookup[b])
    # #         g_y_sumvals.append(mean(sumdels))
    # #         g_y_votevals.append(mean(votedels))
    # #     xvals = sorted(glookup.keys())
    # #     for yvals in g_y_sumvals, g_y_votevals:
    # #         plt.scatter(xvals, yvals)
    # #         new_buckets = numpy.linspace(xvals[0], xvals[-1], 50)
    # #         z = numpy.polyfit(xvals, yvals, 2)
    # #         p = numpy.poly1d(z)
    # #         plt.plot(new_buckets, p(new_buckets))
    # #     labels = labels + ["group (sum)", "group (vote)"]
    #
    # plt.legend(labels)
    # plt.xlabel("Student time elapsed (months)")
    # plt.ylabel("Relative position in syllabus (qns)")
    # plt.show()


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
