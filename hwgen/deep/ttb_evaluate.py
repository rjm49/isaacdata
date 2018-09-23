from random import choice

import numpy
from collections import Counter
from statistics import mean, median, stdev

import pandas
from sklearn.externals import joblib

from hwgen.common import get_q_names, get_student_list, get_user_data, get_all_assignments, make_gb_question_map
from matplotlib import pyplot as plt

gb_qmap = make_gb_question_map()


def evaluate2(tt, sxua, model, sc,fs, load_saved_data=False):
    if load_saved_data:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load(
            "tt.data")
    else:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(
            tt, sxua)
    print(s_list.shape)
    try:
        s_list = sc.transform(s_list)
    except ValueError as ve:
        print(ve)
        print("Don't forget to check your flags... maybe you have do_train==False and have changed S ...")


    x_list = x_list[:, fs]
    deltas = []
    signed_deltas = []
    exacts = 0
    for sl,xl,hxtt,hxtd in zip(s_list,x_list,hexes_to_try_list, hexes_tried_list):
        predictions = model.predict([sl.reshape(1,-1), xl.reshape(1,-1)])
        y_hats = list(reversed(numpy.argsort(predictions)[0]))
        for candix in y_hats:
            if pid_override[candix] not in hxtd:
                y_hat = candix
                break

        p_hat = pid_override[y_hat]
        # y_trues = []
        # for p_true in sorted(hxtt):
        # p_true = sorted(hxtt)[(len(hxtt)-1) //2]
        #     y_trues.append(pid_override.index(p_true))
        # y_true = median(y_trues)
        p_true = sorted(hxtt)[0]
        y_true = pid_override.index(p_true)
        if(y_true == y_hat):
            exacts+=1
        # p_true = pid_override[int(y_true)]
        print(y_hat, y_true, p_hat, p_true)
        deltas.append( abs((y_hat-y_true)))
        signed_deltas.append( (y_hat - y_true))
    mu = numpy.mean(deltas)
    signed_mu = numpy.mean(signed_deltas)
    sigma = numpy.std(deltas) if len(deltas)>1 else 0.0
    print("Mean of {} difference from teachers, std={}".format(mu,sigma))
    print("Signed delta is {}".format(signed_mu))
    print("{} exact matches out of {} = {}".format(exacts, len(s_list), (exacts/len(s_list))))
    # exit()

def evaluate3(tt,sxua, model, sc,fs, load_saved_data=False, pid_override=None):
    maxdops = 700
    if load_saved_data:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load(
            "tt.data")
    else:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(
            tt, sxua)
    print(s_list.shape)
    try:
        s_list = sc.transform(s_list)
    except ValueError as ve:
        print(ve)
        print("Don't forget to check your flags... maybe you have do_train==False and have changed S ...")

    x_list = x_list[:, fs]
    ct = 0
    if maxdops:
        dops = [ sr[1] for sr in s_raw_list if sr[1] <= maxdops ]
    else:
        dops = [ sr[1] for sr in s_raw_list ]
    dopdelta = max(dops)

    delta_dict = {}
    exact_ct = Counter()
    strat_list = ["hwgen","step","lin","random"]

    for sl,sr,xl,hxtt,hxtd in zip(s_list,s_raw_list,x_list,hexes_to_try_list, hexes_tried_list):
        if sr[1] > maxdops:
            continue
        predictions = model.predict([sl.reshape(1,-1), xl.reshape(1,-1)])
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
        lin_y_hat = int((41) * (sr[1] / dopdelta))

        # y_trues = []
        # for p_true in sorted(hxtt):
        # p_true = sorted(hxtt)[(len(hxtt)-1) //2]
        #     y_trues.append(pid_override.index(p_true))
        # y_true = median(y_trues)
        p_true = sorted(hxtt)[0]
        y_true = pid_override.index(p_true)

        for strat, y_hat in zip(strat_list, [hwgen_y_hat, step_y_hat, lin_y_hat, random_y_hat]):
            if strat not in delta_dict:
                delta_dict[strat] = []
            delta_dict[strat].append(((y_hat-y_true)))
            if(y_true == y_hat):
                exact_ct[strat]+=1

        ct += 1

    for strat in strat_list:
        delta_list = delta_dict[strat]
        sq_list = [ d*d for d in delta_list ]
        mu = numpy.mean(delta_list)
        medi = numpy.median(delta_list)
        stdev = numpy.std(delta_list)
        print("{}: mean={} med={} std={}".format(strat, mu, medi, stdev))
        print("Exact = {} of {} = {}".format(exact_ct[strat], ct, (exact_ct[strat]/ct)))
        print("MSE = {}\n".format(numpy.mean(sq_list)))
    input("tam")

def evaluate_by_bucket(tt,sxua, model, sc,fs, load_saved_data=False, group_data = None, pid_override=None):
    bucket_step = 30
    bucket_width = 7

    # first_asst = {}
    all_assts = get_all_assignments()
    # for ts, grid in zip(list(all_assts["creation_date"]), list(all_assts["group_id"])):
    #     # print(ts,grid)
    #     students = get_student_list(grid)
    #     for psi in students:
    #         if psi not in first_asst:
    #             first_asst[psi] = ts


    buckets = [i for i in range(13)] #22 for bw 30
    if load_saved_data:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load(
            "tt.data")
    else:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(
            tt, sxua)
    # dops = [ s[1] for s in s_raw_list ]

    first_asst = {}
    for psi, aidts in zip(psi_list, ts_list):
        if psi not in first_asst:
            first_asst[psi] = aidts

    dops = [ (ts - first_asst[psi]).days for psi,ts in zip(psi_list, ts_list) ]

    s_list = sc.transform(s_list)
    exacts = 0
    stepexacts = 0
    linexacts = 0
    randexacts = 0
    mlookup = {}
    glookup = {}

    # dopdelta = min(max(dops),7+30*max(buckets)) - max(min(dops),30*min(buckets)-7)
    dopdelta = min(max(dops),(max(bucket_step,1)*max(buckets)))
    assert fs is not None
    x_list = x_list[:, fs]

    PLOT_MODE = "DELTA"
    collist = ["assignment", "group", "teacher", "student", "bucket", "actdop", "y_true", "random", "linear", "canonical", "hwgen"]
    if group_data:
        collist += ["group_sum","group_vote"]
    apc_df = pandas.DataFrame(columns=collist)
    report_ix = 0
    ct = 0
    bucket_counter = Counter()

    tid_pnt = 0
    psi_pnt = 0
    anon_tids = {}
    anon_psis = {}
    for aid,grid,psi,sl,actdop,xl,hxtt, hxtd, aidts in zip(aid_list,gr_id_list,psi_list,s_list, dops, x_list,hexes_to_try_list, hexes_tried_list, ts_list):
        # xl = top_n_of_X(xl,fs)
        predictions = model.predict([sl.reshape(1,-1), xl.reshape(1,-1)])

        # actdop = (aidts - first_asst[psi]).days

        teacher_id = (all_assts[all_assts["group_id"]==grid]["owner_user_id"]).iloc[0]
        # if teacher_id not in anon_tids:
        #     anon_tids[teacher_id] = tid_pnt
        #     tid_pnt +=1
        # teacher_id = anon_tids[teacher_id]

        # if psi not in anon_psis:
        #     anon_psis[psi] = psi_pnt
        #     psi_pnt +=1
        # psi = anon_psis[psi]

        for b in buckets:
            if abs((bucket_step*b) - actdop)<=bucket_width:
                bucket_counter[b] += 1
                ct += 1
                y_hats = []
                y_hats_raw = list(reversed(numpy.argsort(predictions)[0]))
                yix = 0
                while len(y_hats)<len(hxtt):
                    cand = pid_override[y_hats_raw[yix]]
                    yix+=1
                    if cand not in hxtd:
                        y_hats.append(pid_override.index(cand))

                # p_hat = pid_override[y_hat]

                #RANDOM SELECTION
                options = [p for p in pid_override if p not in hxtd]

                #N+1
                hts = [h for h in hxtd if (h in pid_override)]
                # n1_p_hat = [h for h in sorted(hxtt) if (h in pid_override)][0]
                # n1_y_hat = pid_override.index(n1_p_hat)

                ix = 0
                for p_true, y_hat in zip(sorted([hxtt[0]]), [y_hats[0]]):
                    random_p_hat = choice(options)
                    random_y_hat = pid_override.index(random_p_hat)
                    n1_y_hat = 0 if not hts else min(len(pid_override) - 1, pid_override.index(hts[-1]) + ix)
                    lin_y_hat = int((len(pid_override)-2)*(bucket_step*b / dopdelta))
                    ix+=1

                    # y_trues=[]
                    # for p_true in sorted(hxtt):
                    #     y_trues.append(pid_override.index(p_true))
                    # y_true = mean(y_trues)
                    y_true = pid_override.index(sorted(hxtt)[0])

                    # y_true = pid_override.index(p_true)
                    if (y_true == y_hat):
                        exacts += 1
                    if (y_true == n1_y_hat):
                        stepexacts += 1
                    if (y_true == lin_y_hat):
                        linexacts += 1
                    if (y_true == random_y_hat):
                        randexacts += 1

                    #print(y_hat, y_true, p_hat, p_true)delta = abs(y_hat - y_true)


                    if PLOT_MODE=="ABS":
                        delta = abs(y_hat - y_true)
                        signed_delta = abs(y_hat - y_true)
                        rand_delta = abs(random_y_hat - y_true)
                        n1_delta = abs(n1_y_hat - y_true)
                        lin_delta = abs(lin_y_hat - y_true)
                    elif PLOT_MODE=="DELTA":
                        delta = (y_hat - y_true)
                        signed_delta = (y_hat - y_true)
                        rand_delta = (random_y_hat - y_true)
                        n1_delta = (n1_y_hat - y_true)
                        lin_delta = (lin_y_hat - y_true)
                    elif PLOT_MODE=="RAW":
                        delta = y_hat
                        signed_delta = y_hat
                        rand_delta = random_y_hat
                        n1_delta = n1_y_hat
                        lin_delta = lin_y_hat
                    else:
                        raise ValueError("{} is an invalid plotting mode!".format(PLOT_MODE))

                    replist = [int(aid),grid,teacher_id, psi, bucket_step*b, actdop, y_true, random_y_hat, lin_y_hat, n1_y_hat, y_hat]
                    if group_data is not None:
                        (gr_sum_ix, gr_vote_ix) = group_data[aid]
                        if b not in glookup:
                            glookup[b] = [ (gr_sum_ix - y_true, gr_vote_ix - y_true) ]
                        else:
                            deltas = glookup[b]
                            deltas.append( (gr_sum_ix - y_true, gr_vote_ix - y_true) )
                        replist += [gr_sum_ix, gr_vote_ix]

                    if not b in mlookup:
                        mlookup[b] = [ (delta, signed_delta, rand_delta, n1_delta, lin_delta) ]

                    else:
                        deltas = mlookup[b]
                        deltas.append( (delta, signed_delta, rand_delta, n1_delta, lin_delta) )
                        mlookup[b] = deltas

                    apc_df.loc[report_ix] = replist
                    report_ix += 1

    for sm in [exacts,stepexacts,linexacts,randexacts]:
        print(sm, sm/ct)


    bucketx = []
    buckety = []
    for b in sorted(list(bucket_counter.keys())):
        bucketx.append((max(bucket_width, 1) * b))
        buckety.append(bucket_counter[b])
    plt.plot(bucketx, buckety)
    plt.show()

    # apc_df.index = apc_df["assignment"]
    # apc_df.drop("assignment", inplace=True)
    apc_df.to_csv("apc38_train_df.csv")
    y_del_vals = []
    y_randels = []
    y_n1dels = []
    y_lindels = []
    y_actuals = []
    for b in mlookup:
        dels, sdels, randels, n1dels, lin_dels = zip(*mlookup[b])
        n_samples = len(dels)
        totdels = sum(dels)
        totsdels = sum(sdels)
        # mu = totdels / n_samples
        # smu = totsdels / n_samples
        mu = mean(dels)
        smu = mean(sdels)
        smed = median(sdels)
        sig = stdev(sdels) if len(sdels)>1 else 0.0
        print(b, n_samples, mu, smu, smed, sig)
        y_del_vals.append(mean(sdels))
        y_randels.append(mean(randels))
        y_n1dels.append(mean(n1dels))
        y_lindels.append(mean(lin_dels))
        y_actuals.append(0)

    xvals = sorted(mlookup.keys())
    for yvals in y_actuals, y_del_vals, y_randels, y_n1dels, y_lindels:
        plt.scatter(xvals, yvals)
        new_buckets = numpy.linspace(xvals[0], xvals[-1], 50)
        # smooth = spline(xvals, yvals, new_buckets)
        # s = InterpolatedUnivariateSpline(xvals, yvals)
        # ynew = s(new_buckets)
        z = numpy.polyfit(xvals, yvals, 2)
        p = numpy.poly1d(z)
        plt.plot(new_buckets, p(new_buckets))
        # plt.plot(xvals,yvals)
    labels = ['actual', 'hwgen', 'random', 'canonical', 'linear']

    if group_data is not None:
        g_y_sumvals = []
        g_y_votevals = []
        for b in glookup:
            sumdels, votedels = zip(*glookup[b])
            g_y_sumvals.append(mean(sumdels))
            g_y_votevals.append(mean(votedels))
        xvals = sorted(glookup.keys())
        for yvals in g_y_sumvals, g_y_votevals:
            plt.scatter(xvals, yvals)
            new_buckets = numpy.linspace(xvals[0], xvals[-1], 50)
            z = numpy.polyfit(xvals, yvals, 2)
            p = numpy.poly1d(z)
            plt.plot(new_buckets, p(new_buckets))
        labels = labels + ["group (sum)", "group (vote)"]

    plt.legend(labels)
    plt.xlabel("Student time elapsed (months)")
    plt.ylabel("Relative position in syllabus (qns)")
    plt.show()

def evaluate_phybook_loss(tt,sxua, model, sc, load_saved_data=False):

    if load_saved_data:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load("tt.data")
    else:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(tr, sxua)
        joblib.dump( (aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list),"tt.data")

    # hex_list = []
    # all_page_ids = pid_override
    # ailist = []

    for row in tt.iterrows():
        aid = row[1]["id"]
        # ts = row[1]["creation_date"]
        gr_id = row[1]["group_id"]
        gb_id = row[1]["gameboard_id"]
        student_ids = list(get_student_list(gr_id)["user_id"])
        print(student_ids)
        student_data = get_user_data(student_ids)
        hexes= list(gb_qmap[gb_id])
        print(hexes)

        for _ in student_ids:
            aid_list.append(aid)
            # hex_list.append(hexes)

    s_list = sc.transform(s_list)
    s_list = numpy.array(s_list)

    x_list = numpy.array(x_list)
    u_list = numpy.array(u_list)
    a_list = numpy.array(a_list)

    print(s_list.shape, x_list.shape, u_list.shape, a_list.shape)

    print("results")
    print(model.get_input_shape_at(0))
    # x_list = top_n_of_X(x_list,fs)
    predictions = model.predict([s_list, x_list])
    j_max = 0
    thresh_max = 0
    dir_hits_max = 0
    for j_thresh in [0.001, 0.005, 0.01, 0.025, .05, 0.075, .1,.2,0.3, 0.4, 0.5, 0.6, 0.7]:
    # for j_thresh in [0.4]:
        j_sum = 0
        # dir_sum = 0
        incl_sum = 0
        dir_hits = 0
        N = len(predictions)
        this_ai = None
        for ai, p, s,x,a,y in zip(aid_list, predictions, s_list,x_list,a_list,y_list):
            t = [pid_override[yix] for yix,yval in enumerate(y) if yval==1]
            if ai != this_ai:
                print("\n...new asst",ai)
                this_ai = ai
            phxs = []
            probs = []
            print("pshape",p.shape)
            maxpox = numpy.argmax(p)
            print(maxpox, len(pid_override))
            max_guess = pid_override[maxpox]
            phxs.append(max_guess)

            probs.append(p[maxpox])
            for ix, el in enumerate(p):
                if el>j_thresh and pid_override[ix] not in phxs:
                    phxs.append(pid_override[ix])
                    probs.append(p[ix])
            probs_shortlist = list(reversed(sorted(probs)))
            Z = list(reversed( [x for _, x in sorted(zip(probs, phxs))] ))
            # if Z:
            #     for t_el in t:
            #         if t_el in Z:#'direct hit'
            #             dir_sum += 1.0/len(t)
            print(t, Z)
            print(probs_shortlist)
            # print([all_page_ids[hx] for hx,el in enumerate(a) if el==1])
            if max_guess not in t:
                robot="BAD ROBOT"
            else:
                if max_guess == t[0]:
                    robot = "GREAT ROBOT"
                    dir_hits += 1
                else:
                    robot = "GOOD ROBOT"
            print("{} {}, XP={}".format(robot, sc.inverse_transform(s), numpy.sum(x)))
            t=set(t)
            phxs=set(phxs)
            if len(t.intersection(phxs)) > 0:
                incl_sum += 1
            j_sum  += len(t.intersection(phxs)) / len(t.union(phxs))
        j_score = j_sum/N
        # dir_score = dir_sum/N
        if dir_hits > dir_hits_max:
            j_max = j_score
            thresh_max = j_thresh
            dir_hits_max = dir_hits
            # dir_for_j_max = dir_score
        print("j_thresh =",j_thresh)
        print("Jaccard:", j_score)
        print("Incl:", incl_sum/N)
        print("D/H:", dir_hits/N)
        print("~ ~ ~ ~")
    print("max thresh/jacc:", thresh_max, j_max, dir_hits_max/N)
    print("num examples", N)


def class_evaluation(_tt, sxua, model, sc, fs, load_saved_data=False, pid_override=None):
    names_df = get_q_names()
    names_df.index = names_df["question_id"]

    if load_saved_data:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load("tt.data")
    else:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(_tt, sxua)
        joblib.dump( (aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list),"tt.data")


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
    for aid in lkk:
        m_list = []
        s_list = []
        x_list = []
        ts, gr_id = ts_grid_lookup[aid]
        sl, srl, xl, ul, al, yl, psil, hxl = lookup[aid]


        xl = numpy.array(xl)[:,fs]
        sl = sc.transform(sl)

        hxtt = hxl[0]

        # x_arr = top_n_of_X(x_arr,fs)
        y_preds = model.predict([sl, xl])
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
        sum_delta = abs(y_true - max_sum_ix)
        vote_delta = abs(y_true - max_vote_ix)
        if max_sum_lab == p_true:
            sum_exact_match += 1
        if max_vote_lab == p_true:
            vote_exact_match += 1
        print(p_true, max_vote_lab, max_sum_lab)
        sum_deltas.append(sum_delta)
        vote_deltas.append(vote_delta)

        result_lkup[aid] = (max_sum_ix, max_vote_ix)

    print("sum mean delta: {}".format(mean(sum_deltas)))
    print("vote mean delta: {}".format(mean(vote_deltas)))
    n_aids = len(lkk)
    print("sum exacts {}/{} = {}".format(sum_exact_match, n_aids, sum_exact_match/n_aids))
    print("vote exacts {}/{} = {}".format(vote_exact_match, n_aids, vote_exact_match/n_aids))
    return result_lkup
