import gc
from collections import Counter
from random import choice

import numpy

from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, concatenate, Dropout
from keras.optimizers import Adam
from matplotlib import pyplot
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

from hwgen.common import get_q_names
from hwgen.deep.TrainTestBook import make_phybook_model, save_class_report_card
from hwgen.deep.preproc import augment_data, augment_data_old


def train_deep_model(tr, sxua, qid_map, pid_map, sugg_map, n_macroepochs=100, n_epochs=10, use_linear=False, load_saved_tr=False, filter_by_length=True, model_generator=make_phybook_model):
    code="tr"
    model = None
    if load_saved_tr:
        aid_list, s_list, x_list, c_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load("{}.data".format(code))
    else:
        fs = None
        aid_list, s_list, x_list, c_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data_old(tr, sxua, filter_by_length=filter_by_length, pid_map=pid_map, sugg_map=sugg_map, filecode="tr")
        joblib.dump( (aid_list, s_list, x_list, c_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list),"{}.data".format(code))

    # exit() # TODO remove for production
    del aid_list
    del a_list
    del ts_list
    del s_raw_list
    del psi_list
    del gr_id_list
    del hexes_to_try_list
    del hexes_tried_list
    gc.collect()

    SCALE = True
    sc = StandardScaler()

    if SCALE:
        # print(x_list.shape)
        lenX = s_list.shape[0]
        # for ix,x_el in enumerate(x_list):
        #     x_list[ix] = sc.transform(x_list[ix].reshape(1,-1))

        start = 0
        gap = 5000
        while(start<lenX):
            end = min(start+gap, lenX)
            print("fitting scaler",start,end)
            partial_s = s_list[start:end,:]
            sc.partial_fit(partial_s)
            start += gap
        # sc.fit(s_list)

        start = 0
        while(start<lenX):
            end = min(start+gap, lenX)
            print("scaling",start,end)
            s_list[start:end,:] = sc.transform(s_list[start:end,:])
            start += gap
        # s_list = sc.transform(s_list)

    max_mod = None

    x_mask = None
    # x_mask = numpy.nonzero(numpy.any(x_list != 0, axis=0))[0]
    # x_list = x_list[:, x_mask]

    if model is None:
        lrs = []
        accs = []
        BSs = []
        max_acc = -1
        max_BS = -1

        print("making model")
        # x_list = top_n_of_X(x_list, fs)
        # S,X,U,A = s_list[0], x_list[0], u_list[0], a_list[0]
        # print(S.shape, X.shape, U.shape, A.shape, y_list.shape)
        S,X,U,C = s_list[0], x_list[0], u_list[0], c_list[0]

        es = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
        # cves = EarlyStopping(monitor='acc', patience=1, verbose=0, mode='auto')
        # for BS in [50, 64, 100]:
        #     for LR in [0.003, 0.0025, 0.002]:
        # for BS in [16,32,64,128]:
        for BS in [32]: #80
            # for LR in [0.0001, 0.001, 0.01, 0.1]:
            for LR in [0.001]: #0.0015
                model = model_generator(len(S), len(X), len(U), len(C), len(y_list[0]), lr=LR)
                # es = EarlyStopping(monitor='categorical_accuracy', patience=0, verbose=0, mode='auto')
                # history = model.fit([s_list, x_list], [y_list, x_list], verbose=1, epochs=100, callbacks=[es], shuffle=True, batch_size=32)

                # cv = KFold(n_splits=3, shuffle=True, random_state=666)
                # splits = cv.split(s_list, y_list)
                # for trixs,ttixs in splits:
                #     s_tr = s_list[trixs]
                #     x_tr = x_list[trixs]
                #     s_tt = s_list[ttixs]
                #     x_tt = x_list[ttixs]
                #     y_tr = y_list[trixs]
                #     y_tt = y_list[ttixs]
                #     history = model.fit([s_tr, x_tr], y_tr, validation_data=([s_tt,x_tt],y_tt), verbose=1, epochs=100, callbacks=[es], shuffle=True, batch_size=BS)

                # history = model.fit([s_list, x_list, u_list, c_list], y_list, verbose=1, validation_split=0.20, epochs=100, callbacks=[es], shuffle=True, batch_size=BS)
                # scores = model.evaluate([s_list,x_list, u_list, c_list],y_list)
                x_list = numpy.array(x_list)
                s_list = numpy.array(s_list)
                y_list = numpy.array(y_list)

                history = model.fit([s_list, x_list], y_list, verbose=1, validation_split=0.2, epochs=100, callbacks=[es], shuffle=True, batch_size=BS)
                scores = model.evaluate([s_list,x_list],y_list)
                print(scores)
                lrs.append(LR)
                accs.append( scores[1])
                BSs.append(BS)
                if scores[1] > max_acc:
                    max_mod = model
                    max_acc = scores[1]
                print(scores)

            do_plot = True
            if do_plot:
                try:
                    pyplot.plot(history.history['acc'])
                    # pyplot.plot(history.history['binary_crossentropy'])
                    # pyplot.plot(history.history['categorical_crossentropy'])
                    pyplot.plot(history.history['top_k_categorical_accuracy'])
                    pyplot.plot(history.history['loss'])
                    pyplot.plot(history.history['val_acc'])
                    pyplot.plot(history.history['val_loss'])
                    pyplot.plot(history.history['val_top_k_categorical_accuracy'])
                    pyplot.legend(["cat acc","top k acc","loss","val cat acc","val loss","val_top_k"])
                    pyplot.show()
                except:
                    print("some problem occurred during plot .. ignoring")
                    pass

        max_acc_ix = accs.index(max(accs))
        print((max(accs), lrs[max_acc_ix], BSs[max_acc_ix]))
    return max_mod, x_mask, sc  # , sscaler, levscaler, volscaler


def create_student_scorecards(tt, sxua, model, sc, fs, qid_map, pid_map, sugg_map):
    names_df = get_q_names()
    names_df.index = names_df["question_id"]

    aid_list, s_list, x_list, c_list, u_list, a_list, y_true_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(tt, sxua, filter_by_length=False, pid_map=pid_map, sugg_map=sugg_map, filecode="tt_scards")
    joblib.dump( (aid_list, s_list, x_list, c_list, u_list, a_list, y_true_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list),"tt.data")

    for y_true in y_true_list:
        y_true_ix = numpy.argmax(y_true)
        print("y true ix is {}".format(y_true_ix))
        y_true_lab = sugg_map[y_true_ix]
        print("y true lab is {}".format(y_true_lab))
        print("sum is:",sum(y_true))

    # if fs is not None:
    #     x_list = x_list[:, fs]
    #     print(x_list.shape)
        # po_filtered = [pid_map[fsix] for fsix in fs]
    # else:
    #     po_filtered = pid_map

    # for row in tt.iterrows():
    lookup = {}
    ts_grid_lookup = {}
    for aid,s_raw,s,x,u,c,a,y,psi,grid,ts in zip(aid_list, s_raw_list, s_list, x_list, u_list, c_list, a_list, y_true_list, psi_list, gr_id_list, ts_list):
        if aid not in lookup:
            lookup[aid] = ([],[],[],[],[],[],[],[])
            ts_grid_lookup[aid] = (ts,grid)
        sr,sl,xl,ul,cl,al,yl,psil = lookup[aid]
        sr.append(s_raw)
        sl.append(s)
        xl.append(x)
        ul.append(u)
        cl.append(c)
        al.append(a)
        yl.append(y)
        psil.append(psi)
        lookup[aid] = (sr,sl,xl,ul,cl,al,yl,psil)

    lkk = list(lookup.keys())

    for aid in lkk:
        m_list = []
        s_list = []
        x_list = []
        ts, gr_id = ts_grid_lookup[aid]
        sr, sl, xl, ul, cl, al, yl, psil = lookup[aid]
        predictions = []
        for s_raw,s,x,psi in zip(sr,sl,xl,psil):
            # s_list.append(s)
            # x_list.append(x)
            # s_raw_list.append(s_raw)
            m_list.append(s_raw[1])
            print("student {} done".format(psi))

            if len(s_list)==0:
                continue

        s_arr = numpy.array(sl)
        s_arr = sc.transform(s_arr)
        x_arr = numpy.array(xl)
        print(x_arr.shape)
        # u_arr = numpy.array(ul)
        # c_arr = numpy.array(cl)

        predictions = model.predict([s_arr,x_arr]) #,u_arr,c_arr])

        print(numpy.array(yl).shape)
        print(predictions.shape)
        # for ix in range(20):
        #     print(predictions[ix,:])

        save_class_report_card(ts, aid, gr_id, s_raw_list, x_arr, ul, al, yl, m_list, predictions, psil, names_df, pid_map=pid_map, sugg_map=sugg_map)

    with open("a_ids.txt", "w+") as f:
        f.write("({})\n".format(len(aid_list)))
        f.writelines([str(a)+"\n" for a in sorted(aid_list)])
        f.write("\n")



def make_mentoring_model(n_S, n_X, n_U, n_C, n_P, lr):

    # this is our input placeholder
    if n_S:
        input_S = Input(shape=(n_S,), name="student_input_s")
        inner_S = Dense(10, activation="relu")(input_S)

    if n_X:
        input_X = Input(shape=(n_X,), name="encounter_input_x")
        inner_X = Dense(300, activation="relu")(input_X)

    # if n_U:
    #     input_U = Input(shape=(n_U,), name="mastery_input_u")
    #     inner_U = Dense(300, activation="relu")(input_U)
    #
    # if n_C:
    #     input_C = Input(shape=(n_C,), name="count_input_c")
    #     inner_C = Dense(300, activation="relu")(input_C)


    w=200
    #w100: 17% to beat
    do=.2


    if n_S is not None:
        hidden = concatenate([inner_S, inner_X])#, inner_U, inner_C])
    else:
        hidden = inner_X

    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.2)(hidden)
    hidden = Dense(w, activation='relu')(hidden)
    hidden = Dropout(.2)(hidden)
    hidden = Dense(w, activation='relu')(hidden)
    hidden = Dropout(.2)(hidden)
    next_pg = Dense(n_P, activation='softmax', name="next_pg")(hidden)

    o = Adam(lr= lr)

    # m = Model(inputs=[input_S, input_U], outputs=[next_pg, output_U])
    # m.compile(optimizer=o, loss=["binary_crossentropy","binary_crossentropy"], metrics={'next_pg':["binary_accuracy", "top_k_categorical_accuracy"], 'outAuto':['binary_accuracy']})
    if n_S is not None:
        ins = [input_S, input_X] #, input_U, input_C]
    else:
        ins = input_X

    m = Model(inputs=ins, outputs=[next_pg])
    m.compile(optimizer=o, loss='categorical_crossentropy', metrics={'next_pg':['acc', 'top_k_categorical_accuracy']})
    # plot_model(m, to_file='hwgen_model.png')
    m.summary()
    # input(",,,")
    return m

def evaluate3(tt,sxua, model, sc,fs, load_saved_data=False, pid_map = None, sugg_map = None):
    maxdops = 360
    if load_saved_data:
        aid_list, s_list, x_list, c_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load(
            "tt.data")
    else:
        aid_list, s_list, x_list, c_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(
            tt, sxua, pid_map=pid_map, sugg_map=sugg_map, filecode="tt_ev3")
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
            if sugg_map[candix] not in hxtd:
                hwgen_y_hat = candix
                break

        options = [p for p in sugg_map if p not in hxtd]
        hts = [h for h in hxtd if (h in sugg_map)]

        p_hat = sugg_map[hwgen_y_hat]
        random_p_hat = choice(options)
        random_y_hat = sugg_map.index(random_p_hat)
        step_y_hat = 0 if not hts else min(len(sugg_map) - 1, sugg_map.index(hts[-1]) + 1)
        #lin_y_hat = int((len(pid_override) - 2) * (sr[1] / dopdelta))
        lin_y_hat = int((41) * (sr[1] / dopdelta))

        # y_trues = []
        # for p_true in sorted(hxtt):
        # p_true = sorted(hxtt)[(len(hxtt)-1) //2]
        #     y_trues.append(pid_override.index(p_true))
        # y_true = median(y_trues)
        p_true = sorted(hxtt)[0]
        y_true = sugg_map.index(p_true)

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
