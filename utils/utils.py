'''
Created on 7 Sep 2017

@author:  Russell Moore
'''
from collections import Counter, OrderedDict
import numpy as np

ATT_TIM=0
ATT_COR=1
ATT_QID=2

def extract_runs_w_timestamp(attempts):
    times = []
    qids = []
    cors = []
    lens = []
    num_correct = 0
    num_attempts = 0
    c = Counter()
    run_qid=None

    for ix,att_series in attempts.iterrows():
        #we want to isolate contiguous atts against a question into "qids" and determine whether each run was successful or not
        #since students can repeatedly run against a question, it is not sufficient just to filter on the question ID
        new_run_qid = att_series[ATT_QID]
        is_corr = att_series[ATT_COR]
        tm = att_series[ATT_TIM]
        
        if(new_run_qid != run_qid):
#             print("new run")
            qids.append(run_qid)
            lens.append(num_attempts)
            cors.append(num_correct)
            times.append(tm)
            run_qid = new_run_qid
            num_correct = 1 if is_corr else 0
            num_attempts =1
        else:
            num_attempts += 1
            num_correct += (1 if is_corr else 0)

    qids.append(run_qid)
    lens.append(num_attempts)
    cors.append(num_correct)
    times.append(tm)

    
    uni = list( zip(times, qids,lens,cors) )
    return uni[1:] #lose the first entry, which is a dummy


def balanced_supersample(x,y, _prop_of_max=1.0):
    class_xs = []
    max_elems = None
    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if max_elems == None or elems.shape[0] > max_elems:
            max_elems = elems.shape[0]
            print("max elemens class is {} with {} members".format(yi, max_elems))
    use_elems = max_elems
    xs = []
    ys = []

    for ci,this_xs in class_xs:
        x_ = np.copy(this_xs)
        ix_ = np.arange(len(this_xs))
        target = int(_prop_of_max * use_elems)
        n_to_fill = max(len(this_xs),target) - len(this_xs)
        if n_to_fill > 0:
            rix = np.random.choice(ix_, n_to_fill) #choose random indices
            newels = x_[rix] #lookup elements by index
            x_ = np.vstack((x_, newels)) #append elements

        y_ = np.empty(len(this_xs)+n_to_fill)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    return xs,ys

def balanced_subsample(x,y,subsample_size=1.0):
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys