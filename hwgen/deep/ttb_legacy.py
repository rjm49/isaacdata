# def cluster_and_print(assts):
#     xygen = hwgengen2(assts, batch_size=-1, FRESSSH=False)  # make generator object
#     for S, X, U, y, ai, awgt in xygen:
#         y_labs = numpy.array(y)
#         if (X == []):
#             continue
#         S = numpy.array(S)  # strings (labels)
#         X = numpy.array(X)  # floats (fade)
#         U = numpy.array(U)  # signed ints (-1,0,1)
#         print("S", S.shape)
#         print("X", X.shape)
#         print("U", U.shape)
#         assert y_labs.shape[1] == 1  # each line shd have just one hex assignment
#
#         n = 5000
#         lab_set = list(numpy.unique(y_labs))
#         colors = numpy.array([lab_set.index(l) for l in y_labs])[0:n]
#
#         # calc_entropies(X,y_labs)
#         # exit()
#
#         # pca = PCA(n_components=2)
#         tsne = TSNE(n_components=2)
#         # converted = pca.fit_transform(X) # convert experience matrix to points
#         converted = tsne.fit_transform(X[0:n])
#
#         plt.scatter(x=converted[:, 0], y=converted[:, 1], c=colors, cmap=pylab.cm.cool)
#         plt.show()
#         plt.savefig("learning_plot.png")
import math
from collections import defaultdict

def top_n_of_X(X, fs):
    # top_n_features = [412, 413, 415, 417, 420, 421, 429, 431, 655, 656, 657, 658, 659, 660, 661,
    #                   662, 663, 664, 666, 667, 668, 669, 670, 671, 672, 673, 683, 685, 686, 687,
    #                   688, 689, 690, 691, 693, 694, 695, 696, 697, 2494]
    # print(X.shape)
    # Xa = X[:,top_n_features]
    #     return Xa
    if fs is None:
        return X
    else:
        print(fs)
        Xa = fs.transform(X)
        return Xa


def calc_entropies(X, y):
    d = defaultdict(list)
    for x, lab in zip(X, y):
        d[str(lab)].append(x)
    for l in d:
        # print("calc for {}, len {}".format(l,len(d[l])))
        H = entropy(d[l])
        print("{}\t{}\t{}".format(l, H, len(d[l])))


def entropy(lizt):
    "Calculates the Shannon entropy of a list"
    # get probability of chars in string
    lizt = [tuple(e) for e in lizt]
    prob = [float(lizt.count(entry)) / len(lizt) for entry in lizt]
    # calculate the entropy
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy

def evaluate_hits_against_asst(ailist, y, max_y, y_preds, ylb):
    '''This method should take an N~question assignment, compare it against the top N question candidates from the hwgen
    system and return a score between [0..1]'''
    true_qs = set(y)
    N = len(true_qs)
    # print("N",N)

    agg_cnt = Counter()
    for row in y_preds:
        args = numpy.argsort(row)
        topN = list(reversed(args))[0:N]
        for t in topN:
            # print(t)
            tlab = ylb.classes_[t]
            # print(tlab)
            agg_cnt[tlab] += 1
    print(agg_cnt.most_common(N))

    agg_pred = y_preds.sum(axis=0)
    print(agg_pred.shape)
    print(y_preds.shape)

    assert agg_pred.shape[0] == y_preds.shape[1]  # shd have same num of columns as y_preds
    sortargs = numpy.argsort(agg_pred)
    sortargs = list(reversed(sortargs))
    maxN = sortargs[0:N]  # get the top N
    pred_qs = [ylb.classes_[ix] for ix in maxN]
    print(true_qs, " vs ", pred_qs)
    score = len(true_qs.intersection(pred_qs)) / N
    print("score = {}".format(score))
    return score
