import os, sys

import pandas
from sklearn.naive_bayes import BernoulliNB

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backfit.BackfitUtils import init_objects
from backfit.utils.utils import DW_STRETCH, DW_LEVEL, calc_qdiff, load_new_diffs, DW_NO_WEIGHT, DW_BINARY, DW_NATTS, \
    DW_PASSRATE, load_mcmc_diffs, DW_MCMC
from backfit.BackfitTest import train_and_test

print(sys.path)
from sklearn.metrics.classification import f1_score
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import SVC, LinearSVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.dummy import DummyClassifier

import pandas as pd
import numpy
from utils.utils import extract_runs_w_timestamp

QENC_QUAL=False
QENC_DIFF=False
qenc_width = 33
n_classes = 3

FEAT_F33 = "F33"

n_users = 1000
max_runs = None #10000
percTest = 0.20

def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n
    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]


if __name__ == '__main__':

    cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users, seed=666)
    boards = pandas.read_csv("gameboards.txt", sep="~", header=0, index_col=None)
    groups = pandas.read_csv("groups.txt", sep=",", header=0, index_col=None)# gameboard_id	user_id	group_id	assignment_id

    passdiffs, stretches, passquals, all_qids = load_new_diffs()
    mcmcdiffs = load_mcmc_diffs()

    shredfile = open("shredded.csv", "w")

    tbc = 0
    bc = 0
    ucnt = 0
    for ix,board in boards.iterrows():
        tbc += 1
        board_id = board[0]
        board_qs = board[2][1:-1].split(",") # List of qn page IDs
        if(len(board_qs)<2):
            continue

        board_users = groups[groups.iloc[:,0]==board_id].iloc[:,1].unique()
        if board_users.shape[0]==0:
            continue

        bc+=1
        shredfile.write("{}~{}~{}\n".format(str(board_id)+"~"+str(board[1]), " ".join(map(str, board_qs)), len(board_qs)))

        for u in users:
            u_qs = []
            print("user = ", u)
            attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)
            runs = extract_runs_w_timestamp(attempts)
            if len(runs)==0:
                continue
            for run in runs:
                ts, q, n_atts, n_pass = run
                qt = q.replace("|","~")
                page = q.split("|")[0]
                if page in board_qs and page not in u_qs: #append distinct values
                    u_qs.append(page)
            if(len(u_qs)>0):
                ucnt += 1
                shredfile.write(str(int(u))+"\t"+" ".join(map(str, u_qs)))
                lstein = levenshtein(board_qs, u_qs)
                normd_lstein = lstein / float(len(board_qs))
                shredfile.write("\t{}\t{}\t{}\t{}\n".format(len(board_qs), len(u_qs), lstein, normd_lstein))
                shredfile.flush()
    shredfile.close()
