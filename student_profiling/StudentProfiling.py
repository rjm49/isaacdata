from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot

from backfit.BackfitUtils import init_objects
from backfit.utils.utils import load_new_diffs, load_mcmc_diffs
from utils.utils import extract_runs_w_timestamp

if __name__ == '__main__':
    n_users = -1
    cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users, seed=666)
    passdiffs, stretches, passquals, all_qids = load_new_diffs()
    all_qids = list(all_qids)

    users = np.unique(users)
    n_users = len(users)
    print("kickoff for n_users?",n_users)

    n_qids = 1+len(all_qids) #+1 to represent ROOT

    usersf = open("direct_mcmc_users.txt","w")

    udf = pd.read_csv("users_all.csv")
    students = udf[ (udf["role"]=="STUDENT") & (udf["date_of_birth"].notnull()) ]
    print(students["date_of_birth"])
    ages = ( pd.to_datetime(students['registration_date']) - pd.to_datetime(students['date_of_birth'])).dt.total_seconds() / (86400*365.2425)
    print(ages)

    binwidth=0.25
    binz = np.arange(min(ages), max(ages) + binwidth, binwidth)
    ages.plot.hist(alpha=0.5, bins=binz)
    print(np.median(ages))
    print(np.mean(ages))
    print(np.std(ages))
    pyplot.show()