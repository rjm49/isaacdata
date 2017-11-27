'''
Created on 11 Sep 2017

@author:  Russell Moore
'''
from sklearn.model_selection import RandomizedSearchCV
import numpy
from scipy.stats import randint as sp_randint
import copy


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = numpy.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# run randomized search
def run_random_search(clf, X, y, param_dist):
    n_iter_search = param_dist['n_iter']
    #del param_dist['n_iter']
    pcopy = copy.copy(param_dist)
    del pcopy['n_iter']
    del pcopy['name']
    random_search = RandomizedSearchCV(clf, param_distributions=pcopy,
                                   n_iter=n_iter_search, n_jobs=-1)

#     print("RandomizedSearchCV took %.2f seconds for %d candidates"
#             " parameter settings." % ((time() - start), n_iter_search))
    random_search.fit(X, y)
    report(random_search.cv_results_)

    return random_search.best_estimator_