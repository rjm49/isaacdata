'''
Created on 13 Oct 2017

@author:  Russell Moore
'''
import numpy
from sklearn.dummy import DummyClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.svm.classes import SVC
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.linear_model.perceptron import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection._search import RandomizedSearchCV
import copy

predictors = [
            #DummyClassifier(strategy="stratified"),
            #DummyClassifier(strategy="uniform"),
            #SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=-1, class_weight="balanced"),
            SVC(class_weight='balanced'),
            SVC(kernel="linear", class_weight='balanced'),

            #PassiveAggressiveClassifier(class_weight="balanced", max_iter=1000, tol=1e-3, n_jobs=-1),
            #MLPClassifier(max_iter=1000, nesterovs_momentum=True, early_stopping=True),
            #Perceptron(max_iter=1000),
            #GaussianNB(),
            #LogisticRegression(class_weight='balanced'),
]

predictor_params = [
                    #None,
                    #None,
#                     {'n_iter':50,'alpha': numpy.logspace(-3, 2) },
                    {'n_iter':50,'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
                    {'n_iter':50,'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
            
                    #None,
                    #{'n_iter':150,'hidden_layer_sizes':[(100,), (66,10)], 'learning_rate_init':[0.001, 0.01, 0.1], 'alpha': numpy.logspace(-6,2) },
                    #None,
                    #None,
                    #None
                    ]

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

def fit_best(X,y):
    re = []
    print("tuning classifier ...")
    for ix,p in enumerate(predictors):
        print(type(p))
        print(p.get_params().keys())
#             input("prompt")
        if predictor_params[ix]!=None:
            pbest = run_random_search(p, X, y, predictor_params[ix])
        else:
            pbest = p.fit(X, y)
        re.append( pbest)
    return re

def run_random_search(clf, X, y, param_dist):
    n_iter_search = param_dist['n_iter']
    #del param_dist['n_iter']
    pcopy = copy.copy(param_dist)
    del pcopy['n_iter']
    random_search = RandomizedSearchCV(clf, param_distributions=pcopy,
                                   n_iter=n_iter_search, n_jobs=-1)

#     print("RandomizedSearchCV took %.2f seconds for %d candidates"
#             " parameter settings." % ((time() - start), n_iter_search))
    random_search.fit(X, y)
    report(random_search.cv_results_)

    return random_search.best_estimator_