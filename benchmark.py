from __future__ import print_function, division

import numpy as np

from itertools import *
from functools import *
from operator import *

from matplotlib import pyplot as plt

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

from poc import *

def cum(f, xs):
    """Cumulative f

    f :: A -> A -> A
    xs :: [A]

    Assumes: f(x[i]) = x[i]  (which is the case for must stuff add(x) = x, max(x) = x, etc)

    Usage: cum(max, [1,3,4,5,1,2,6,2,3]) => [1,3,4,5,5,5,6,6,6]
    """
    if len(xs) > 0:
        state = xs[0]
        yield state

        for value in xs[1:]:
            state = f(state, value)
            yield state


def mean_validation_scores(grid_estimator):
    return list(map(attrgetter("mean_validation_score"), grid_estimator.grid_scores_))

digits = load_digits()
X, y = digits.data, digits.target

n_iter = 16
clf = RandomForestClassifier(
    bootstrap=True,
    criterion="gini",
)

param_dist = {
    "max_features": sp_randint(1, 32),
    "min_samples_split": sp_randint(1, 32),
    "min_samples_leaf": sp_randint(1, 32),
    "n_estimators": sp_randint(2, 64),
    "max_depth": sp_randint(2, 32),
}

random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=n_iter,
    n_jobs=-1,
)

print("random_search.fit(X, y)")
random_search.fit(X, y)

param_grid = {
    "max_features": list(range(1, 32)),
    "min_samples_split": list(range(1, 32)),
    "min_samples_leaf": list(range(1, 32)),
    "n_estimators": list(range(2, 64)),
    "max_depth": list(range(2, 32)),
}

n_initial_points = 8

bayesian_optimization_search = BayesianOptimizationSearchCV(
    clf,
    param_grid=param_grid,
    n_iter=n_iter-n_initial_points,
    n_initial_points=n_initial_points,
    n_jobs=-1,
)

print("bayesian_optimization_search.fit(X, y)")
bayesian_optimization_search.fit(X, y)

plt.plot(list(cum(max, mean_validation_scores(random_search))), '-ob')
plt.plot(list(mean_validation_scores(random_search)), '--ob')
plt.plot(list(cum(max, mean_validation_scores(bayesian_optimization_search))), '-or')
plt.plot(list(mean_validation_scores(bayesian_optimization_search)), '--or')
plt.show()
