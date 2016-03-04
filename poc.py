from __future__ import (
    division,
    print_function,
)

# # level 1 [check]
# GP | kernel and acqusition function
# visualizations

# # level 2 [check]
# multidimensional

# # level A
# integrate with sklearn's model selection

# # level 3
# cost

# # level 4
# integrated acqusition function

# # level 5
# monte carlo acqusition for parallelizing bayesian optimization


# # Considerations
# noice in overvations


# # Issues
# WhiteKernel is trained to be kinda restrictive (however it's perhaps what we want?)
# if you start sampling more points the whitekernel gets more and more irrelevant (Am I using it the right way?)


# # Simple Comparison Test
# SearchCV(ElasticNet(), [{'l1_ratio': np.linspace(0, 1, 64)}]


# # Idea
# Mix informativity (or whatever it's called, the thing in probabilistic numerics on where you should choose the first set off points when for examples integrating (which is evenly spread out))
# with trying to find a minimum
# enforcing some informativity will basically make it lean about towards exploration (is this true?)
#


from warnings import warn
from functools import *
from itertools import *
from operator import *

import numpy as np

from scipy.special import (
    erf,
)

from scipy.optimize import (
    fmin,
    minimize,
    brent,
    brute,
)
from scipy.stats import norm
from scipy import sqrt
from scipy.integrate import simps

from matplotlib import pylab as plt


from sklearn.model_selection import _search, _split, _validation  # TEMP temporary, still will be dealt with properly later when integrated into sklearn

from sklearn.preprocessing import normalize
from sklearn.base import is_classifier, clone, BaseEstimator
from sklearn.metrics.scorer import check_scoring
from sklearn.cross_validation import _fit_and_score
from sklearn.utils.validation import _num_samples, indexable


from sklearn.gaussian_process import (
    GaussianProcessRegressor,
    kernels,
)

from joblib import (
    Parallel,
    delayed,
)


def mean_mean_validation_scores(results):
    return np.mean(list(map(
        itemgetter(0),
        results
    )))


def co(f, *gs):
    if gs:
        return lambda *x: f(co(*gs)(*x))
    else:
        return f

# TODO Should the ranges for the grid be represented by something smarter then lists?

# TODO which of this is needed? is it needed to have this anywhere at all? is it accessed by anything outside of the Gridsearch itself??
# class ParameterGridlike(object):
#   __iter__: current implementations are independent of prior evaluations, but GaussianOptimization is not
#   __len__: easy for n_evaluations case but impossible for time-constraints
#   __getitem__: (optional) only exists in ParameterGrid, NOTE ParameterSampler uses a ParameterGrid in the background and it's __getitem__

# TODO time-constraints (also includes the other model_selectors?)
# TODO how to handle log-scales

# TODO could the std in grid_scores_ be used for "input noice" like WhiteKernel  (it's not really the same thing right?)

# FIXME without CV as well
# TODO maximum on param_grid (since param_grid isn't real but rasterized)
class BayesianOptimizationSearchCV(_search.BaseSearchCV):
    def __init__(self, estimator, param_grid, n_iter, n_initial_points, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):
        #assert(n_jobs == 1)
        super(BayesianOptimizationSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score)
        self.param_grid = param_grid
        assert(n_iter >= 0)
        self.n_iter = n_iter
        assert(n_initial_points > 0)
        self.n_initial_points = n_initial_points
        _search._check_param_grid(param_grid)


    def _fit(self, X, y, labels, parameter_iterable):
        raise NotImplementedError()  # too catch accidental use

    def fit(self, X, y=None, labels=None):
        #return self._fit(
        #    X, y, labels,
        #    parameter_iterable # parameter_iterable, \in Sized, it actually does len(parameter_iterable) in _fit
        #)

        # FIXME code duplication from BaseSearchCV._fit
        estimator = self.estimator
        cv = _split.check_cv(self.cv, y, classifier=is_classifier(estimator))
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        n_samples = _num_samples(X)
        X, y, labels = indexable(X, y, labels)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                  'of samples (%i) than data (X: %i samples)'
                                  % (len(y), n_samples))

        n_splits = cv.get_n_splits(X, y, labels)

        if self.verbose > 0 and isinstance(parameter_iterable, Sized):
            n_candidates = len(parameter_iterable)
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))

        base_estimator = clone(self.estimator)

        pre_dispatch = self.pre_dispatch
        # FIXME how to handle pre_dispatch


        # FIXME recursively getting new parameters to evaluate

#        parameter_iterable = ...  # the magic
#
#        # The evaluation (Parallel) stuff
#        out = Parallel(
#            n_jobs=self.n_jobs, verbose=self.verbose,
#            pre_dispatch=pre_dispatch
#        )(delayed(_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
#                                  train, test, self.verbose, parameters,
#                                  self.fit_params, return_parameters=True,
#                                  error_score=self.error_score)
#            for parameters in parameter_iterable
#            for train, test in cv.split(X, y, labels))
#

        # n_fits on each (train, test)
        def cross_validation(raw_parameters):
            parameters = dict(zip(
                self.param_grid.keys(), raw_parameters
            ))  # TODO more robust way of doing this
            print(parameters)

            return Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose,
                pre_dispatch=pre_dispatch
            )(delayed(_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
                                      train, test, self.verbose, parameters,
                                      self.fit_params, return_parameters=True,
                                      error_score=self.error_score)
               for train, test in cv.split(X, y, labels))

        x = cartesian_product(*self.param_grid.values())

        # FIXME implement as non-recursive
        def bo_(x_obs, y_obs, n_iter):
            if n_iter > 0:
                kernel = kernels.Matern() + kernels.WhiteKernel()
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=16)
                gp.fit(x_obs, 1-y_obs)

                a = a_EI(gp, x_obs=x_obs, y_obs=1-y_obs)

                argmax_f_x_ = x[np.argmax(a(x))]

                # heavy evaluation
                f_argmax_f_x_ = cross_validation(argmax_f_x_)

                y_ob = np.atleast_2d(mean_mean_validation_scores(f_argmax_f_x_)).T

                return f_argmax_f_x_ + bo_(
                    x_obs=np.vstack((x_obs, argmax_f_x_)),
                    y_obs=np.vstack((y_obs, y_ob)),
                    n_iter=n_iter-1,
                )

            else:
                return []


        # FIXME (most informative) decision like Numerical Probabilistics stuff for integrations
        # sobol initilization?

        sampled_x_ind = np.random.choice(
            x.shape[0],
            size=self.n_initial_points,
            replace=False,
        )
        print(sampled_x_ind)

        x_obs = x[sampled_x_ind]
        f_x_obs = list(map(cross_validation, x_obs))

        y_obs = np.atleast_2d(list(map(mean_mean_validation_scores, f_x_obs))).T

        out = sum(f_x_obs, []) + bo_(x_obs, y_obs, n_iter=self.n_iter)

        n_fits = len(out)

        scores = list()
        grid_scores = list()
        for grid_start in range(0, n_fits, n_splits):
            n_test_samples = 0
            score = 0
            all_scores = []
            for this_score, this_n_test_samples, _ , parameters in \
                    out[grid_start:grid_start + n_splits]:
                all_scores.append(this_score)
                if self.iid:
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                score += this_score
            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_splits)
            scores.append((score, parameters))

            grid_scores.append(_search._CVScoreTuple(
                parameters,
                score,
                np.array(all_scores)))

        self.grid_scores_ = grid_scores

        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]

        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        if self.refit:
            best_estimator = clone(base_estimator).set_params(
                **best.parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator

        return self


def negate(f):
    def _f(*args, **kwargs):
        return -f(*args, **kwargs)

    return _f


def cartesian_product(*arrays):
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
    # FIXME needs to handle different datatypes (this is probably a much more involved problem to solve then what can be solved inside cartesian_product)
    out = np.empty(rows * cols, dtype=np.int) # np.empty(rows * cols, dtype=broadcasted[0].dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows

    return out.reshape(cols, rows).T


def pmap(f, *xss):
    return Parallel(n_jobs=-1)(
        starmap(delayed(f), *xss)
    )

def pmap_outer(f, *xss):
    return Parallel(n_jobs=-1)(
        starmap(delayed(f), *product(*xss))
    )


def f(x, crunch=True):
    min_value = -5  # crunch it to fit withing [0, 1] for the range [0, 10]
    max_value = 8

    return (x * np.sin(x) - min_value) / (max_value - min_value) if crunch else x * np.sin(x)


def f2d(x):
    xa, xb = x[:,0], x[:,1]
    min_value = -60
    max_value = 80
    return ((xa - 3 + xb/3) * (xb + 1) * np.sin(xb) - min_value) / (max_value - min_value)
    #return 0.5 * (1 - x)**2 + (x_ - x**2)**2


def a_PI(gp_model, x_obs, y_obs, theta=0.01):
    assert(theta >= 0.0)
    fx_min = y_obs.min()

    def a_PI_given(x):
        mu_x, sigma_x = gp_model.predict(x, return_std=True)
        sigma_x = np.atleast_2d(sigma_x).T
        dx = (fx_min - mu_x) - theta
        Z = dx / sigma_x

        return norm.cdf(Z)

    return a_PI_given


def a_EI(gp_model, x_obs, y_obs, theta=0.01):
    # NOTE a_EI is trying to find a minimum value of f
    # theta = 0.01
    # http://arxiv.org/pdf/1012.2599v1.pdf
    # TODO theta (scaled by signal variance if necessary)

    assert(theta >= 0.0)

    # TODO handle no observations case  (fx_min = 1?)
    fx_min = y_obs.min()


    def a_EI_given(x):
        """x : array-like, shape = [n_observations, n_features]
        """
        mu_x, sigma_x = gp_model.predict(x, return_std=True)
        sigma_x = np.atleast_2d(sigma_x).T
        dx = (fx_min - mu_x) - theta
        Z = dx / sigma_x

        return np.maximum(
            dx * norm.cdf(Z) + sigma_x * norm.pdf(Z),
            0
        )
        # FIXME why do we cut out all negative a?

    return a_EI_given


# FIXME performs really badly
def a_LCB(gp_model, x_obs, y_obs, kappa=1.0):
    """
    kappa could partially be estimated, see [Srinivas et al., 2010]
    """
    def a_LCB_given(x):
        mu_x, sigma_x = gp_model.predict(x, return_std=True)

        return -(mu_x - kappa * sigma_x)  # FIXME fix this properly, the minus is a "hack" (or show that it's not a hack)

    return a_LCB_given


def plot_confidence(x, y_mean, y_sigma, confidence=1):
    procent_confidence = erf(confidence/sqrt(2))

    return plt.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([y_mean + confidence * y_sigma, (y_mean - confidence * y_sigma)[::-1]]),
        alpha=.1,
        fc='b',
        ec='None',
    )


def plot_confidences(x, y_pred, sigma, confidences=range(1, 3)):
    return list(map(
        partial(plot_confidence, x, y_pred, sigma),
        confidences
    ))


def plot(model, x_obs, y_obs, argmin_a_x, a, x):
    y_mean, y_sigma = model.predict(x, return_std=True)
    y_sigma = np.atleast_2d(y_sigma).T
    a_x = a(x)
    plt.plot(x, y_mean, 'r-')
    plt.plot(x_obs, y_obs, 'o')

    plt.axvline(argmin_a_x)

    plt.plot(
        x,
        a_x*(1/np.max(a_x)),
        'g--',
    )

    plt.plot(x, f(x), 'm--')
    plot_confidences(x, y_mean, y_sigma, confidences=[1])


def plot_2d(gp_model, x_obs, y_obs, argmin_a_x, a, xs):
    x, x_ = xs
    X, X_ = np.meshgrid(*xs)
    Y_true = f2d(cartesian_product(*xs)).reshape((xs[-1].shape[0],-1))

    a_x = a(cartesian_product(*xs)).reshape((xs[-1].shape[0],-1))

    mu_x, sigma_x = gp_model.predict(cartesian_product(*xs), return_std=True)
    mu_x = mu_x.reshape((xs[-1].shape[0],-1))
    sigma_x = sigma_x.reshape((xs[-1].shape[0],-1))

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    ax1.set_title("$f$")
    ax1.pcolormesh(X, X_, Y_true, cmap='viridis')
    #ax1.contour(X, X_, Y_true, [0.1,0.5,0.9])

    ax2.set_title("$a$")
    ax2.pcolormesh(X, X_, a_x, cmap='viridis')

    ax3.set_title("$\\mu$")
    ax3.pcolormesh(X, X_, mu_x, cmap='viridis')

    ax4.set_title("$\\sigma$")
    ax4.pcolormesh(X, X_, sigma_x, cmap='viridis')

    def plot_observations_and_query_on(ax):
        ax.plot(
            x_obs[:,0],
            x_obs[:,1],
            'ro'
        )
        ax.axvline(argmin_a_x[0])
        ax.axhline(argmin_a_x[1])

    list(map(plot_observations_and_query_on, [ax1, ax2, ax3, ax4]))



def bo(X, y):

    data = list(zip(X, y))

    x = np.atleast_2d(np.linspace(0, 10, 1024)).T
    x_= np.atleast_2d(np.linspace(0, 10, 1024)).T


    kernel = kernels.Matern() + kernels.WhiteKernel()

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=16, )#normalize_y=True)

    gp.fit(X, y)
    # FIXME is it possible for mu(x) < min{x \in observed_x}?
    # is this due to that GaussainProcess's prior states that mu(x) = 0?
    # will this effect the performance of GO, since everything not observed will automatically give an additional boost since the prior plays a bigger role (look it up) [we know that the loss we in the end are optimizing is \in [0, 1]
    y_pred, sigma = gp.predict(x, return_std=True)


    #http://www.scipy-lectures.org/advanced/mathematical_optimization/

    # x_min = fmin(negate(silly_f), 5)  # TODO better maximizer
    # Strong points: it is robust to noise, as it does not rely on computing gradients. Thus it can work on functions that are not locally smooth such as experimental data points, as long as they display a large-scale bell-shape behavior. However it is slower than gradient-based methods on smooth, non-noisy functions.


    #opt_result = minimize(negate(silly_f), 5, bounds=[(0, 10)])  # TODO better maximizer
    #print(opt_result)
    #assert(opt_result.success)


    #x_min = opt_result.x


    # x_min = brent(negate(silly_f), brack=(0, 10))  # NOTE 1D only, NOTE not guaranteed to be within range brack=(0, 10) (see documentation)

    # TODO getting the gradient the gaussian would unlock all gradient based optimization methods!! (including L_BFGS)


    a = a_EI(gp, x_obs=X, y_obs=y, theta=0.01)
    a_x = np.apply_along_axis(a, 1, x)

    (x_min_,) = max(x, key=a)

    # TODO have a reasonable optimization (this doesn't scale well)
    #(x_min_,) = brute(
    #    negate(a),
    #    ranges=((0, 10),),
    #    Ns=64,
    #    finish=fmin,
    #)
    # FIXME brute can return numbers outside of the range! X = np.linspace(0, 10, 32), Ns=64, ranges=((0, 10)  (x_min_ = 10.22...)
    # I think it occurs when the function is pretty flat (but not constant)
    # TODO verify that finish function gets the same range as brute and don't wonder off (perhaps this is intended behaviour?)
    # TODO check https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py#L2614 to see if it's possible for x_min to end up outside of the range (and if then when)

    print(x_min_)


    #plot_2d(x=x, x_=x_, y_pred=y_pred, sigma = sigma, a_x=a_x)
    #plot(x=x, y_pred=y_pred, x_obs=X, y_obs=y, x_min_=x_min_, sigma=sigma, a_x=a_x)
    #plt.show()

    # evaluate
    fx_min_ = f(x_min_)
    bo(
        X=np.vstack(
            (X,[x_min_,])
        ),
        y=np.hstack(
            (y,[fx_min_,])
        ),
    )


#if __name__ == "__main__":
#    X = np.atleast_2d(
#        [ 1, 2, 3.5]  # FIXME really sensative to input
#    ).T
#    y = f(X)#.ravel()
#    print(X.shape, y.shape,)
#
#    bo(X, y)


def bo_(x_obs, y_obs):
    kernel = kernels.Matern() + kernels.WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=16)
    gp.fit(x_obs, y_obs)

    xs = list(repeat(np.atleast_2d(np.linspace(0, 10, 128)).T, 2))
    x = cartesian_product(*xs)

    a = a_EI(gp, x_obs=x_obs, y_obs=y_obs)

    argmin_a_x = x[np.argmax(a(x))]

    # heavy evaluation
    print("f({})".format(argmin_a_x))
    f_argmin_a_x = f2d(np.atleast_2d(argmin_a_x))


    plot_2d(gp, x_obs, y_obs, argmin_a_x, a, xs)
    plt.show()


    bo_(
        x_obs=np.vstack((x_obs, argmin_a_x)),
        y_obs=np.hstack((y_obs, f_argmin_a_x)),
    )


if __name__== "__main__":
    x_obs = np.array(
        [
            [ 1, 1],
            [ 2, 7],
            [ 3.5, 1],
        ]
    )
    y_obs = f2d(x_obs)  # FIXME shouldl y_obs have same dimensions as x_obs?

    bo_(x_obs, y_obs)
