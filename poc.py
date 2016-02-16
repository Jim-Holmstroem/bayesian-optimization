
# # level 1
# GP | kernel and acqusition function
# visualizations

# # level 2
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
from sklearn.utils.validation import _num_samples, indexable

from sklearn.gaussian_process import (
    GaussianProcessRegressor,
    kernels,
)

from joblib import (
    Parallel,
    delayed,
)


# TODO which of this is needed? is it needed to have this anywhere at all? is it accessed by anything outside of the Gridsearch itself??
# class ParameterGridlike(object):
#   __iter__: current implementations are independent of prior evaluations, but GaussianOptimization is not
#   __len__: easy for n_evaluations case but impossible for time-constraints
#   __getitem__: (optional) only exists in ParameterGrid, NOTE ParameterSampler uses a ParameterGrid in the background and it's __getitem__

# TODO time-constraints (also includes the other model_selectors?)

# FIXME without CV as well
# TODO maximum on param_grid (since param_grid isn't real but rasterized)
class GaussianOptimizationCV(_search.BaseSearchCV):
    def __init__(self, estimator, param_grid, n_iter, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):
        assert(n_jobs == 1)
        super(GaussianOptimizationCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score)
        self.param_grid = param_grid
        self.n_iter = n_iter
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
        cv = _split.check_cv(self.cv, y, classifiers=is_classifier(estimator))
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

def f2d(x, x_):
    min_value = -60
    max_value = 80
    return ((x - 3 + x_/3) * (x_ + 1) * np.sin(x_) - min_value) / (max_value - min_value)
    #return 0.5 * (1 - x)**2 + (x_ - x**2)**2


def a_PI(data, gp_model):
    raise NotImplementedError()

    def a_PI_given(x):
        pass

    return a_PI_given


# TODO this only supports 1D
def a_EI(gp_model, data, theta=0.01):
    # NOTE a_EI is trying to find a minimum value of f
    # theta = 0.01
    # http://arxiv.org/pdf/1012.2599v1.pdf
    # TODO theta (scaled by signal variance if necessary)

    assert(theta >= 0.0)
    assert(len(data) > 0)

    x_min, fx_min = min(data, key=itemgetter(1))


    def a_EI_given(x):
        """x : array-like, shape = [1, n_features] or [n_features]
        """
        (mu_x,), (sigma_x,) = gp_model.predict(x.reshape((1, -1)) , return_std=True)
        dx = (fx_min - mu_x) - theta
        Z = dx / sigma_x

        return dx * norm.cdf(Z) + sigma_x * norm.pdf(Z) if sigma_x > 0 else 0

    return a_EI_given


def plot_confidence(x, y_pred, sigma, confidence=1):
    procent_confidence = erf(confidence/sqrt(2))
    return plt.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred + confidence*sigma, (y_pred - confidence*sigma)[::-1]]),
        alpha=.1,
        fc='b',
        ec='None',
    )


def plot_confidences(x, y_pred, sigma, confidences=range(1, 3)):
    return list(map(
        partial(plot_confidence, x, y_pred, sigma),
        confidences
    ))


def integrated_sigma(n_samples, alpha=1.0, n_restarts_optimizer=16, f=f):
    print("integrated_sigma(n_samples={n_samples}, alpha={alpha})".format(
        n_samples=n_samples,
        alpha=alpha,
    ))
    X = np.atleast_2d(
        np.linspace(1, 9, n_samples)
    ).T
    y = f(X).ravel()
    x = np.atleast_2d(np.linspace(0, 10, 16 * 1024)).T

    kernel = kernels.Matern() + kernels.WhiteKernel(noise_level=alphak, noise_level_bounds='fixed')
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts_optimizer,
    )
    gp.fit(X, y)

    y_pred, sigma = gp.predict(x, return_std=True)

    return simps(
        x=x.ravel(),
        y=sigma,
    )


def plot_integrated_sigma(n_samples=range(4, 64, 1)):
    for alpha in np.linspace(0, 2, 16):
        pl.plot(
            n_samples,
            pmap_outer(
                integrated_sigma,
                n_samples,
                [alpha, ],
            ),
            label="$\\alpha$={alpha:.2f}".format(alpha=alpha),
            alpha=0.8,
        )
    pl.legend(framealpha=0.8)
    pl.show()


def plot_2d(x, x_, y_pred, sigma, a_x):
    X, X_ = np.meshgrid(x, x_)
    Y = f2d(X, X_)
    plt.pcolormesh(X, X_, Y, cmap='viridis')
    plt.colorbar()
    plt.contour(X, X_, Y, [0.1,0.5,0.9])



def plot(x, y_pred, x_min_, sigma, a_x):
    plt.plot(x, y_pred, 'r-')
    plt.plot(X, y, 'x')

    plt.axvline(x_min_)

    plt.plot(
        x,
        a_x*(1/np.max(a_x)),
        'g--',
    )


    plt.plot(x, list(map(f, x)), 'm--')
    plt.plot(x, y_pred, 'r-')
    plot_confidences(x, y_pred, sigma, confidences=[1])


def negate(f):
    def _f(*args, **kwargs):
        return -f(*args, **kwargs)

    return _f


def bo(X, y):

    data = list(zip(X, y))

    x = np.atleast_2d(np.linspace(0, 10, 1024)).T
    x_= np.atleast_2d(np.linspace(0, 10, 1024)).T


    kernel = kernels.Matern() + kernels.WhiteKernel()

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=16, )#normalize_y=True)
    # NOTE kernel hyperparams is optimized (on a default kernel RBF)

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


    a = a_EI(gp, data, theta=0.01)
    a_x = np.array(list(map(a, x)), ndmin=2).T

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


    plot_2d(x=x, x_=x_, y_pred=y_pred, sigma = sigma, a_x=a_x)
    #plot(x=x, y_pred=y_pred, x_min_=x_min_, sigma=sigma, a_x=a_x)

    plt.show()

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


if __name__ == "__main__":
    X = np.atleast_2d(
        [1,2,3.5]  # FIXME really sensative to input
    ).T
    y = f(X).ravel()

    bo(X, y)
