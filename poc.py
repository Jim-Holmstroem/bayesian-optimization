
# # level 1
# GP | kernel and acqusition function
# visualizations

# # level 2
# multidimensional

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

from sklearn.gaussian_process import (
    GaussianProcessRegressor,
    kernels,
)

from joblib import (
    Parallel,
    delayed,
)


def pmap(f, *xss):
    return Parallel(n_jobs=-1)(
        starmap(delayed(f), *xss)
    )

def pmap_outer(f, *xss):
    return Parallel(n_jobs=-1)(
        starmap(delayed(f), *product(*xss))
    )


def f(x):
    return x * np.sin(x)

def f2d(x):
    """ The rosenbrock function
    """
    return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2


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
        (mu_x,), (sigma_x,) = gp_model.predict(x.reshape((-1, 1)), return_std=True)
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



def negate(f):
    def _f(*args, **kwargs):
        return -f(*args, **kwargs)

    return _f

def silly_f(x):
    (y_pred,), (sigma,) = gp.predict(x.reshape((-1, 1)), return_std=True)

    return y_pred + sigma

def bo(X, y):

    data = list(zip(X, y))

    x = np.atleast_2d(np.linspace(0, 10, 1024)).T

    kernel = kernels.Matern() + kernels.WhiteKernel(noise_level=0.01, noise_level_bounds="fixed")

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=16)
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


    a = a_EI(gp, data)

    # TODO have a reasonable optimization (this doesn't scale well)
    (x_min_,) = brute(
        negate(a),
        ranges=((0, 10),),
        Ns=64,
        finish=fmin,
    )
    # FIXME brute can return numbers outside of the range! X = np.linspace(0, 10, 32), Ns=64, ranges=((0, 10)  (x_min_ = 10.22...)
    # I think it occurs when the function is pretty flat (but not constant)
    # TODO verify that finish function gets the same range as brute and don't wonder off (perhaps this is intended behaviour?)
    # TODO check https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py#L2614 to see if it's possible for x_min to end up outside of the range (and if then when)

    print(x_min_)

    plt.plot(x, y_pred, 'r-')
    plt.plot(X, y, 'x')

    plt.axvline(x_min_)

    plt.plot(
        x,
        list(map(a, x)),
        'g--',
    )


    plt.plot(x, list(map(f, x)), 'm--')
    plt.plot(x, y_pred, 'r-')
    plot_confidences(x, y_pred, sigma)

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
