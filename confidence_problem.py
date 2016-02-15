from __future__ import (
    print_function,
    division,
)

from functools import *
from itertools import *

import numpy as np

from scipy.special import erf
from scipy import sqrt
from scipy.integrate import simps

from matplotlib import pylab as pl

from joblib import (
    Parallel,
    delayed,
)

from sklearn.gaussian_process import (
    GaussianProcessRegressor,
    kernels,
)


def f(x):
    return x * np.sin(x)

def pmap(f, *xss):
    return Parallel(n_jobs=-1)(
        map(delayed(f), *xss)
    )

def pmap_outer(f, *xss):
    return Parallel(n_jobs=-1)(
        starmap(delayed(f), product(*xss))
    )


def plot_confidence(x, y_pred, sigma, confidence=1):
    procent_confidence = erf(confidence/sqrt(2))
    return pl.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred + confidence*sigma, (y_pred - confidence*sigma)[::-1]]),
        alpha=.1,
        fc='b',
        ec='None',
    )


def plot_confidences(x, y_pred, sigma, confidences=range(1, 5)):
    return list(map(
        partial(plot_confidence, x, y_pred, sigma),
        confidences
    ))


def integrated_sigma(alpha, n_samples, n_restarts_optimizer=16, f=f):
    print("integrated_sigma(n_samples={n_samples}, alpha={alpha})".format(
        n_samples=n_samples,
        alpha=alpha,
    ))
    X = np.atleast_2d(
        np.linspace(1, 9, n_samples)
    ).T
    y = f(X).ravel()
    x = np.atleast_2d(np.linspace(0, 10, 16 * 1024)).T

    kernel = kernels.Matern() + (kernels.WhiteKernel(noise_level=alpha) if alpha is not None else 0.0)
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


def main(n_samples=range(4, 64, 1)):
    for alpha in [None, 0.0]:
        pl.plot(
            n_samples,
            pmap(
                partial(integrated_sigma, alpha),
                n_samples,
            ),
            label="$\\alpha$={alpha:.2f}".format(alpha=alpha) if alpha is not None else "$\\alpha$=None",
            alpha=0.8,
        )
    pl.legend(framealpha=0.8)
    pl.show()


if __name__ == "__main__":
    main()
