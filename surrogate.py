# -*- coding: utf-8 -*-

__all__ = [
    "Prior", "NormalPrior",
    "SurrogateModel", "Surrogate",
]

import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import minimize

import george
from george import kernels


class Prior(object):

    def sample(self, size=1):
        raise NotImplementedError()

    def evaluate(self, theta):
        raise NotImplementedError()


class NormalPrior(Prior):

    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev
        self.ivar = 1.0 / stddev ** 2

    def sample(self, size=1):
        return self.mean + self.stddev * np.random.randn(size, len(self.mean))

    def evaluate(self, theta):
        return -0.5 * np.sum((theta - self.mean)**2 * self.ivar, axis=-1)


class SurrogateModel(object):

    def simulate(self, theta):
        raise NotImplementedError()

    def get_stats(self, x):
        raise NotImplementedError()

    def _get_simulated_stats(self, theta):
        return self.get_stats(self.simulate(theta))


class Surrogate(object):

    def __init__(self, prior, model, data, kernel=None):
        self.prior = prior
        self.model = model
        self.data_stats = self.model.get_stats(data)

        if kernel is None:
            kernel = kernels.Matern32Kernel(1.0)
        self.base_kernel = kernel

    def initialize(self, num=1000, verbose=True):
        self.training_thetas = self.prior.sample(size=num)
        self.training_stats = np.array([
            self.model._get_simulated_stats(t) for t in self.training_thetas
        ])
        self._initialize_gps(verbose=True, optimize=True)

    def _initialize_gps(self, **kwargs):
        thetas = self.training_thetas
        stats = self.training_stats
        metric = np.var(thetas, axis=0)
        self.gps = []
        for i in range(stats.shape[1]):
            y = stats[:, i]
            amp = np.var(y)
            sigma2 = 1e-8 * np.sqrt(np.median((y - np.median(y))**2))
            kernel = amp * kernels.Matern32Kernel(metric,
                                                  ndim=thetas.shape[1])
            gp = george.GP(kernel, white_noise=sigma2, fit_white_noise=True,
                           mean=np.mean(y), fit_mean=True)
            self.gps.append((gp, 0.0))
        self.update_gps(**kwargs)

    def update_gps(self, verbose=False, optimize=False):
        thetas = self.training_thetas
        stats = self.training_stats
        for i in range(stats.shape[1]):
            y = stats[:, i]
            gp = self.gps[i][0]
            gp.compute(thetas)

            if optimize:
                bounds = gp.get_bounds()
                bounds[1] = (-8, None)
                result = minimize(gp.nll, gp.get_vector(), jac=gp.grad_nll,
                                  args=(y, ), method="L-BFGS-B", bounds=bounds)
                gp.set_vector(result.x)
                if verbose:
                    print(result)

            self.gps[i] = (gp, float(np.exp(gp.white_noise.get_vector())))

    def mh_step(self, theta, error_tol=0.1, M=1000, stepsize=0.1):
        while True:
            theta_prime = theta + stepsize*np.random.randn(len(theta))
            alpha = self.prior.evaluate(theta_prime) + np.zeros(M)
            alpha -= self.prior.evaluate(theta)
            X = np.vstack((theta, theta_prime))
            for i, (gp, sigma2) in enumerate(self.gps):
                y = self.training_stats[:, i]
                mu = gp.sample_conditional(y, X, size=M)
                norm = -0.5 * (mu - self.data_stats[i])**2 / sigma2
                alpha += np.diff(norm, axis=1)[:, 0]

            tau = np.median(alpha)

            u = np.log(np.random.rand(1000))
            p_minus = np.mean(alpha[None, :] <= u[:, None], axis=1)
            p_plus = 1.0 - p_minus
            error = np.mean((u <= tau) * p_minus + (u > tau) * p_plus)
            print(error)
            if error > error_tol:
                sim_stats = self.model._get_simulated_stats(theta_prime)
                self.training_thetas = np.concatenate((
                    self.training_thetas, theta_prime[None, :]), axis=0)
                self.training_stats = np.concatenate((
                    self.training_stats, sim_stats[None, :]), axis=0)
                self.update_gps(optimize=False)
            else:
                if np.log(np.random.rand()) < tau:
                    return theta_prime
                return theta

    def plot_gps(self, xi=0, yi=1):
        figs = []
        for i, (gp, sig2) in enumerate(self.gps):
            x = self.training_thetas[:, xi]
            y = self.training_thetas[:, yi]
            z = self.training_stats[:, i]

            mu = gp.predict(z, self.training_thetas, return_cov=False)

            vmin = z.min()
            vmax = z.max()

            fig, axes = pl.subplots(1, 2)
            axes[0].scatter(x, y, c=z, edgecolor="none", vmin=vmin, vmax=vmax)
            axes[1].scatter(x, y, c=mu, edgecolor="none", vmin=vmin, vmax=vmax)

            figs.append(fig)
        return figs


if __name__ == "__main__":
    class Model(SurrogateModel):

        def __init__(self, dim):
            self.dim = dim

        def simulate(self, theta):
            mu, lns = theta
            return mu + np.exp(lns) * np.random.randn(self.dim)

        def get_stats(self, x):
            mu, lns = np.mean(x), np.log(np.std(x))
            return np.array([mu * lns, mu + lns])

    prior = NormalPrior(np.zeros(2), np.ones(2))
    model = Model(1000)
    data = model.simulate([0.5, -0.1])

    gpsabc = Surrogate(prior, model, data)
    gpsabc.initialize()

    theta = prior.sample()[0]
    chain = np.empty((10, len(theta)))
    for i in range(len(chain)):
        chain[i] = theta = gpsabc.mh_step(theta)

    figs = gpsabc.plot_gps()
    for i, fig in enumerate(figs):
        fig.savefig("initial-pred-{0}.png".format(i))
