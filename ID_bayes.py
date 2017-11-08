import ID_pmf as pmf
from copy import deepcopy
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import numpy as np

def likelihood(probs, J_meas, V_meas, T, J_err):
    '''
    Compute Bayesian likelihood, assuming Gaussian error with stdev of J_err.

    probs is a Pmf object of some flavor
    rest of inputs are lists of equal length

    For now, I'm "running the model" inside this function. In "real" versions,
    modeled data should probably be an input.

    2D-ness is also hard-coded for now.

    TODO:
        * fix above-mentioned stuff
        * other error models (e.g. exponential for current etc.)
        * make a Bayes class that this can inherit from
        * allow to feed in list of observations
    '''

    lkl = deepcopy(probs)

    for point in lkl.points:
        J_model = lkl.compute_ID(V_meas, T, point.params)
        point.prob = norm.pdf(J_meas, loc=J_model, scale=J_err)

    lkl.normalize()
    return lkl

def resample_probs(pmf, num_samples):

    """
    Because the seaborn functions don't support weighting but make really pretty plots, I wrote this function to resample so that points in parameter space are just duplicated a number of times proportional to their probability. It generates these points uniformly spaced across the range for aesthetic purposes, and returns a DataFrame.

    TODO:
    * write a more general version of this - maybe even just move it to the Pmf class
    """

    n_name = "ideality factor: $n$"
    J_0_name = "log saturation current: $\log(J_0)$"
    n = []
    J_0 = []
    for point in pmf.points:
        num_samples_here = int(round(num_samples*point.prob,0))
        n_bounds = point.param_bounds['n']
        J_0_bounds = point.param_bounds['J_0']
        n.extend(np.random.uniform(n_bounds[0],n_bounds[1],num_samples_here))
        J_0.extend(np.random.uniform(np.log10(J_0_bounds[0]),np.log10(J_0_bounds[1]),num_samples_here))
    return pd.DataFrame(data={n_name:n, J_0_name:J_0})

def visualize_probs(pmf, type):
    """
    Plot joint distribution of n and J_0. Type should either be "hex" or "kde"

    TODO:
    * write a more general version of this that can check for itself when things are logarithmic and handle them appropriately with labels and such
    * Add anotation with entropy
    * maybe just rewrite calling JointGrid directly if need be
    """
    n_name = "ideality factor: $n$"
    J_0_name = "log saturation current: $\log(J_0)$"
    sns.set(style="white")
    samples = 10000
    min_prob = 0.0001
    df = resample_probs(pmf,samples)
    xvals = [np.log10(p.params['J_0']) for p in pmf.points if p.prob>min_prob]
    yvals = [p.params['n'] for p in pmf.points if p.prob>min_prob]
    sns.jointplot(x=J_0_name, y=n_name, data=df, kind=type, xlim=(min(xvals),max(xvals)), ylim=(min(yvals),max(yvals)), stat_func=None)
