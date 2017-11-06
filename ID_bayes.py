import ID_pmf as pmf
from copy import deepcopy
from scipy.stats import norm

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
        * allow feeding in a list of observations
        * make a Bayes class that this can inherit from
    '''

    assert len(J_meas)==len(V_meas)==len(T)==len(J_err), "Lengths of observed quantities must match!"

    lkl = deepcopy(probs)

    for i in range(len(J_meas)):
        for point in lkl.points:
            J_model = lkl.compute_ID(V_meas[i], T[i], point.params)
            point.prob = norm.pdf(J_meas, loc=J_model, scale=J_err)

    lkl.normalize()
    return lkl
