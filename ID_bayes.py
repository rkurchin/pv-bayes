import ID_pmf as pmf
from copy import deepcopy
from scipy.stats import norm

def likelihood(probs, J_meas, V_meas, T, J_err):
    '''
    Compute Bayesian likelihood, assuming Gaussian error with stdev of J_err.

    For now, I'm "running the model" inside this function. In "real" versions,
    modeled data should probably be an input.

    2D-ness is also hard-coded for now.

    TODO:
        * fix above-mentioned stuff
        * other error models (e.g. exponential for current etc.)
    '''

    lkl = deepcopy(probs)

    '''
    This should end up looking something like:
    for param_vals in active_param_list:
        J = calc_model(param_vals['point'], conds) # the model calc should then take in a dict
        prob = calc_prob(J, conds, param_vals['point']) # or whatever
        probs[inds] = prob / num_boxes
    '''

    for i in range(len(param_vals['n'])):
        for j in range(len(param_vals['J_0'])):
            if lkl._m[i, j] == lkl._M:  # should this point get updated?
                # if so, compute modeled value
                J_model = lkl.compute_ID(V_meas, T, {'n':param_vals['n'][i],'J_0':param_vals['J_0'][j]})
                # and update likelihood
                lkl.probs[i,j] = norm.pdf(J_meas, loc=J_model, scale=J_err)

    lkl.normalize()
    return lkl
