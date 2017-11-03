import numpy as np
from itertools import product

class Pmf(object):
    """
    Class that stores a PMF capable of nested sampling / "adaptive mesh refinement".

    Stores probabilities in a list of Param_point objects which associate points in parameter space with probabilities.

    A class for a specific case (e.g. JVTi with PC1D modeling, potentially TIDLS with SRH modeling, etc.) should inherit from this and include methods for calling the relevant model, computing likelihoods, etc.

    QUESTIONS TO THINK ABOUT:
    Where should info about observation conditions be stored? Does it need to be?

    How about simulation results? Can they be stored separately but maintain a
    parallel/analogous data structure? Perhaps just a big index array?
    """

    def __init__(self, param_names, dim_lengths, dim_mins, dim_maxes, log_spacing):

        """
        Instantiate a uniform prior.

        param_names = list of strings
        dim_lengths, dim_mins, dim_maxes = lists of ints
        log_spacing = list of bools
        """

        # check that you haven't fed in anything silly
        l = len(param_names)
        assert len(dim_lengths)==l and len(dim_mins)==l and len(dim_maxes)==l and len(log_spacing)==l, "Lengths of all inputs need to match!"

        assert all([dim_maxes[i]>dim_mins[i] for i in range(len(dim_maxes))]), "Maximum values must be greater than minimum values!"

        # copy in things
        self.params = param_names # order corresponds to order of indices
        self.param_ranges={self.params[i]:[dim_mins[i],dim_maxes[i]] for i in range(len(param_names))}
        self.logspacing = log_spacing

        # make lists of values of each param
        # do a itertools.product on that to get list of points
        # make those into the dictionaries (maybe write a helper method for this?)
        # use those to initialize the Param_point objects

    def normalize(self):

        """
        Normalize overall PMF.
        """

        # do things


    def subdivide(self, threshold_prob):

        """
        Subdivide all boxes with P > threshold_prob and assign "locally uniform" probabilities within each box.
        """

        # do things


    def multiply(self, other_pmf):
        '''
        Compute and store renormalized product of this Pmf with other_pmf.

        TODO:
            * maybe this should return a new Pmf object instead? TBD
        '''

        # check for silliness
        assert isinstance(other_pmf, Pmf), "You didn't feed in a Pmf object!"

        # do things

    def most_probable(self, n):
        '''
        Returns the n largest probabilities and the associated parameter values.
        '''

        # do things
