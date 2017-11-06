import numpy as np
from itertools import product
from Param_point import Param_point

class Pmf(object):
    """
    Class that stores a PMF capable of nested sampling / "adaptive mesh refinement".

    Stores probabilities in a list of Param_point objects which associate points in parameter space with probabilities.

    A class for a specific case (e.g. JVTi with PC1D modeling, potentially TIDLS with SRH modeling, etc.) should inherit from this and include methods for calling the relevant model, computing likelihoods, etc.\

    TODO:
    * make helper fcns for spacing, etc. based on log/linear to tidy up code and punt if statements to inside of another fcn

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
        self.logspacing = {param_names[i]:log_spacing[i] for i in range(len(param_names))}

        # make lists of values of each param
        param_vals=[]
        dim_lengths={param_names[i]:dim_lengths[i] for i in range(len(param_names))}
        self.param_spacing={} # difference if linear, quotient if log
        param_edges={}
        for param in self.params:
            param_range = self.param_ranges[param]
            if self.logspacing[param]:
                edges = np.geomspace(param_range[0], param_range[1], num=dim_lengths[param]+1)
                vals = [np.sqrt(edges[j]*edges[j+1]) for j in range(dim_lengths[param])]
                self.param_spacing[param] = vals[1]/vals[0]
            elif not self.logspacing[param]:
                edges = np.linspace(param_range[0], param_range[1], num=dim_lengths[param]+1)
                vals = [0.5*(edges[j]+edges[j+1]) for j in range(dim_lengths[param])]
                self.param_spacing[param] = vals[1]-vals[0]
            param_vals.append(vals)
            param_edges[param] = edges

        # get list of points and make dicts
        points = product(*param_vals)
        point_dicts = [{self.params[i]:point[i] for i in range(len(self.params))} for point in points]

        # initialize the Param_point objects
        self.probs=[]
        init_prob = 1./len(point_dicts)
        for point_dict in point_dicts:
            param_bounds={}
            for param in self.params:
                spacing = self.param_spacing[param]
                if self.logspacing[param]:
                    param_bounds[param]=(point_dict[param]/np.sqrt(spacing),point_dict[param]*np.sqrt(spacing))
                elif not self.logspacing[param]:
                    param_bounds[param]=(point_dict[param]-spacing/2.,point_dict[param]+spacing/2.)
            point=Param_point(point_dict, param_bounds, init_prob)
            self.probs.append(point)

    def normalize(self):

        """
        Normalize overall PMF.

        Should really figure out how to do the overloading properly in Param_point to make this more elegant eventually.
        """

        norm_const = sum([float(prob) for prob in self.probs])
        for prob in self.probs:
            prob.prob = prob.prob/norm_const

    def subdivide(self, threshold_prob):

        """
        Subdivide all boxes with P > threshold_prob and assign "locally uniform" probabilities within each box.

        For now, just divides into two along each direction. Ideas for improvement:
        * divide proportional to probability mass in that box such that minimum prob is roughly equal to maximum prob of undivided boxes
        * user-specified divisions along dimensions
        """

        num_divs = {param:2 for param in self.params} #dummy for now

        to_subdivide = [prob for prob in test.probs if prob.prob>threshold_prob]

        for box in to_subdivide:
            # compute new parameter values and ranges
            centers={}
            bounds={}
            for param in self.params:
                if self.logspacing[param]:
                    edges = np.geomspace(box.param_bounds[param][0], box.param_bounds[param][1], num=num_divs[param]+1)
                    bounds[param] = [(edges[i],edges[i+1]) for i in range(num_divs[param])]
                    centers[param] = [np.sqrt(bounds[param][0]*bounds[param[1]]) for i in range(num_divs[param])]
                elif not self.logspacing[param]:
                    edges = np.linspace(box.param_bounds[param][0], box.param_bounds[param][1], num=num_divs[param]+1)
                    bounds[param] = [(edges[i],edges[i+1]) for i in range(num_divs[param])]
                    centers[param] = [0.5*(bounds[param][0]+bounds[param[1]]) for i in range(num_divs[param])]
            # create new points
            num_boxes = np.product(num_divs.values())
            for i in range(num_boxes):
                params = {param:centers[param][i] for param in self.params}
                param_bounds = {param:bounds[param][i] for param in self.params}
                self.probs.append(Param_point(params, param_bounds, box.prob/num_boxes))
            # remove old one
            self.probs.remove(box)

        # should be normalized already, but just in case:
        self.normalize()

    def multiply(self, other_pmf):

        """
        Compute and store renormalized product of this Pmf with other_pmf.

        TODO:
            * maybe this should return a new Pmf object instead? TBD
        """

        # check for silliness
        assert isinstance(other_pmf, Pmf), "You didn't feed in a Pmf object!"
        # check that probs exist at same points

        # do things

    def most_probable(self, n):
        '''
        Returns the n largest probabilities and the associated parameter values.
        '''

        # do things
