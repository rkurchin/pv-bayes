import numpy as np
from itertools import product
from Param_point import Param_point
from copy import deepcopy

class Pmf(object):
    """
    Class that stores a PMF capable of nested sampling / "adaptive mesh refinement".

    Stores probabilities in a list of Param_point objects which associate points in parameter space with probabilities.

    A class for a specific case (e.g. JVTi with PC1D modeling, potentially TIDLS with SRH modeling, etc.) should inherit from this and include methods for calling the relevant model, computing likelihoods, etc.\

    TODO:
    * make helper fcns for spacing, etc. based on log/linear to tidy up code and punt if statements to inside of another fcn
    * more intuitive name for probs and Param_point?

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
        param_spacing={} # difference if linear, quotient if log
        param_edges={}
        for param in self.params:
            param_range = self.param_ranges[param]
            if self.logspacing[param]:
                edges = np.geomspace(param_range[0], param_range[1], num=dim_lengths[param]+1)
                vals = [np.sqrt(edges[j]*edges[j+1]) for j in range(dim_lengths[param])]
                param_spacing[param] = vals[1]/vals[0]
            elif not self.logspacing[param]:
                edges = np.linspace(param_range[0], param_range[1], num=dim_lengths[param]+1)
                vals = [0.5*(edges[j]+edges[j+1]) for j in range(dim_lengths[param])]
                param_spacing[param] = vals[1]-vals[0]
            param_vals.append(vals)
            param_edges[param] = edges

        # get list of points and make dicts
        points = product(*param_vals)
        point_dicts = [{self.params[i]:point[i] for i in range(len(self.params))} for point in points]

        # initialize the Param_point objects
        self.points=[]
        init_prob = 1./len(point_dicts)
        for point_dict in point_dicts:
            param_bounds={}
            for param in self.params:
                spacing = param_spacing[param]
                if self.logspacing[param]:
                    param_bounds[param]=(point_dict[param]/np.sqrt(spacing),point_dict[param]*np.sqrt(spacing))
                elif not self.logspacing[param]:
                    param_bounds[param]=(point_dict[param]-spacing/2.,point_dict[param]+spacing/2.)
            point=Param_point(point_dict, param_bounds, init_prob)
            self.points.append(point)

    def __str__(self):
        return "Parameter ranges: " + str(self.param_ranges) + "\n" + \
               "Logspacing: " + str(self.logspacing) + "\n" + \
               "Number of points: " + str(len(self.points))

    def normalize(self):

        """
        Normalize overall PMF.

        Should really figure out how to do the overloading properly in Param_point to make this more elegant eventually.
        """

        norm_const = sum([point.prob for point in self.points])
        for point in self.points:
            point.prob = point.prob/norm_const

    def subdivide(self, threshold_prob):

        """
        Subdivide all boxes with P > threshold_prob and assign "locally uniform" probabilities within each box.

        For now, just divides into two along each direction. Ideas for improvement:
        * divide proportional to probability mass in that box such that minimum prob is roughly equal to maximum prob of undivided boxes
        * user-specified divisions along dimensions (including NOT dividing in a given direction)
        """

        num_divs = {param:2 for param in self.params} #dummy for now

        to_subdivide = [point for point in self.points if point.prob>threshold_prob]

        for box in to_subdivide:
            # compute new parameter values and ranges
            centers={}
            bounds={}
            for param in self.params:
                if self.logspacing[param]:
                    edges = np.geomspace(box.param_bounds[param][0], box.param_bounds[param][1], num=num_divs[param]+1)
                    bounds[param] = [(edges[i],edges[i+1]) for i in range(num_divs[param])]
                    centers[param] = [np.sqrt(bounds[param][i][0]*bounds[param][i][1]) for i in range(num_divs[param])]
                elif not self.logspacing[param]:
                    edges = np.linspace(box.param_bounds[param][0], box.param_bounds[param][1], num=num_divs[param]+1)
                    bounds[param] = [(edges[i],edges[i+1]) for i in range(num_divs[param])]
                    centers[param] = [0.5*(bounds[param][i][0]+bounds[param][i][1]) for i in range(num_divs[param])]
            # create new points
            num_boxes = np.product(num_divs.values())
            vals = [centers[self.params[i]] for i in range(len(self.params))] #to preserve ordering
            point_vals = product(*vals)
            val_dicts = []
            bound_dicts = []
            for point in point_vals:
                val_dict = {}
                bound_dict = {}
                for i in range(len(self.params)):
                    param = self.params[i]
                    val_dict[param] = point[i]
                    val_ind = centers[param].index(point[i])
                    bound_dict[param] = bounds[param][val_ind]
                val_dicts.append(val_dict)
                bound_dicts.append(bound_dict)
            # awkward...
            for i in range(len(val_dicts)):
                self.points.append(Param_point(val_dicts[i], bound_dicts[i], box.prob/num_boxes))

            # remove old one
            self.points.remove(box)

        # should be normalized already, but just in case:
        self.normalize()

    def multiply(self, other_pmf):

        """
        Compute and store renormalized product of this Pmf with other_pmf.
        """

        # check for silliness
        assert isinstance(other_pmf, Pmf), "You didn't feed in a Pmf object!"
        assert len(self.points) == len(other_pmf.points), "Pmf's are over different numbers of points. Can't exactly do a pointwise multiplication on that, can I?"

        probs_temp = deepcopy(self.points)

        # do things
        for prob in probs_temp:
            # find matching point in other_pmf
            match_point = [point for point in other_pmf.points if point.params == prob.params and point.param_bounds == prob.param_bounds]
            assert len(match_point)==1, "Something is wrong! Either no matches or multiple matches to the following parameter space point: " + str(prob)
            prob.prob = prob.prob * match_point[0].prob

        self.points = probs_temp
        self.normalize()

    def most_probable(self, n):

        """
        Return the n largest probabilities.
        """

        sorted_probs = sorted(self.points, key=lambda p: p.prob, reverse=True)
        return sorted_probs[0:n]
