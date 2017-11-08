

class Param_point(object):
    """
    Class for a very simple object to hold a set of values for parameters and an associated probability. Also stores ranges over which its parameter values are the centers.
    """

    def __init__(self, params, param_bounds, prob):
        """
        Instantiate a Param_point.

        params should be a dict of the format {param1:val1, param2:val2...}
        param_bounds should be like {param1:(min1,max1), param2:(min2,max2)}
        prob should be a number in [0,1]
        """
        self.params = params
        self.prob = prob
        self.param_bounds = param_bounds

    def __add__(self,other):

        """
        Overload addition operator so probabilities can add for normalization.

        Returns a float, not a Param_point.
        """

        return self.prob + other.prob

    def __mul__(self,other):

        """
        Overload multiplication to multiply together probabilities for Bayesian updating.
        """

        assert self.params == other.params, "These probabilities are at different points in parameter space! You probably don't actually want to multiply them together."

        return Param_point(self.params, self.param_bounds, self.prob * other.prob)

    def __float__(self):
        return float(self.prob)

    def __str__(self):
        return "param ranges: " + str(self.param_bounds) + "\n" + \
               "centers: " + str(self.params) + "\n" + \
               "probability: " + str(self.prob) + "\n"
