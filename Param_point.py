

class Param_point(object):
    """
    Class for a very simple object to hold a set of values for parameters and an associated probability.
    """

    def __init__(self, params, prob):
        """
        Instantiate a Param_point.

        params should be a dict of the format {param1:val1, param2:val2...}
        prob should be a number in [0,1]
        """
        self.params = params
        self.prob = prob

    def __add__(self,other):

        """
        Overload addition operator so probabilities can add in an intuitive way.
        """

        assert self.params == other.params, "These probabilities are at different points in parameter space! You probably don't actually want to add them together."

        return Param_point(self.params, self.prob + other.prob)

    def __mul__(self,other):

        """
        Overload multiplication to multiply together probabilities for Bayesian updating.
        """

        assert self.params == other.params, "These probabilities are at different points in parameter space! You probably don't actually want to multiply them together."

        return Param_point(self.params, self.prob * other.prob)

    def __str__(self):
        return "param values" + str(self.params) + " probability: " + str(self.prob)
