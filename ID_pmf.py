from pmf import Pmf
import math

class ID_Pmf(Pmf):
    '''
    Test case to do an ideal diode fit and practice my Python inheritance chops.
    '''

    def __init__(self, dim_lengths):
        param_names = ['n','J_0']
        dim_mins = [1,0.1] #hard-coded for now
        dim_maxes = [2,100] #hard-coded for now
        log_spacing = [False,True]
        Pmf.__init__(self, param_names, dim_lengths, dim_mins, dim_maxes, log_spacing)
        # define other stuff by self.whatever = things

    def V_T(self, T):
        return T*0.02585/300

    def compute_ID(self, V, T, params):
        '''
        Computes ideal diode output current. Params is a dict that should have
        keys the same as param names.
        '''
        # check that names match (using sets so order doesn't matter)
        # probably shouldn't have overloaded the name params either...oops
        assert set(params.keys())==set(self.params), "Parameter names must match"

        return params['J_0']*(math.exp(V/(params['n']*self.V_T(T)))-1)
