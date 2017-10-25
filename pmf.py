import numpy as np

class Pmf(object):
    """
    Class that stores a PMF capable of nested sampling / "adaptive mesh refinement".

    Stores probabilities on a matrix at maximum level of refinement but only updates
    values where indicator variable m (stored in a parallel matrix) is equal to M,
    the global value.

    Includes methods for normalization and multiplication that will be useful in
    Bayesian updating steps, which will likely be implemented elsewhere, but basic
    procedure would be: at each step we will create a likelihood Pmf that's uniform
    and is a clone of the "master" one with respect to M and m values, then only
    update at places where m=M

    [TODO: IT SHOULD UPDATE AT OTHER PLACES BUT ONLY AT THE COARSENESS OF THAT
    LEVEL OF SUBDIVISION]
    --> this will prevent "freezing out" of regions that later observations make more probable
    --> could in principle be avoided by clever initial bounds and thresholding

    I think for generality there will have to be a separate kernel (or something?)
    for how actual likelihoods are computed and allowing for flexible number of
    observation conditions etc. too...needs more thought.

    QUESTIONS TO THINK ABOUT:
    Where should info about observation conditions be stored? Does it need to be?

    How about simulation results? Can they be stored separately but maintain a
    parallel/analogous data structure? Perhaps just a big index array?

    Maybe for each particular system (e.g. JVTi with PC1D sims, TIDLS with SRH
    solver, etc. there would be a class that inherits these methods but includes more
    particulars to that case?

    """

    def __init__(self, param_names, dim_lengths, dim_mins, dim_maxes, log_spacing):
        '''
        instantiate a uniform prior, map from dimension to
        parameter name, and protected indicators

        param_names = list of strings
        dim_lengths, dim_mins, dim_maxes = lists of ints
        log_spacing = list of bools

        '''
        # check that you haven't fed in anything silly
        l = len(param_names)
        assert len(dim_lengths)==l and len(dim_mins)==l and len(dim_maxes)==l and len(log_spacing)==l, "Lengths of all inputs need to match!"

        assert all([dim_maxes[i]>dim_mins[i] for i in range(len(dim_maxes))]), "Maximum values must be greater than minimum values!"

        # copy in things
        self.params = param_names # order corresponds to order of indices
        self.param_ranges={self.params[i]:[dim_mins[i],dim_maxes[i]] for i in range(len(param_names))}
        self.logspacing = log_spacing

        # initiate uniform prior with m=1 everywhere and M=1 for whole PMF
        self.probs = np.ones(dim_lengths)/np.prod(dim_lengths)
        self._M = 1
        self._m = np.ones(dim_lengths)

    def dim_lengths(self):
        '''
        Return current lengths along each dimension.

        '''
        return {self.params[i]:self.probs.shape[i] for i in range(len(self.params))}

    def var_range(self, param_ind, num):
        '''
        Helper function for box_edges and box_centers
        '''
        range = self.param_ranges[self.params[param_ind]]
        if self.logspacing[param_ind] == True:
            return np.logspace(np.log10(range[0]),np.log10(range[1]),num)
        elif self.logspacing[param_ind] == False:
            return np.linspace(range[0],range[1],num)

    def box_edges(self):
        '''
        Calculate boundaries of each box along each dimension
        (outputs will be of length dim_lengths * M + 1)
        '''
        return {self.params[i]:self.var_range(i,self.probs.shape[i]+1) for i in range(len(self.params))}

    def box_centers(self):
        '''
        Calculate values at center of each box along each dimension
        (outputs will be of length dim_lengths * M)

        (it takes 2N+1 vals (i.e. edges and centers) and then picks odd-indexed values
        to get centers)
        '''
        return {self.params[i]:self.var_range(i,2*self.probs.shape[i]+1)[np.arange(1,2*self.probs.shape[i]+1,2)] for i in range(len(self.params))}


    def normalize(self):
        '''
        should fix probability values at lower values of M than the current one
        only change the values where m=M
        (this may not be necessary depending on how Bayesian update is implemented
        but probably easier to have this way)

        TODO: if this gets slow, reimplement with numba.jit
        '''
        # first create mask array of where m == M
        #mask = self._m==1
        mask = self._m[:]==self._M
        #print('mask:',mask)

        # check that parts we won't change sum to less than 1, otherwise can't normalize
        assert np.sum(self.probs[np.invert(mask)]) < 1.0, "Can't normalize, fixed values sum to greater than 1!"

        # compute normalization constant
        n = np.sum(self.probs[mask])/(1-np.sum(self.probs[np.invert(mask)]))
        #print('sum:',np.sum(self.probs),'norm const:',n)

        # normalize
        self.probs[mask] = self.probs[mask]/n

        # deal with some machine precision issues
        self.probs = self.probs/np.sum(self.probs)

        # check that overall distribution is normalized
        assert abs(np.sum(self.probs)-1.0)<1e-14, "normalization didn't work!"

    def subdivide(self, threshold_prob):
        '''
        Subdivide all boxes with P > threshold_prob and assign "locally uniform"
        probabilities within each box.

        Functionally, what this means is subdivide the whole matrix but only update
        values of m in the boxes with large enough probability.

        (eventually should allow thresholding as a fraction or number of total boxes
        and implement some sensible default options - also perhaps allow for flexibly
        sized subdivision - customized along each direction, etc.)

        (either only subdivide by even numbers or be smart about copying results
        back into center boxes if subdividing oddly)

        TODO: if this gets slow, it's probably due to np.kron - could rewrite with np.repeat

        '''
        # number of subdivisions along each dimension
        num_divs = 2 * np.ones(len(self.params),dtype=np.int) #dummy for now
        #print('product of divs:', np.prod(num_divs))
        # increment m values before expanding matrix
        mask = self.probs>threshold_prob
        #print('mask:',mask)
        #self._m[mask] = self._m[mask]+1
        #print('old m:',self._m)
        self._m[mask] = self._M+1 #I'm pretty sure this is actually the correct one
        #print('new m:',self._m)
        # use Numpy Kronecker product to expand the matrices (normalizing probs)
        self._m = np.kron(self._m,np.ones(num_divs))
        #print('expanded m:',self._m)
        #print('probs before expanding:',self.probs)
        #print('expanded probs before normalizing:',np.kron(self.probs,np.ones(num_divs)))
        self.probs = np.kron(self.probs,np.ones(num_divs))/np.prod(num_divs)

        # increment overall subdivision indicator
        self._M = self._M +1

    def multiply(self, other_pmf):
        '''
        Compute and store renormalized product of this Pmf with other_pmf.
        Only multiply and normalize boxes where m = M in THIS Pmf.
        (Note that this makes it a non-commutative operation)

        TODO:
            * maybe this should return a new Pmf object instead? TBD
            * figure out how to check
        '''

        # check for silliness
        assert isinstance(other_pmf, Pmf), "You didn't feed in a Pmf object!"
        assert other_pmf.probs.shape == self.probs.shape, "Pmf sizes and shapes must match!"
        #maybe also add checks for parameter names and ranges? Should probably just throw
        #warnings though if mismatched

        # make mask and multiply
        mask = self._m==self._M
        self.probs[mask] = np.multiply(self.probs[mask],other_pmf.probs[mask])

        # and normalize
        self.normalize()

    def most_probable(self, n):
        '''
        Returns the n largest probabilities and the associated parameter values.

        TODO:
        Make sure this is fully general to number of parameters.
        '''

        flat_indices = np.argpartition(self.probs.ravel(), -n)[-n:]
        inds = np.unravel_index(flat_indices, self.probs.shape)

        highest_probs=[{} for a in range(n)]
        for i in range(n):
            ind = [inds[j][i] for j in range(len(inds))]
            prob_dict = {'prob': self.probs[tuple(ind)]}
            for k in range(len(self.params)):
                prob_dict[self.params[k]]=self.box_centers()[self.params[k]][ind[k]]
            highest_probs[i]=prob_dict

        return highest_probs
