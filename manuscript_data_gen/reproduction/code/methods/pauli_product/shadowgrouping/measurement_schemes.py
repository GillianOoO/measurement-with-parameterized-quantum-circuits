import numpy as np
from itertools import product
from time import time

##########################################################################################
### Helper functions #####################################################################
##########################################################################################
def hit_by(O,P):
    """ Returns whether o is hit by p """
    for o,p in zip(O,P):
        if not (o==0 or p==0 or o==p):
            return False
    return True

def setting_to_str(arr):
    out = ""
    for a in np.array(arr).flatten():
        out += str(a)
    return out

# equation 6 from manuscript
N_delta = lambda delta: 4*(2*np.sqrt(-np.log(delta))+1)**2

##########################################################################################
### Measurement schemes used for benchmark ###############################################
##########################################################################################

class L1_sampler:
    """ Comparison class that does not reconstruct the Hamiltonian expectation value by its components, but by its relative signs. """
    
    def __init__(self,observables,weights,epsilon):
        assert len(observables.shape) == 2, "Observables has to be a 2-dim array."
        M,n = observables.shape
        weights = weights.flatten()
        assert len(weights) == M, "Number of weights not matching number of provided observables."
        assert epsilon > 0, "Epsilon has to be strictly positive"
        abs_vals = np.abs(weights)
        
        self.obs         = observables
        self.num_obs     = M
        self.num_qubits  = n
        self.w           = weights
        self.prob        = abs_vals / np.sum(abs_vals)
        self.eps         = epsilon
        self.shots       = 0
        self.is_sampling = True
        self.is_adaptive = False
        
        return
    
    def reset(self):
        self.shots = 0
    
    def find_setting(self,num_samples=1):
        self.shots += num_samples
        inds = np.random.choice(self.num_obs,size=(num_samples,),p=self.prob)
        return inds
        
    def get_Hoeffding_bound(self):
        return 2*np.exp(-0.5*self.eps**2*self.shots/np.sum(np.abs(self.w))**2)
    
    def get_epsilon_sys_stat(self,delta):
        return (0,np.sqrt(2/self.shots*np.log(2/delta)) * np.sum(np.abs(self.w)))

class Measurement_scheme:
    """ Parent class for measurement schemes. Requires
        observables: Array of shape (num_obs x num_qubits) with entries in {0,1,2,3} (the Pauli operators)
        weights:     Array of shape (num_obs) with the corresponding weight in the Hamiltonian decomposition.
                     Array is flattened upon input.
        epsilon:     Absolute error threshold, see child methods for an individual interpretation.
    """
    
    def __init__(self,observables,weights,epsilon):
        assert len(observables.shape) == 2, "Observables has to be a 2-dim array."
        M,n = observables.shape
        weights = weights.flatten()
        assert len(weights) == M, "Number of weights not matching number of provided observables."
        assert epsilon > 0, "Epsilon has to be strictly positive"
        
        self.obs           = observables
        self.num_obs       = M
        self.num_qubits    = n
        self.w             = weights
        self.eps           = epsilon
        self.scheme_params = {"eps": epsilon, "num_obs": M}
        self.N_hits        = np.zeros(M,dtype=int)
        self.is_adaptive   = False # useful default to be given to any child class
        
        return
        
    def find_setting(self):
        pass
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        return
    
    def get_epsilon_sys_stat(self,delta):
        """ Applies the truncation strategy (see truncate() for details) and returns the corresponding epsilon values for the 
            systematic and the statistical error, respectively. Does not alter the scheme in-place, compared to truncate() would do.
        """
        N_crit = N_delta(delta)
        keep = self.N_hits > int(N_crit) # round down to integer value
        if np.sum(keep) == 0:
            # only systematic error
            eps_syst = np.sum(np.abs(self.w))
            eps_stat = 0
        elif np.sum(keep) == len(keep):
            # only statistical error
            eps_syst = 0
            eps_stat = self.get_epsilon_Bernstein(delta)
        else:
            w, N = self.w, self.N_hits
            # override temporarily
            self.w = self.w[keep]
            self.N_hits = self.N_hits[keep]
            # calculate guarantees
            eps_syst = np.sum(np.abs(w[np.bitwise_not(keep)]))
            eps_stat = self.get_epsilon_Bernstein(delta)
            # undo overwriting
            self.w = w
            self.N_hits = N
        return eps_syst, eps_stat
    
    def truncate(self,delta):
        """ Truncation function to apply the truncation criterion given a certain inconfidence level delta.
            Assumes that scheme has called the function find_setting() sufficiently often.
            Truncates all observables that fulfill the truncation criterion and save the sum of their absolute coefficient values.
            Returns the resulting introduced systematic error epsilon.
        """
        N_unmeasured = np.sum(self.N_hits == 0)
        if N_unmeasured > 0:
            print("Warning! {} observable(s) have not been measured at least once.".format(N_unmeasured))
            print("If you have set alpha large, this can result in a non-optimal truncation.")
        N_crit = N_delta(delta)
        keep = self.N_hits > int(N_crit) # round down to integer value
        if np.sum(keep) == 0:
            print("No observable reached the threshold. Ensure that you have sampled often enough or provide a smaller delta!")
            print("Scheme unaltered.")
            return 0
        if np.sum(keep) == len(keep):
            print("Nothing had to be truncated.")
            return 0
        eps_sys = np.sum(np.abs(self.w[np.bitwise_not(keep)]))
        self.w = self.w[keep]
        self.obs = self.obs[keep]
        self.N_hits = self.N_hits[keep]
        self.num_obs = len(self.w)
        return eps_sys
    
    def get_epsilon_Bernstein(self,delta):
        """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
            If at least one of the N_hits is 0, epsilon is set equal to infinity.
            Else, epsilon = 2*|weights/sqrt(N_hits)| * (1 + 2sqrt(log(1/delta)))
        """
        if np.min(self.N_hits) == 0:
            return np.infty
        w_abs  = np.abs(self.w)
        w_abs /= np.sqrt(self.N_hits)
        norm   = np.sum(w_abs)
        w_abs /= np.sqrt(self.N_hits)
        norm2  = np.sum(w_abs)
        epsilon = norm * np.sqrt(N_delta(delta))
        if epsilon > 2*norm*(1+2*norm/norm2):
            print("Warning! Epsilon out of validity range.")
        return epsilon
        
class Shadow_Grouping(Measurement_scheme):
    """ Grouping method based on weights obtained from classical shadows.
        The next measurement setting p is found as follows: it is initialized as the identity operator.
        Next, we obtain an ordering of the observables in terms of their respective weight_function.
        For each observable o in the ordered list of observables in descending order, it checks qubit-wise commutativity (QWC).
        If so, the qubits in p that fall in the support of o are overwritten by those in o.
        Eventually, the list is either exhausted or p does not contain identity operators anymore.
        The function weight_function takes in the weights,epsilon and the current number of N_hits and is supposed to return an numpy-array of length len(w).
        Instead, weight_function can also be set to None (this is useful for instances where the function is actually never called).
        
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon,weight_function):
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        if self.weight_function is not None:
            test = self.weight_function(self.w,self.eps,self.N_hits)
            assert len(test) == len(self.w), "Weight function is supposed to return an array of shape {} (i.e. number of observables) but returned an array of shape {}".format(self.w.shape,test.shape)
        self.is_sampling = False
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        return
    
    def get_inconfidence_bound(self):
        inconf = np.exp( -0.5*self.eps*self.eps*self.N_hits/(self.w**2) )
        return np.sum(inconf)
    
    def get_Bernstein_bound(self):
        if np.min(self.N_hits) == 0:
            bound = -1
        else:
            bound = np.exp(-0.25*(self.eps/2/np.sum(np.abs(self.w)/np.sqrt(self.N_hits))-1)**2)
        return bound            
        
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        # sort observable list by respective weight
        weights = self.weight_function(self.w,self.eps,self.N_hits)
        order = np.argsort(weights)
        setting = np.zeros(self.num_qubits,dtype=int)

        if verbose:
            print("Checking list of observables.")
        tstart = time()
        for idx in reversed(order):
            o = self.obs[idx]
            if verbose:
                print("Checking",o)
            if hit_by(o,setting):
                non_id = o!=0
                # overwrite those qubits that fall in the support of o
                setting[non_id] = o[non_id]
                if verbose:
                    print("p =",setting)
                # break sequence is case all identities in setting are exhausted
                if np.min(setting) > 0:
                    break
                    
        tend = time()

        # update number of hits
        is_hit = np.array([hit_by(o,setting) for o in self.obs],dtype=bool)
        self.N_hits += is_hit
        
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info
    
class Brute_force_matching(Shadow_Grouping):
    """ Comparison class to Shadow_Grouping. Runs through all 3**num_qubit possibilities, thus finding the optimal next
        measurement setting p.
        The target (str or user_function) specifies the member function (if str) to maximize over (defaults to Bernstein bound).
        
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon,target="Bernstein_bound"):
        super().__init__(observables,weights,epsilon,None)
        if isinstance(target,str):
            self.target_is_member_function = True
            try:
                self.weights = getattr(self,"get_"+target)
            except:
                print("Warning! Unknown member-function get_{} called. Defaulting to get_Bernstein_bound instead.".format(target))
                self.weights = self.get_Bernstein_bound
        else:
            self.target_is_member_function = False
            self.weights = target
        self.is_sampling = False
        return
    
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        best_setting, best_weight = [], np.infty
        if verbose:
            print("Brute-force searching all measurement settings")
        tstart = time()
        for P in product(range(1,4),repeat=self.num_qubits):
            temp_hit = np.array([hit_by(o,P) for o in self.obs])
            self.N_hits += temp_hit
            temp = self.weights() if self.target_is_member_function else np.sum(self.weights(self.w,self.eps,self.N_hits))
            self.N_hits -= temp_hit
            if temp < best_weight:
                best_setting, best_weight = [P], temp
            elif temp == best_weight:
                best_setting.append(P)
        tend = time()
        if verbose:
            print("Best assignment are {} with max weight of {}".format(best_setting,best_weight))
        
        # if multiple setting have been found, returns one at random
        n = len(best_setting)
        if n==1:
            setting = best_setting[0]
        else:
            setting = best_setting[np.random.choice(n)]
            
        # update number of hits
        is_hit = np.array([hit_by(o,setting) for o in self.obs],dtype=bool)
        self.N_hits += is_hit
        
        info = {"best_settings":      best_setting,
                "total_weight":       best_weight,
                "inconfidence_bound": self.get_inconfidence_bound(),
                "Bernstein bound":    self.get_Bernstein_bound(),
                "run_time":           tend - tstart
               }
            
        return np.array(setting), info        

class AdaptiveShadows(Shadow_Grouping):
    """ Comparison class to Shadow_Grouping, based on https://github.com/charleshadfield/adaptiveshadows/.
        Starts-off as classical shadows (uniformly at random) but biases the distribution
        the more the Pauli bases have been set. Does not require any hyperparameters.
        epsilon (optional): parameter solely used for comparison with other methods. Defaults to 0.1.
        
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon=0.1):
        super().__init__(observables,weights,epsilon,None)
        self.is_sampling = True
        return
    
    def __isCompatible(self, pauli, j, qubits_shift, bases_shift):
        """ Helper function to check whether the current pauli term is compatible with the current
            partial assignment and whether the pauli term has a non-identity at the current qubit index.
        """
        if pauli[qubits_shift[j]] == 0:
            return False
        for k in range(j):
            i = qubits_shift[k]
            if not pauli[i] in (0, bases_shift[k]):
                return False
        return True
    
    def __generateBeta(self, j, qubits_shift, bases_shift):
        """ Calculate the probabilities for drawing either X,Y or Z for the j-th qubit in permuted order.
            This assignment is conditioned on the previously assigned qubits in the current iteration.
        """
        constants = [0.0, 0.0, 0.0]
        # loop through all Pauli terms with their respective weights
        for coeff, pauli in zip(self.w, self.obs):
            # if current term is still compatible with current assignment
            # and does not yield an identity at the current qubit index,
            # adjust the corresponding weights
            if self.__isCompatible(pauli, j, qubits_shift, bases_shift):
                index = pauli[qubits_shift[j]] - 1 # index pauli[...] cannot be the identity
              ##  print(coeff, pauli, index) ###noted
                index = int(index)
                constants[index] += coeff**2
        beta_unnormalized = np.sqrt(constants)
        norm = np.sum(beta_unnormalized)
        if norm == 0:
            beta = np.ones(3)/3
        else:
            beta = beta_unnormalized / norm
        return beta
    
    def __generateBasisSingle(self, j: int, qubits_shift: list, bases_shift: list) -> str:
        """ Sample the operator for the j-th qubit in permuted order. """
        assert len(bases_shift) == j
        beta = self.__generateBeta(j, qubits_shift, bases_shift)
        basis = np.random.choice([1, 2, 3], p=beta)
        return basis
    
    def find_setting(self,verbose=False):
        """ Generate the next Pauli measurement string by randomly permuting the qubits and sampling from
            beta = otimes_i beta_i
        """
        n = self.num_qubits
        # randomly permute the qubit order
        tstart = time()
        qubits_shift = list(np.random.permutation(n))
        bases_shift = []
        for j in range(n):
            basisSingle = self.__generateBasisSingle(j, qubits_shift, bases_shift)
            bases_shift.append(basisSingle)
        # undo the permutation by adding the single operators to output basis B
        setting = []
        for i in range(n):
            j = qubits_shift.index(i)
            setting.append(bases_shift[j])
            
        tend = time()
            
        # update number of hits
        is_hit = np.array([hit_by(o,setting) for o in self.obs],dtype=bool)
        self.N_hits += is_hit
        
        info = {"inconfidence_bound": self.get_inconfidence_bound(),
                "Bernstein bound":    self.get_Bernstein_bound(),
                "run_time":           tend - tstart
               }
            
        return np.array(setting), info
            
class SettingSampler(Measurement_scheme):
    """ Comparison class to ShadowGrouping if the sampling distribution p can be provided explicitly.
        filename_for_distribution: string that points to the file containing the distribution and its corresponding settings
            see load_distribution_setting() for further information of data formatting.
        epsilon (optional): parameter solely used for comparison with other methods. Defaults to 0.1.
        
        Returns p and a dictionary info holding further details on the matching procedure.
        Note that due to the sampling, find_setting() can yield multiple settings.
    """
    def __init__(self,observables,weights,filename_for_distribution,epsilon=0.1):
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.load_distribution_setting(filename_for_distribution)
        self.is_sampling = True
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        return

    def load_distribution_setting(self,filename):
        """ Helper function to read the distribution and the corresponding settings from file.
            Data must be stored as a matrix of form (N+1,n) where n = # qubits and N = # settings.
            The last row corresponds to the entries of the distribution
        """
        data = np.loadtxt(filename)
        self.p = data[-1]
        self.p /= np.sum(self.p)
        self.settings = data[:-1].T
        return

    def find_setting(self,N_samples=1):
        """ Generate settings from the given distribution p. Can find multiple settings at once by providing a value for
            N_samples (int). Returns the setting(s) and a dictionary holding the information about the number of settings sampled.
        """
        inds = np.random.choice(len(self.p),size=(N_samples,),p=self.p)
        Q = self.settings[inds]
        for ind, repeats in zip(*np.unique(inds,return_counts=True)):
            # update number of hits for each of the unique elements in Q
            # by counting over the index vector, instead
            is_hit = np.array([hit_by(o,self.settings[ind]) for o in self.obs],dtype=int)
            self.N_hits += is_hit*repeats
        if N_samples==1:
            Q = Q.flatten()
        return Q, {"N_samples": N_samples}
    
class Derandomization(Shadow_Grouping):
    
    """ Finds the next measurement setting following the derandomization procedure.
        Optionally, a parameter delta in [0,1] can be provided to vary the degree of randomness (delta == 1 fully random, delta == 0 as proposed).
        If num_measurements is provided, the corresponding inconfidence bound is adapted to that.
        If use_one_norm, implements a 1-norm weighting to the bound as proposed in the paper.
    """

    def __init__(self,observables,weights,epsilon,delta=0,num_measurements=None,use_one_norm=False):
        super().__init__(observables,weights,epsilon,None)
        
        self.num_measurements = num_measurements
        # (n x M) integer array with entries in {0,1,2,3} == {E,X,Y,Z}
        self.localities = np.zeros((self.num_qubits+1,self.num_obs),dtype=int) # keep the last zero as the support of an empty Pauli string
        self.localities[:-1,:] = np.array([np.sum(observables[:,i:]!=0,axis=1) for i in range(self.num_qubits)])
        self.N_hits = np.zeros(self.num_obs,dtype=int)
        self.eps_greedy = delta
        self.scheme_params["eps_greedy"] = delta
        self.scheme_params["use_one_norm"] = use_one_norm
        
        if use_one_norm:
            self.use_one_norm = True
            self.w_factor = np.abs(self.w)
            self.w_factor /= np.max(self.w_factor)
            #self.wmax = np.max(np.abs(self.w))
            self.nu = 1 - np.exp(-epsilon*epsilon/2)
        else:
            self.use_one_norm = False
            self.w_factor = self.w**2
            self.nu = 1 - np.exp(-epsilon*epsilon/2/self.w/self.w)
            
        self.log_locality_factor = np.log(1-self.nu/(3**self.localities[0]))
        
        self.assignments = [] # for the next measurement setting
        self.m_k_counter = [0,0] # convenience internal counter = (num_settings so far, current qubit pos)
        self.last_assignment = None
        self.is_sampling = False
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.assignments = []
        self.m_k_counter = [0,0]
        self.last_assignment = None
        return
    
    def __step(self, action):
        """ Tries out the effect of the chosen assignment.
            Returns the corresponding inconfidence bound upon this choice and an increment.
            It is a boolean list in case a new measurement setting is produced and None-type else.
        """
        
        self.assignments.append(action) # actions are in {1,2,3}
        self.m_k_counter[1] += 1
        # check whether to roll over to next measurement setting
        if len(self.assignments) >= self.num_qubits:
            self.m_k_counter = [self.m_k_counter[0]+1, 0]
            # start new measurement setting and check whether the previous setting hits any observables
            self.last_assignment = self.assignments.copy()
            increment = np.array([hit_by(self.obs[i],self.last_assignment) for i in range(self.num_obs)],dtype=int)
            self.N_hits += increment
            self.assignments = []
        else:
            increment = None

        return self.derandom_bound(), increment
    
    def __step_back(self,increment=None):
        """ Reverts the effect of _step() in terms of internal counters. """
        if len(self.assignments) == 0:
            # revert to old measurement setting in case of roll-over
            self.m_k_counter[0] -= 1 # decrease num_settings by one
            self.m_k_counter[1] = self.num_qubits - 2
            assert increment is not None, "Increment should not have been None-type when rolling back."
            self.N_hits -= increment
            if self.last_assignment is not None:
                self.assignments = self.last_assignment[:-1]
            else:
                self.m_k_counter = [0,0] # reinitialize in this case
        else:
            self.assignments.pop()
            self.m_k_counter[1] -= 1
        return

    def derandom_bound(self):
        """ Given a set of previous assignments in self.assignments, calculates the current inconfidence bound. """
        m,qubit_k = self.m_k_counter
        p = self.assignments
        temp = self.nu/(3**self.localities[qubit_k])
        # calculate product of the second term for the first k qubit operators
        sign = np.array([hit_by(o[:qubit_k],p) for o in self.obs])
        temp = np.log(1-temp*sign) # element-wise operations
        # first term for every observable
        if self.use_one_norm:
            temp -= self.eps*self.eps/2*self.N_hits
            temp /= self.w_factor
        else:
            temp -= self.eps*self.eps/2*self.N_hits/self.w_factor
        # third term for every observable if applicable
        if self.num_measurements is not None:
            temp += (self.num_measurements-m-1)*self.log_locality_factor
        bound = np.sum(np.exp(temp))
        return bound
    
    def find_setting(self, verbose=False, previous_bound=None):
        """ Tries all three possible Pauli assignments and picks epsilon-greedy to minimize the inconf. bound  """
        assert self.assignments == [], "Current assignment list is not empty. Please empty first."
        if self.num_measurements is not None:
            if self.m_k_counter[0] >= self.num_measurements:
                print("Warning! Measurement scheme already reached the max. number of measurements, given by {}. Returned an empty assignment".format(self.num_measurements))
                return [], {}
        previous_bound = self.get_inconfidence_bound() if previous_bound is None else previous_bound
        info = {"previous_bound": previous_bound}
        tstart = time()
        if verbose:
            print("Running epsilon-greedy derandomized scheme with epsilon = {}".format(self.eps_greedy))
        for n in range(self.num_qubits):
            if np.random.rand() < self.eps_greedy:
                # check for random action with probability eps_var
                action = np.random.choice(3) + 1
                inconf, increment = self.__step(action)
                assert increment is None or n+1 == self.num_qubits, "Increment was not None-type but should have been."
            else:
                # pick among argmin else
                temp = []
                for i in range(1,4):
                    inconf, increment = self.__step(i)
                    assert increment is None or n+1 == self.num_qubits, "Increment was not None-type but should have been."
                    temp.append(previous_bound - inconf)
                    self.__step_back(increment)
                action = np.argmax(temp) + 1
                inconf, increment = self.__step(action)
            previous_bound = inconf
            if verbose:
                temp = self.assignments if n + 1 < self.num_qubits else self.last_assignment
                print(temp)
        tend = time()
        assert increment is not None, "Increment was None-type but should have been list."        
            
        # further information
        #info["total_weight"] = np.sum(self.get_inconf()[increment])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        #if verbose:
            #print("Finished assigning with total weight of",info["total_weight"])
        
        return np.array(self.last_assignment), info
    

