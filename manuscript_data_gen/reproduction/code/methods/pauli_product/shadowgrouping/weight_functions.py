import numpy as np

class Inconfidence_bound():
    
    def __init__(self,focus_on_greedy=True):
        self.greedy = focus_on_greedy
        return
    
    def get_weights(self,w,eps,N_hits):
        inconf = np.exp( -0.5*eps*eps*N_hits/(w**2) )
        inconf -= np.exp( -0.5*eps*eps*(N_hits+1)/(w**2) )
        if self.greedy:
            inconf[N_hits == 0] -= 1
            inconf[N_hits == 0] *= -1
        return inconf
    
    def __call__(self):
        return self.get_weights
    
class Bernstein_bound():
    def __init__(self,alpha=1):
        self.alpha = alpha
        assert alpha >= 1, "alpha has to be chosen larger or equal 1, but was {}.".format(alpha)
        return
    
    def get_weights(self,w,eps,N_hits):
        inconf = self.alpha * np.abs(w)
        condition = N_hits != 0
        N = np.sqrt(N_hits[condition])
        Nplus1 = np.sqrt(N_hits[condition] + 1)
        inconf[condition] /= self.alpha*N*Nplus1/(Nplus1-N)
        return inconf
    
    def __call__(self):
        return self.get_weights
    

