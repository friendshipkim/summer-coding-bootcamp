import torch
import math


# Simple Form 

def h(i: int, j: int, m: int, n: int) -> float: 
    "h(i, j, m, n) - |\frac{i}{m} - \frac{j}{n} |"
    return - abs(i / m - j /n)

def delta(i, j, m, n, p_0, lambd):
    if j == 0:
        return p_0
    elif 0 < j <= n:
        def score(j2):
            return math.exp(lambd * h(i, j2, m, n))
        return (1 - p_0) * \
          (score(j) / (sum([score(j2) for j2 in range(n)])))
    else:
        return 0


Cat = torch.distributions.Categorical

def h_matrix(m: int, n: int) -> torch.Tensor:
    """
    Figure 1 - h function

    $h_{i, j}(m, n) - |\frac{i}{m} - \frac{j}{n}|$
    """
    
    # i: shape - m x 1
    i = torch.arange(m )[:, None]

    # j: shape - 1 x (n + 1)
    j = torch.arange(n + 1)[None, :]

    # Shape m x (n + 1)
    return -torch.abs(i / m - j / n)

def delta_matrix(m, n, p_0, lambd) -> torch.Tensor:
    ""

    # shape: m x (n + 1)
    H = h_matrix(m, n)

    # shape: m x (n + 1)
    p = (lambd * H).logsoftmax(-1)
    
    # Shape: 1 x (n + 1)
    j = torch.arange(n + 1)[None, :]

    
    return torch.where(
        # 1 x (n + 1)
        j == 0,
        #     1
        torch.tensor(p_0).log(),
        # m x (n + 1)
        (1 - p_0).log() * p)


# $p(e, a| f, m, n)$

class FastAlignDistribution:
    def __init__(self, p_0, lamd, theta):
        self.p_0 = p_0
        self.lamd =lamd
        self.theta = theta

    def prob(self, e, f, m, n):
        "return p(e_i=e_i, a_i | f, m, n) for all e_i"

        # m x (n + 1)
        d = delta_matrix(m, n, self.p_0, self.lamb)

        # Vf x Ve
        theta

        # Shape: b x (n + 1), type: [V_f]
        f

        # Shape: b x (m), type: [V_e]
        e 

        # Torch version
        # b x 1 x n+1
        # b x m x 1
        #------------
        # b x m x n+1
        t = self.theta[f[:, None, :], e[:, :, None]]

    
        # Shape m x n+1, type [0, 1]
        
        #     m x (n + 1)
        # b x m x (n + 1)
        # -----------------
        # b x m x (n + 1)
        return d + t
        
    def word_probs(self, e, f, m, n):
        # Shape m x 1, type [0, 1]
        return self.prob(e, f, m, n).logsumexp(-1, keepdim=True)

    def posterior(self, e, f, m, n):
        # Shape m x n + 1, type [0, 1]
        num = self.prob(e, f, m, n)
        
        # Shape m x 1, type [0, 1]
        den = self.word_probs(e, f, m, n)
        
        return num - den

    def alignment_prob(self, e, f, m, n):
        return self.word_probs(e, f, m, n).sum(-1)
    
    

        # alignment = torch.zeros(m, n+1)
        # for i in range(m):
        #     for a_i in range(n+1):
        #         alignment[i, a_i] = d[i, a_i] * t'[i, a_i]
        
        # # 
        # alignment = torch.zeros(m, n+1)
        # for i in range(m):
        #     for a_i in range(n+1):
        #         alignment[i, a_i] = d[i, a_i] * theta[f[a_i], e[i]]

        
        # # Shape: m x (n+1), type [0, 1]
        # return alignment
        
        


def sample(m, n, p_0, lamd, f, theta):
    """
     theta -> Vf x Ve, type: [0, 1]
     f -> Shape: (n + 1), type V (n x V)
     lambda -> scalar
     p_0 -> scalar. 
     
    """

    # Shape: (n + 1), type V (n x V)
    f

    # Shape: V x Ve
    theta

    # m x (n + 1)
    cat_param = delta_matrix(m, n, p_0, lamb)

    # i =  1...m distributions over n + 1 possible values. 
    d = Cat(cat_param)

    # Shape: m ,  Type {0, ... n} 
    a_i = d.sample()

    # f_{a_i}: Shape: m Type: V
    f_a_i = f[a_i]

    # Shape: m x Ve
    word_param = theta[f_a_i]

    # i = 1...m distributions over Ve possible output
    d_e = Cat(word_param)

    # Shape: m Type: Ve
    e_i = d_e.sample()

    # Returning a sequence of m english words. 
    return e_i

# In log probs -> (+, *) -> (logsumexp(), +)
