import torch
import math
from dataclasses import dataclass

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


# Torch implementation
    
Cat = torch.distributions.Categorical

def h_matrix(m: tensor, n: tensor) -> torch.Tensor:
    """
    Figure 1 - h function

    $h_{i, j}(m, n) - |\frac{i}{m} - \frac{j}{n}|$
    """

    # shape: batch
    m_max = m.max()
    
    # shape: batch
    n_max = n.max()
    
    # i: shape - m_max x 1
    i = torch.arange(m_max)[:, None]

    # j: shape - 1 x (n_max + 1)
    j = torch.arange(n_max + 1)[None, :]

    # Shape
    # i / m
    # i : m_max x 1
    # m : batch x 1 x 1
    # -----------
    
    return -torch.abs(i / m[:, None, None] - j / n[:, None, None])

    # Shape batch x m_max x (n_max + 1)


def delta_matrix(m, n, p_0, lambd) -> torch.Tensor:
    ""
    assert p_0 >= 0 and p_0 <= 1.0
    assert lambd >= 0

    # m, n are batch size vectors

    
    # shape: batch x m_max x (n_max + 1)
    H = h_matrix(m, n)

    # shape: m x n
    p = (lambd * H[:, 1:]).log_softmax(-1)
    
    
    #     1
    # m x n
    # ----------
    # m x (n + 1)
    delta = torch.cat([p_0.log()[None, :].expand(p.shape[0], 1),
                       (1 - p_0).log() + p], dim=-1)
    return delta
    # Shape: 1 x (n + 1)
    # j = torch.arange(n + 1)[None, :]
    # return torch.where(
    #     # 1 x (n + 1)
    #     j == 0,
    
    #     p_0.log(),
    #     # m x n
    #     (1 - p_0).log() + p)


# $p(e, a| f, m, n)$

class FastAlignDistribution(torch.nn.Module):
    def __init__(self, E_SIZE, F_SIZE):
        super().__init__()
        self.p_0 = torch.nn.Parameter(torch.zeros(1))
        self.lambd = torch.nn.Parameter(torch.zeros(1))
        self.theta = torch.nn.Parameter(torch.zeros((F_SIZE, E_SIZE)))

    def prob(self, e, f, m, n, theta, lambd, p_0):
        "return p(e_i=e_i, a_i | f, m, n) for all e_i"

        # m x (n + 1)
        d = delta_matrix(m, n, p_0, lambd)

        assert torch.isclose(d[0].exp().sum(), torch.tensor(1.0)), d[0].exp()
        
        # # Vf x Ve
        # theta

        # # Shape: b x (n + 1), type: [V_f]
        # f

        # # Shape: b x (m), type: [V_e]
        # e 

        # Torch version
        # b x 1 x n+1
        # b x m x 1
        #------------
        # b x m x n+1
        t = theta[f[:, None, :], e[:, :, None]]

    
        # Shape m x n+1, type [0, 1]
        
        #     m x (n + 1)
        # b x m x (n + 1)
        # -----------------
        # b x m x (n + 1)
        return d + t

    def word_probs(self, e, f, m, n, theta, lambd, p_0):
        # Shape m x 1, type [0, 1]
        return self.prob(e, f, m, n, theta, lambd, p_0).logsumexp(-1, keepdim=True)

    
    def forward(self, e, f, m, n):
        # p(e | f)
        theta = self.theta.log_softmax(-1)
        lambd = self.lambd.exp()
        p_0 = self.p_0.sigmoid()
        return self.word_probs(e, f, m, n, theta, lambd, p_0).sum(-1)
    

    def posterior(self, e, f, m, n):
        # Shape m x n + 1, type [0, 1]
        num = self.prob(e, f, m, n)
        
        # Shape m x 1, type [0, 1]
        den = self.word_probs(e, f, m, n)
        
        return num - den

@dataclass
class Data:
    e_examples : torch.Tensor
    f_examples : torch.Tensor
    e_size : int
    f_size : int

e_examples = []
f_examples = []
for i in range(100):
    word = i % 4 + 1
    word2 = i % 6 + 1
    e, f = [word, word2], [0, word2, word]
    e_examples.append(e)
    f_examples.append(f)

e_examples = torch.tensor(e_examples)
f_examples = torch.tensor(f_examples)
data = Data(e_examples, f_examples, 7, 7)
    
def train(train_data):
    model = FastAlignDistribution(train_data.e_size, train_data.f_size)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    BATCH_SIZE = 10
    for epochs in range(10):
        for i in range(0, train_data.e_examples.shape[0], BATCH_SIZE):
            opt.zero_grad()
            e = train_data.e_examples[i: i + BATCH_SIZE]
            f = train_data.f_examples[i: i + BATCH_SIZE]
            loglikelihood = model.forward(e, f, e.shape[-1], f.shape[-1] - 1)
            loss = -loglikelihood.mean()
            loss.backward()
            print("loss", loss)
            print(model.theta.softmax(-1)[1])
            opt.step()

            # 1 x E_SIZE () E_SIZE x 1  => E_SIZE x E_SIZE =>(view) E_SIZE * E_SIZE

            # A + B -> cat 
            
            # (E_SIZE * E_SIZE) x 2

            # (0, 0)
            # (0, 1)
            # (0, 2)
            # ....
            # (E_SIZE-1, E_SIZE-1)
            # E = train_data.e_size
            # all_es = torch.cat([torch.arange(E)[None, :, None].expand(E, E, 1),
            #                     torch.arange(E)[:, None, None].expand(E, E, 1)],
            #                    dim=-1).view(-1, 2)

            # # E_SIZE x (1 + 1)
            # f_data = f[0][None, :]
            # out = model.forward(all_es, f_data, 1, f_data.shape[-1] - 1)
            # assert out.exp().sum(), out.exp()
            # print(out.exp())
            

train(data)
        


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
