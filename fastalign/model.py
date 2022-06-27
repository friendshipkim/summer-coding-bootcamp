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
    
    # i: shape - (m + 1) x 1
    i = torch.arange(m + 1)[:, None]

    # j: shape - 1 x (n + 1)
    j = torch.arange(n + 1)[None, :]

    # Shape (m + 1) x (n + 1)
    return -torch.abs(i / m - j / n)

def delta_matrix(m, n, p_0, lambd) -> torch.Tensor:
    ""

    # shape m x n
    H = h_matrix(m, n)

    # shape m x n
    p = (lambd * H).softmax(-1)
    
    # Shape 1 x n
    j = torch.arange(n)[None, :]

    
    return torch.where(
        # 1 x n
        j == 0,
        #     1
        torch.tensor(p_0),
        # m x n
        (1 - p_0) * p)


def sample(p_0, lamd):
    Cat(delta_matrix(p_0, lamb))
