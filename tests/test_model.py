from fastalign import h, h_matrix, delta, delta_matrix
import torch


def test_h():
    m, n = 6, 10
    
    mat = h_matrix(m, n)
    mat2 = torch.zeros(m, n).float()
    for i in range(m):
        for j in range(n):
            mat2[i, j] = h(i, j, m, n)
    torch.testing.assert_close(mat, mat2)

def test_delta():
    m, n, p_0, lamb = 6, 10, 0.1, 0.1
    mat = delta_matrix(m, n, p_0, lamb)
    mat2 = torch.zeros(m, n).float()
    for i in range(m):
        for j in range(n):
            mat2[i, j] = delta(i, j, m, n, p_0, lamb)
    torch.testing.assert_close(mat, mat2)
    
