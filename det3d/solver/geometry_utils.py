import torch
import scipy
from scipy.sparse import csr_matrix

class SparseMatPrep:
    def __init__(self, n):
        self.n = n
        self.data, self.row, self.col = [], [], []

    def add_block_diag(self, M):
        """
            M shape=[B, D, D]
        """
        B, D = M.shape[:2]
        assert B*D == self.n
        for r in range(D):
            for c in range(D):
                self.row.append(torch.arange(B)*D + r)
                self.col.append(torch.arange(B)*D + c)
                self.data.append(M[:, r, c])
        
    def add_off_diag(self, M, row_offset, col_offset):
        """
            M shape = [E, D, D]
            offset [E, 2]
        """
        E, D = M.shape[:2]
        for r in range(D):
            for c in range(D):
                self.row.append(row_offset*D + r)
                self.col.append(col_offset*D + c)
                self.data.append(M[:, r, c])

    def output(self):
        self.data = torch.cat(self.data, dim=0)
        self.col = torch.cat(self.col, dim=0)
        self.row = torch.cat(self.row, dim=0)
        spmat = csr_matrix((self.data, (self.row, self.col)),
                           shape=(self.n, self.n))
        return spmat
       
def inverse(T):
    """
    Args:
        T torch.tensor([N, 4, 4]): transformation matrices

    Return:
        Tinv [N, 4, 4]
    """
    R, t = T[:, :3, :3], T[:, :3, 3]
    Tinv = T.clone()
    RT = R.transpose(-1, -2)
    b = -(RT @ t.unsqueeze(-1)).squeeze(-1)
    Tinv[:, :3, :3] = RT
    Tinv[:, :3, 3] = b
    return Tinv

def cross_op(r):
    """
    Args:
        r torch.tensor([N, 3]): vectors
    Return:
        rX torch.tensor([N, 3, 3]): the cross product matrix operators
    """
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    zero = torch.zeros_like(x)
    rX = torch.stack([zero,     -z,     y,
                         z,   zero,    -x,
                        -y,      x,  zero], dim=-1).reshape(-1, 3, 3)
    return rX

def rodriguez(r):
    """Compute rodrigues rotation operator.

    Args:
        r torch.tensor([N, 3]): rotations

    Return:
        R torch.tensor([N, 3, 3]): rotation matrices
    """
    theta = r.norm(p=2, dim=-1)
    valid_mask = theta > 1e-7
    R = torch.eye(3).repeat(r.shape[0], 1, 1).to(r)
    if valid_mask.any():
        r_valid = r[valid_mask]
        theta_valid = theta[valid_mask]
        k = r_valid / theta_valid.unsqueeze(-1)
        sint = theta_valid.sin().unsqueeze(-1).unsqueeze(-1)
        cost = theta_valid.cos().unsqueeze(-1).unsqueeze(-1)
        k_outer = k.unsqueeze(-1) @ k.unsqueeze(-2)
        R[valid_mask] = cost*torch.eye(3).to(r) \
                        + sint*cross_op(k) + (1-cost)*k_outer
    return R

def sym_op(a, b):
    """Compute a^TbI + ab^T

    Args:
        a torch.tensor([N, 3])
        b torch.tensor([N, 3])

    Returns:
        H torch.tensor([N, 3]): a^TbI + ab^T
    """
    H = (a * b).sum(-1)
    H = H.unsqueeze(-1).unsqueeze(-1) * torch.eye(3)
    H -= a.unsqueeze(-1) @ b.unsqueeze(-2)

    return H
