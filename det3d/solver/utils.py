from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np

def find_graphs(edges, temporal=True, return_subgraphs=False):
    """Find connected components of the graph defined by
    self.vv and self.vv_f.

    Args:
        edges (M, 2): edges torch.tensor

    Returns:
        graph_indices (list[np.ndarray])
    """
    #vv0 = self.vv0.cpu()
    #vv1 = self.vv1.cpu()
    e0, e1 = edges.T.cpu()

    A = csr_matrix((torch.ones_like(vv0), (vv0, vv1)),
                   shape=(self.num_voxels, self.num_voxels))
    #if temporal:
    #    vv0_f = self.active_voxels[self.vv0_f].cpu()
    #    vv1_f = self.active_voxels[self.vv1_f].cpu()
    #    vv0_b = self.active_voxels[self.vv0_b].cpu()
    #    vv1_b = self.active_voxels[self.vv1_b].cpu()
    #    B = csr_matrix((torch.ones_like(vv0_f), (vv0_f, vv1_f)),
    #                   shape=(self.num_voxels, self.num_voxels))
    #    C = csr_matrix((torch.ones_like(vv0_b), (vv0_b, vv1_b)),
    #                   shape=(self.num_voxels, self.num_voxels))
    #    A = A + B + C

    num_graphs, graph_idx = connected_components(A, directed=False)
    if return_subgraphs:
        graph_indices = []
        for i in range(num_graphs):
            graph_indices.append(np.where(graph_idx == i)[0])
        return graph_indices
    else:
        return graph_idx
