import torch
from det3d.ops.primitives.primitives_cpu import (
    voxelization,
    voxel_graph,
    query_point_correspondence as query_corres,
)
from torch_scatter import scatter
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import time
from PytorchHashmap.torch_hash import HashTable

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
    R = torch.eye(3).repeat(r.shape[0], 1, 1)
    if valid_mask.any():
        r_valid = r[valid_mask]
        theta_valid = theta[valid_mask]
        k = r_valid / theta_valid.unsqueeze(-1)
        sint = theta_valid.sin().unsqueeze(-1).unsqueeze(-1)
        cost = theta_valid.cos().unsqueeze(-1).unsqueeze(-1)
        k_outer = k.unsqueeze(-1) @ k.unsqueeze(-2)
        R[valid_mask] = cost*torch.eye(3) + sint*cross_op(k) + (1-cost)*k_outer
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

class ARAPSolver:
    def __init__(self,
                 points,
                 voxels,
                 vp_edges,
                 voxel_size):
        self.ref_points = points
        self.ref_voxels = voxels
        self.vp0, self.vp1 = vp_edges
        self.p2v = torch.zeros(points.shape[0], dtype=torch.long)
        self.p2v[self.vp1] = self.vp0
        self.num_voxels = self.vp0.max()+1
        self.voxel_size = voxel_size
        self.lamb = 10

        # moving points and voxels
        self.points = points.clone()
        self.voxels = voxels.clone()
        
        # per-voxel transformation variables
        self.T = torch.eye(4).repeat(self.num_voxels, 1, 1)
        
        # static edges, voxel to voxel, same frame
        self.corres_voxel_size = voxel_size.clone()
        self.corres_voxel_size[:2] *= 7
        self.corres_voxel_size[2] *= 2
        
        self.vgraph_voxel_size = voxel_size.clone()
        self.vgraph_voxel_size[:3] *= 2
        self.vv0, self.vv1 = voxel_graph(
                                 self.ref_voxels, self.ref_voxels,
                                 self.vgraph_voxel_size, 0, 32).T.long()
        
        remove_loop_mask = (self.vv0 != self.vv1)
        self.vv0 = self.vv0[remove_loop_mask]
        self.vv1 = self.vv1[remove_loop_mask]
        self.ht = HashTable(2**21)

    def find_graphs(self):
        """Find connected components of the graph defined by
        self.vv and self.vv_f.

        Returns:
            graph_indices (list[np.ndarray])
        """
        A = csr_matrix((torch.ones_like(self.vv0), (self.vv0, self.vv1)),
                       shape=(self.num_voxels, self.num_voxels))
        B = csr_matrix((torch.ones_like(self.vv0_f), (self.vv0_f, self.vv1_f)),
                       shape=(self.num_voxels, self.num_voxels))
        A = A + B
        num_graphs, graph_idx = connected_components(A, directed=False)
        graph_indices = []
        for i in range(num_graphs):
            graph_indices.append(np.where(graph_idx == i)[0])
        return graph_indices

    def energy_reg(self, p, q, rt):
        f0, f1 = self.corres_f.T
        b0, b1 = self.corres_b.T
        # p = p + r X p + t
        rt = rt[self.p2v]
        p_new = p.clone()
        p_new[:, :3] = p_new[:, :3] + torch.cross(rt[:, :3], p_new[:, :3]) + rt[:, :3]
        self.energy['reg_vec'] = torch.zeros(self.num_voxels,
                                             dtype=torch.float32)
        energy = 0.5*self.weight['reg']*(p_new[f0, :3] - q[f1, :3]).square().sum(-1)
        self.energy['reg_vec'] += scatter(
                                      energy, self.p2v[f0],
                                      dim=0, reduce='sum',
                                      dim_size=self.num_voxels)
        energy = 0.5*self.weight['reg']*(p_new[b1, :3] - q[b0, :3]).square().sum(-1)
        self.energy['reg_vec'] += scatter(
                                      energy, self.p2v[b1],
                                      dim=0, reduce='sum',
                                      dim_size=self.num_voxels)
        return self.energy['reg_vec'].sum()

    def energy_all(self, p, q, rt):
        e_reg = self.energy_reg(p, q, rt)
        r, t = rt[:, :3], rt[:, 3:]
        tempT = self.T.clone()
        R = cross_op(r) #rodriguez(r)
        Ti = torch.eye(4).repeat(self.num_voxels, 1, 1)
        Ti[:, :3, :3] = R
        Ti[:, :3, 3] = t
        self.T = Ti @ self.T
        
        se0, se1 = self.edges0, self.edges1
        R, trans = self.T[:, :3, :3], self.T[:, :3, 3]
        R0, R1 = R[se0], R[se1]
        t0, t1 = trans[se0], trans[se1]
        w = self.w
        w1 = w.unsqueeze(-1)
        w11 = w1.unsqueeze(-1)
        rigid_energy = 0.5*(w11*(R0 - R1)).square().sum() \
                      +0.5*( w1*(t0 - t1)).square().sum()

        self.T = tempT

        return rigid_energy + e_reg
        

    def solve_reg(self, p, q, lamb=10):
        """Construct the problem w.r.t. registration energy

        Args:
            p: moving points
            q: reference points

        Returns:
            H (V, 6, 6): per voxel hessian block diagonal
            g (V, 6, 6): per voxel residual vectors
            energy: registration energy

        """
        
        H_diag = torch.zeros(self.num_voxels, 6, 6)
        g = torch.zeros(self.num_voxels, 6)
        self.corres_f = self.find_corres(
            q, p, self.corres_voxel_size, 1).T.cpu()
        self.corres_b = self.find_corres(
            p, q, self.corres_voxel_size, 1).T.cpu()
        #self.corres_f = self.ht.find_corres(q.cuda(), p.cuda(),
        #                              self.corres_voxel_size.cuda(), 1).T.cpu()
        #self.corres_b = self.ht.find_corres(p.cuda(), q.cuda(),
        #                              self.corres_voxel_size.cuda(), -1).T.cpu()
        #self.corres_f = query_corres(p, q, self.corres_voxel_size, 1).long()
        
        f0, f1 = self.corres_f.T
        b0, b1 = self.corres_b.T
        self.energy['reg_vec'] = torch.zeros(self.num_voxels,
                                             dtype=torch.float32)

        def sub_solve(p_xyz, q_xyz, aggr_index, H_diag, g):
            num_corres = aggr_index.shape[0]
            energy = 0.5*self.weight['reg']*(p_xyz - q_xyz).square().sum(-1)
            H1 = torch.cat([cross_op(p_xyz),
                            torch.eye(3).repeat(num_corres, 1, 1)],
                           dim=1) # [num_corres, 6, 3]
            H1H1T = H1 @ H1.transpose(-1, -2)
            
            H_diag += scatter(H1H1T.view(-1, 36), self.p2v[aggr_index],
                              reduce='sum', dim=0,
                              dim_size=self.num_voxels).view(-1, 6, 6)
            g1 = torch.cat([torch.cross(p_xyz, q_xyz-p_xyz), q_xyz - p_xyz],
                            dim=-1) # [num_corres, 6]
            g1 = scatter(g1, self.p2v[aggr_index], dim=0,
                         reduce='sum', dim_size=self.num_voxels)
            g += g1
            self.energy['reg_vec'] += scatter(
                                          energy, self.p2v[aggr_index],
                                          dim=0, reduce='sum',
                                          dim_size=self.num_voxels)
        
        """ Forward 
        p --> q, (f0, f1)
        aggregate on p
        """
        sub_solve(p[f0, :3], q[f1, :3], f0, H_diag, g)
        """ Backward
        q --> p, (b0, b1)
        aggregate on p
        """
        sub_solve(p[b1, :3], q[b0, :3], b1, H_diag, g)
        H_diag = H_diag * self.weight['reg']
        g = g * self.weight['reg']
        H_diag += torch.eye(6).repeat(self.num_voxels, 1, 1)*lamb

        return H_diag, g

    def solve_rigid(self, graph_index):
        """Construct local rigidity loss.

        Args:
            graph_index [N]: selected voxel indices

        Returns:
            H_diag [N, 6, 6]: containing diagonal of H
            spH (SparseMatrixPrep): containing sparse blocks of H
            g [N, 6]: containing the residual (linear) term

        """
        # construct sub-graph edges (se)
        self.selected[graph_index] = True
        mask = self.selected[self.edges0] & self.selected[self.edges1]
        se0, se1 = self.edges0[mask], self.edges1[mask]
        w = self.w[mask]
        self.selected[graph_index] = False
        graph_size = graph_index.shape[0]
        self.index_map[graph_index] = torch.arange(graph_size)
        se0, se1 = self.index_map[se0], self.index_map[se1]
        num_edges = se0.shape[0]
        
        """ local rigidity energy (se0, se1) """
        spH = SparseMatPrep(graph_size*6)
        R, trans = self.T[graph_index, :3, :3], self.T[graph_index, :3, 3]
        R0, R1 = R[se0], R[se1]
        t0, t1 = trans[se0], trans[se1]
        H_off_diag = torch.zeros(num_edges, 6, 6)
        H_diag = torch.zeros(graph_size, 6, 6)
        g = torch.zeros(graph_size, 6)

        w1 = w.unsqueeze(-1)
        w11 = w1.unsqueeze(-1)
        rigid_energy = 0.5*(w11*(R0 - R1)).square().sum() \
                      +0.5*( w1*(t0 - t1)).square().sum()

        # rotation terms
        for l in range(3):
            r0 = R0[:, :, l]
            H0 = torch.cat([cross_op(r0),
                            torch.zeros(num_edges, 3, 3)],
                           dim=1) # [num_edges, 6, 3]
            r1 = R1[:, :, l]
            H1 = torch.cat([cross_op(r1),
                            torch.zeros(num_edges, 3, 3)],
                           dim=1) # [num_edges, 6, 3]
            H0H0T = (H0 @ H0.transpose(-2, -1)) * w.unsqueeze(-1).unsqueeze(-1)
            H1H1T = (H1 @ H1.transpose(-2, -1)) * w.unsqueeze(-1).unsqueeze(-1)
            H0H1T = (H0 @ H1.transpose(-2, -1)) * w.unsqueeze(-1).unsqueeze(-1)
            H_diag += scatter(H0H0T.view(-1, 36), se0, reduce='sum',
                              dim=0, dim_size=graph_size).view(-1, 6, 6)
            H_diag += scatter(H1H1T.view(-1, 36), se1, reduce='sum',
                              dim=0, dim_size=graph_size).view(-1, 6, 6)
            H_off_diag += -H0H1T
            H0r10 = (H0 @ (r1 - r0).unsqueeze(-1)).squeeze(-1)
            H1r01 = (H1 @ (r0 - r1).unsqueeze(-1)).squeeze(-1)
            g += scatter(H0r10*w.unsqueeze(-1), se0, dim=0,
                         reduce='sum', dim_size=graph_size)
            g += scatter(H1r01*w.unsqueeze(-1), se1, dim=0,
                         reduce='sum', dim_size=graph_size)
        
        # translation terms
        H0 = torch.cat([cross_op(t0),
                        torch.eye(3).repeat(num_edges, 1, 1)],
                       dim=1) # [num_edges, 6, 3]
        H1 = torch.cat([cross_op(t1),
                        torch.eye(3).repeat(num_edges, 1, 1)],
                       dim=1) # [num_edges, 6, 3]
        H0H0T = (H0 @ H0.transpose(-2, -1)) * w.unsqueeze(-1).unsqueeze(-1)
        H1H1T = (H1 @ H1.transpose(-2, -1)) * w.unsqueeze(-1).unsqueeze(-1)
        H0H1T = (H0 @ H1.transpose(-2, -1)) * w.unsqueeze(-1).unsqueeze(-1)
        H_diag += scatter(H0H0T.view(-1, 36), se0, reduce='sum',
                          dim=0, dim_size=graph_size).view(-1, 6, 6)
        H_diag += scatter(H1H1T.view(-1, 36), se1, reduce='sum',
                          dim=0, dim_size=graph_size).view(-1, 6, 6)
        H_off_diag += -H0H1T
        H0t10 = (H0 @ (t1 - t0).unsqueeze(-1)).squeeze(-1)
        H1t01 = (H1 @ (t0 - t1).unsqueeze(-1)).squeeze(-1)
        g += scatter(H0t10*w.unsqueeze(-1), se0, dim=0,
                     reduce='sum', dim_size=graph_size)
        g += scatter(H1t01*w.unsqueeze(-1), se1, dim=0,
                     reduce='sum', dim_size=graph_size)
        spH.add_off_diag(H_off_diag, se0, se1)
        spH.add_off_diag(H_off_diag.transpose(-1, -2), se1, se0)
        self.energy['rigid'] += rigid_energy

        return H_diag, spH, g

    def visualize_corres(self):
        self.vis.curvenetwork('voxel-graph', self.voxels[:, :3], self.vedges)

    def find_corres(self, ref_p, p, vs, toffset):
        if False:
            return self.ht.find_corres(
                       ref_p.cuda(), p.cuda(), vs.cuda(), toffset).cpu().long()
        else:
            return query_corres(
                       p, ref_p, vs, toffset).T.long()

    def solve(self, max_iter=1000, lamb=0.1):
        """Solve the optimization.
        Args:
            max_iter (int)

        Returns:
            
        """
        self.ht = HashTable(2**21)
        p = self.points
        v = self.voxels
        q = self.ref_points
        ref_v = self.ref_voxels
        from det3d.core.utils.visualization import Visualizer
        self.vis = Visualizer([0.2, 0.2], [-75.2, -75.2], size_factor=1)

        ps_ref = self.vis.pointcloud('ref', q[:, :3])
        ps_refv = self.vis.pointcloud('ref-V', v[:, :3])
        ps_ref.add_scalar_quantity('frame', q[:, -1] % 2, enabled=True)
        ps_refv.add_scalar_quantity('frame', v[:, -1] % 2, enabled=True)

        self.energy = {}
        self.weight = dict(rigid=1, rigid_f = 1, reg=1.0)
        self.construct_time, self.solve_time = 0.0, 0.0
        for itr in range(max_iter):
            # registration energy
            # 0.5*(r; t)^T H (r; t) - (r; t)^T g
            self.energy['reg'] = 0.0
            self.energy['rigid'] = 0.0
            H_reg, g_reg = self.solve_reg(p, q, self.lamb)
            rt = torch.zeros(self.num_voxels, 6)
            
            # construct spatio-temporal voxel graph
            self.construct_time -= time.time()
            self.vv0_f, self.vv1_f = self.find_corres(
                ref_v, v, self.vgraph_voxel_size, 1)
            self.vv1_b, self.vv0_b = self.find_corres(
                v, ref_v, self.vgraph_voxel_size, -1)
            #self.ht.find_corres(
            #    ref_v, v, self.vgraph_voxel_size, 1).cpu().long()
            #self.vv1_b, self.vv0_b = self.ht.find_corres(
            #    v, ref_v, self.vgraph_voxel_size, 1).cpu().long()
            #query_corres(
            #    v, ref_v, self.vgraph_voxel_size, 1).T.long()
            #self.vv1_b, self.vv0_b = query_corres(
            #    ref_v, v, self.vgraph_voxel_size, -1).T.long()
            self.edges0 = torch.cat([self.vv0, self.vv0_f, self.vv0_b], dim=-1)
            self.edges1 = torch.cat([self.vv1, self.vv1_f, self.vv1_b], dim=-1)
            self.w = torch.ones(self.edges0.shape[0], dtype=torch.float32)
            self.w[:self.vv0.shape[0]] = self.weight['rigid']
            self.w[self.vv0.shape[0]:] = self.weight['rigid_f']
            self.vedges = torch.stack([self.edges0, self.edges1], dim=-1)
            self.selected = torch.zeros(self.num_voxels).bool()
            self.index_map = torch.zeros(self.num_voxels).long()
            self.construct_time += time.time()

            # solve for each connected components
            graph_indices = self.find_graphs()
            graph_size = 0
            graph_index = np.zeros((0))
            #graph_indices = [np.arange(self.num_voxels)]
            # loop through each graphs
            for g, _graph_index in enumerate(graph_indices):
                if graph_index.shape[0] > 200000:
                    assert False
                graph_index = _graph_index
                #graph_index = np.concatenate(
                #    [graph_index, _graph_index], axis=0)
                _graph_size = _graph_index.shape[0]
                #graph_size += _graph_size
                graph_size = _graph_size
                #if (graph_size < 100) and (g != len(graph_indices)-1):
                #    continue
                self.energy['reg'] += self.energy['reg_vec'][graph_index].sum() 
                self.construct_time -= time.time()
                H_diag, spH, g = self.solve_rigid(graph_index)
                H_diag += H_reg[graph_index]
                g += g_reg[graph_index]
                
                # solve linear system
                spH.add_block_diag(H_diag)
                H = spH.output()
                #Hd = H.todense()
                #print(np.linalg.eigvalsh(Hd)[-1])
                #np.save(f'pp{itr}.npy', self.pp)
                #np.save(f'Hreg{itr}.npy', H_reg[graph_index])
                #np.save(f'greg{itr}.npy', g_reg[graph_index])
                #np.save(f'H{itr}.npy', H.todense())
                #np.save(f'g{itr}.npy', g)
                #np.save(f'e0{itr}.npy', self.vv0)
                #np.save(f'e1{itr}.npy', self.vv1)
                self.construct_time += time.time()
                
                self.solve_time -= time.time()
                #print(f'graph_size={graph_size}')
                rt_g = scipy.sparse.linalg.spsolve(
                           H, g.reshape(-1).numpy()).reshape(-1, 6)
                self.solve_time += time.time()
                rt[graph_index] = torch.tensor(rt_g, dtype=torch.float32) 
                graph_size = 0
                graph_index = np.zeros((0))
            
            # apply transformations
            r, t = rt[:, :3], rt[:, 3:]
            R = rodriguez(r)
            Ti = torch.eye(4).repeat(self.num_voxels, 1, 1)
            Ti[:, :3, :3] = R
            Ti[:, :3, 3] = t
            self.T = Ti @ self.T
            p[:, :3] = (p[:, :3].unsqueeze(-2) @ R[self.p2v].transpose(-1, -2)).squeeze(-2) + t[self.p2v]
            v[:, :3] = (v[:, :3].unsqueeze(-2) @ R.transpose(-1, -2)).squeeze(-2) + t
            ps_mp = self.vis.pointcloud('moving', p[:, :3])
            ps_mp.add_scalar_quantity('frame', p[:, -1] % 2, enabled=True)
            ps_mv = self.vis.pointcloud('moving-voxels', v[:, :3])
            ps_mv.add_scalar_quantity('frame', v[:, -1] % 2, enabled=True)
            message = f'iter={itr}, construct={self.construct_time:.4f}, '
            message += f'solve={self.solve_time:.4f}, '
            message += f'edgesize={self.edges0.shape[0]}, '
            message += f'rigid={self.energy["rigid"]:.4f}, '
            message += f'reg={self.energy["reg"]:.4f}, '
            message += f'total={(self.energy["reg"]+self.energy["rigid"]):.4f}'
            print(message)
            if (itr+1) % 10 == 0:
                import ipdb; ipdb.set_trace()
                pass 

