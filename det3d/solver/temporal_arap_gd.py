import torch
from det3d.ops.primitives.primitives_cpu import (
    voxelization,
    voxel_graph,
    query_point_correspondence as query_corres,
)
from torch_scatter import scatter
import numpy as np
import scipy
import time
from PytorchHashmap.torch_hash import HashTable
from .geometry_utils import *
from collections import defaultdict

class ARAPGDSolver:
    def __init__(self,
                 points,
                 normals,
                 voxels,
                 vp_edges,
                 voxel_size,
                 annos,
                 lamb=10):
        self.device = 'cpu'
        self.dtype = torch.float64
        self.lamb = lamb
        self.voxel_size = voxel_size.to(self.dtype).to(self.device)
        self.ref_points = points.to(self.device).to(self.dtype)
        self.ref_voxels = voxels.to(self.device).to(self.dtype)
        self.ref_normals = normals.to(self.device).to(self.dtype)
        self.vp0, self.vp1 = vp_edges.to(self.device)
        self.p2v = torch.zeros(points.shape[0], dtype=torch.long,
                               device=self.device)
        self.corners = np.concatenate(annos['corners'], axis=0)
        self.boxes = np.concatenate(annos['boxes'], axis=0)
        self.classes = np.concatenate(annos['classes'], axis=0)
        self.object_tokens = np.concatenate(annos['tokens'], axis=0)
        self.frame_ids = np.concatenate(annos['frame_ids'], axis=0)
        self.anno_T = annos['transformations']
       
        # find components
        if False:
            self.vv0, self.vv1 = voxel_graph(
                                     voxels, voxels, self.voxel_size.float().cpu(),
                                     0, 64).T.long().to(self.device)
            remove_loop_mask = (self.vv0 != self.vv1)
            self.vv0 = self.vv0[remove_loop_mask]
            self.vv1 = self.vv1[remove_loop_mask]
            self.num_voxels = voxels.shape[0]
            graph_index = torch.tensor(self.find_graphs(temporal=False)
                                       ).long().to(self.devie)
            self.num_voxels = graph_index.max().item()+1
            self.ref_voxels = scatter(self.ref_voxels, graph_index, dim=0,
                                      reduce='mean', dim_size=self.num_voxels)
            self.p2v[self.vp1] = graph_index[self.vp0]
        else:
            self.p2v[self.vp1] = self.vp0
            self.num_voxels = self.vp0.max()+1
        
        self.ht_size = 1
        while self.ht_size < points.shape[0]*2:
            self.ht_size *= 2
        self.ht = HashTable(self.ht_size, dtype=self.dtype)

        # moving points and voxels
        self.bvoxels = self.ref_voxels.clone()
        self.fvoxels = self.ref_voxels.clone()
        self.bpoints = self.ref_points.clone()
        self.fpoints = self.ref_points.clone()
        self.bnormals = self.ref_normals.clone()
        
        # static edges, voxel to voxel, same frame
        self.corres_voxel_size = self.voxel_size.clone()
        self.corres_voxel_size[:2] *= 7
        self.corres_voxel_size[2] *= 2
        
        self.vgraph_voxel_size = self.voxel_size.clone()
        self.vgraph_voxel_size[:3] *= 2
        
        self.tgraph_voxel_size = self.voxel_size.clone()
        self.tgraph_voxel_size[:3] *= 7
        
        self.vv0, self.vv1 = voxel_graph(
                                 self.ref_voxels.cpu().float(),
                                 self.ref_voxels.cpu().float(),
                                 self.vgraph_voxel_size.float().cpu(),
                                 0, 32).T.long().to(self.device)
        remove_loop_mask = (self.vv0 != self.vv1)
        self.vv0 = self.vv0[remove_loop_mask]
        self.vv1 = self.vv1[remove_loop_mask]
            
        # per-voxel transformation variables
        self.T = torch.eye(4, dtype=self.dtype,
                           device=self.device).repeat(self.num_voxels, 1, 1)

        self.construct_time, self.solve_time = 0.0, 0.0
        self.energy = dict(rigid=0.0, reg=0.0)
        self.weight = dict(rigid=10, rigid_f=10, reg_point=0.0, reg_plane=1)
        from det3d.core.utils.visualization import Visualizer
        self.vis = Visualizer([0.2, 0.2], [-75.2, -75.2], size_factor=1)
        
        if False:
            # cheating
            torch.save((self.p2v == 38), '38.pt')
            torch.save((self.p2v == 206), '206.pt')
            
            #self.vv0_f, self.vv1_f = self.ht.find_corres(self.ref_voxels, self.voxels, 
            #                                             self.vgraph_voxel_size, 1)
            #graph_indices = self.find_graphs()

            #idx = self.p2v[79424].cpu().item() # 13382, 81996
            #idx = np.where([(idx in g) for g in graph_indices])[0].item()
            #graph_index = graph_indices[idx]
            #mask = (self.p2v < 0)
            #for gi in graph_index:
            #    mask |= (self.p2v == gi)
            #self.points = self.points[mask]
            #self.ref_points = self.ref_points[mask]
            #self.ref_normals = self.ref_normals[mask]
            #self.voxels = self.voxels[graph_index]
            #self.ref_voxels = self.ref_voxels[graph_index]
            #voxel_map = torch.zeros(self.num_voxels).long().to(self.device) - 1
            #voxel_map[graph_index] = torch.arange(graph_index.shape[0]).long().to(self.device)
            #self.p2v = voxel_map[self.p2v]
            #self.p2v = self.p2v[self.p2v != -1]
            #
            #self.vv0, self.vv1 = voxel_graph(
            #                         self.voxels.cpu(), self.voxels.cpu(),
            #                         self.vgraph_voxel_size.cpu(),
            #                         0, 32).T.long().to(self.devicself.devicee)

            #remove_loop_mask = (self.vv0 != self.vv1)
            #self.vv0 = self.vv0[remove_loop_mask]
            #self.vv1 = self.vv1[remove_loop_mask]
            #self.num_voxels = graph_index.shape[0]
            #self.T = torch.eye(4, dtype=self.dtype,
            #                   device=self.device).repeat(self.num_voxels, 1, 1)


    @torch.no_grad()
    def solve_reg(self):
        """Construct the problem w.r.t. registration energy

        Args:
            p: moving points
            q: reference points
            n: reference normals

        """
        def sub_solve(p_xyz, q_xyz, n_xyz, aggr_index, H_diag, g, aggr='mean', sigma2=1.0**2):
            num_corres = aggr_index.shape[0]
            diff = q_xyz - p_xyz
            diff_n = (diff * n_xyz).sum(-1)
            pXn = torch.cross(p_xyz, n_xyz)
            wpoint = self.weight['reg_point']*(sigma2/(diff.square().sum(-1)+sigma2))
            self.wpoint.append(wpoint)

            wpoint1 = wpoint.unsqueeze(-1)
            wpoint11 = wpoint1.unsqueeze(-1)
            wplane = self.weight['reg_plane']*(sigma2/(diff_n.square()+sigma2))
            self.wplane.append(wplane)
            wplane1 = wplane.unsqueeze(-1)
            wplane11 = wplane1.unsqueeze(-1)
            energy = 0.5*wpoint*diff.square().sum(-1)
            energy += 0.5*wplane*diff_n.square()
            p2v = self.p2v[self.active_points]
            p2v = self.active_voxel_map[p2v]

            num_active_voxels = self.active_voxels.shape[0]
            # computation per correspondence
            H1 = torch.cat([cross_op(p_xyz),
                            torch.eye(3).repeat(num_corres, 1, 1).to(p_xyz)],
                           dim=1) # [num_corres, 6, 3]
            h1 = torch.cat([pXn, n_xyz], dim=1)
            H1H1T = (wpoint11 * H1) @ H1.transpose(-1, -2)
            H1H1T += h1.unsqueeze(-1) @ h1.unsqueeze(-2) * wplane11
            
            g1 = torch.cat([torch.cross(p_xyz, diff), diff],
                            dim=-1) * wpoint1 # [num_corres, 6]
            g1 += h1 * diff_n.unsqueeze(-1) * wplane1 
            
            # scatter to voxels
            H_diag += scatter(H1H1T.view(-1, 36), p2v[aggr_index],
                              reduce=aggr, dim=0,
                              dim_size=num_active_voxels).view(-1, 6, 6)
            g += scatter(g1, p2v[aggr_index], dim=0,
                         reduce=aggr, dim_size=num_active_voxels)
            self.energy['reg_vec'] += scatter(
                                          energy, p2v[aggr_index],
                                          dim=0, reduce=aggr,
                                          dim_size=num_active_voxels) 
       
        self.construct_time -= time.time()
        f0, f1 = self.corres_f
        b0, b1 = self.corres_b
        self.energy['reg_vec'] = torch.zeros(self.active_voxels.shape[0],
                                             dtype=self.dtype,
                                             device=self.device)
        H_diag = self.H_diag[self.active_voxels]
        g = self.g[self.active_voxels]
        self.wpoint, self.wplane = [], []

        """ Forward 
        (R, t) p --> q, (f0, f1)
        aggregate on p(f0)
        """
        sub_solve(self.fpoints[f0, :3], self.ref_points[f1, :3],
                  self.ref_normals[f1, :3], f0, H_diag, g)
        """ Backward
        (R, t)^-1 p --> q, (b0, b1)
        <=> p --> (R, t) q, (b0, b1)
        <=> (R, t) q --> p, (b1, b0)
        aggregate on p(b0)
        """
        sub_solve(self.ref_points[b1, :3], self.bpoints[b0, :3],
                  self.bnormals[b0, :3], b0, H_diag, g)
        self.wplane = torch.cat(self.wplane, dim=0)
        self.H_diag[self.active_voxels] = H_diag
        self.g[self.active_voxels] = g
        self.energy['reg'] = self.energy['reg_vec'].sum()
        self.construct_time += time.time()

    def solve_rigid(self):
        """Construct local rigidity loss.

        Args:
            graph_index [N]: selected voxel indices

        Returns:
            H_diag [N, 6, 6]: containing diagonal of H
            spH (SparseMatrixPrep): containing sparse blocks of H
            g [N, 6]: containing the residual (linear) term

        """
        # construct sub-graph edges (se)
        self.construct_time -= time.time()
        se0, se1 = self.edges0, self.edges1
        w = self.w
        graph_size = self.active_voxels.shape[0]
        num_edges = se0.shape[0]
       
        T = self.T[self.active_voxels]
        """ local rigidity energy (se0, se1) """
        R, trans = T[:, :3, :3], T[:, :3, 3]

        R0, R1 = R[se0], R[se1]
        t0, t1 = trans[se0], trans[se1]
        H_off_diag = torch.zeros(num_edges, 6, 6).to(self.device)

        w1 = w.unsqueeze(-1)
        w11 = w1.unsqueeze(-1)

        rigid_energy_vec = 0.5*(w11*(R0-R1)).cpu().square().sum(-1).sum(-1)
        rigid_energy_vec += 0.5*(w1*(t0 - t1)).cpu().square().sum(-1)
        edges = torch.stack([se0, se1], dim=-1).cpu()
        ps_rigid = self.vis.curvenetwork(
                       'rigid graph',
                       self.ref_voxels[self.active_voxels, :3].cpu(), edges)
        ps_rigid.add_scalar_quantity('rigid loss',
                                     rigid_energy_vec.cpu().numpy(),
                                     defined_on='edges', enabled=True)
        rigid_energy = rigid_energy_vec.sum()
        
        H_diag = self.H_diag[self.active_voxels]
        g = self.g[self.active_voxels]
        # rotation terms
        for l in range(3):
            r0 = R0[:, :, l]
            H0 = torch.cat([cross_op(r0),
                            torch.zeros(num_edges, 3, 3).to(r0)],
                           dim=1) # [num_edges, 6, 3]
            r1 = R1[:, :, l]
            H1 = torch.cat([cross_op(r1),
                            torch.zeros(num_edges, 3, 3).to(r1)],
                           dim=1) # [num_edges, 6, 3]
            H0H0T = (H0 @ H0.transpose(-2, -1)) * w11
            H1H1T = (H1 @ H1.transpose(-2, -1)) * w11
            H0H1T = (H0 @ H1.transpose(-2, -1)) * w11
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
                        torch.eye(3).repeat(num_edges, 1, 1).to(t0)],
                       dim=1) # [num_edges, 6, 3]
        H1 = torch.cat([cross_op(t1),
                        torch.eye(3).repeat(num_edges, 1, 1).to(t1)],
                       dim=1) # [num_edges, 6, 3]
        H0H0T = (H0 @ H0.transpose(-2, -1)) * w11
        H1H1T = (H1 @ H1.transpose(-2, -1)) * w11
        H0H1T = (H0 @ H1.transpose(-2, -1)) * w11
        H_diag += scatter(H0H0T.view(-1, 36), se0, reduce='sum',
                               dim=0, dim_size=graph_size).view(-1, 6, 6)
        H_diag += scatter(H1H1T.view(-1, 36), se1, reduce='sum',
                               dim=0, dim_size=graph_size).view(-1, 6, 6)
        H_off_diag += -H0H1T
        H0t10 = (H0 @ (t1 - t0).unsqueeze(-1)).squeeze(-1)
        H1t01 = (H1 @ (t0 - t1).unsqueeze(-1)).squeeze(-1)
        g += scatter(H0t10*w1, se0, dim=0,
                          reduce='sum', dim_size=graph_size)
        g += scatter(H1t01*w1, se1, dim=0,
                          reduce='sum', dim_size=graph_size)
        self.energy['rigid'] += rigid_energy
        self.H_diag[self.active_voxels] = H_diag
        self.g[self.active_voxels] = g
        self.construct_time += time.time()

        return H_off_diag

    def build_rigid_graph(self, sigma=1.0):
        """Build dynamic spatio-temporal voxel graph
            
        """
        # find edges
        self.construct_time -= time.time()
        sigma2 = sigma*sigma
        bv = self.bvoxels[self.active_voxels]
        ref_v = self.ref_voxels[self.active_voxels]
        self.vv1_b, self.vv0_b = self.ht.find_corres(
                                     ref_v, bv, self.tgraph_voxel_size, -1)
        fv = self.fvoxels[self.active_voxels]
        self.vv0_f, self.vv1_f = self.ht.find_corres(
                                     ref_v, fv, self.tgraph_voxel_size, 1)
        mask = self.is_active_voxels[self.vv0] | self.is_active_voxels[self.vv1]
        vv0, vv1 = self.vv0, self.vv1
        #voxel_graph(v.cpu().float(), v.cpu().float(), self.vgraph_voxel_size.cpu(),
        #                       0, 32).T.long().to(self.device)

        self.edges0 = torch.cat([vv0, self.vv0_f, self.vv0_b], dim=-1).to(self.device)
        self.edges1 = torch.cat([vv1, self.vv1_f, self.vv1_b], dim=-1).to(self.device)
        self.w = torch.ones(self.edges0.shape[0], dtype=self.dtype).to(self.device)

        # spatial voxel graph weights
        dist_vv = (fv[vv0] - fv[vv1]).square().sum(-1)
        w_vv = (-dist_vv/sigma2).exp()
        self.w[:vv0.shape[0]] = self.weight['rigid']*w_vv
       
        # temporal voxel graph weights
        dist_vvf = (fv[self.vv0_f] - ref_v[self.vv1_f]).square().sum(-1)
        w_vvf = (-dist_vvf/sigma2).exp()
        dist_vvb = (ref_v[self.vv1_b] - fv[self.vv0_b]).square().sum(-1)
        w_vvb = (-dist_vvb/sigma2).exp()
        self.w[vv0.shape[0]:] = torch.cat([w_vvf, w_vvb], dim=0)*self.weight['rigid_f']
        self.construct_time += time.time()

    def visualize_corres(self):
        points = torch.cat([self.fpoints, self.ref_points], dim=0)
        edges = self.corres_f #torch.cat([self.corres_f, self.corres_b], dim=-1)
        edges[1] += self.fpoints.shape[0]
        ps_corres = self.vis.curvenetwork('corres-f', points[:, :3].cpu(), edges.T.cpu())
        
        points = torch.cat([self.bpoints, self.ref_points], dim=0)
        edges = self.corres_b #torch.cat([self.corres_f, self.corres_b], dim=-1)
        edges[1] += self.fpoints.shape[0]
        ps_corres = self.vis.curvenetwork('corres-b', points[:, :3].cpu(), edges.T.cpu())
        ps_corres.add_scalar_quantity('weight', self.wplane.cpu().numpy(), defined_on='edges')

    def transform(self, _p, T):
        R, t = T[:, :3, :3], T[:, :3, 3]
        RT = R.transpose(-1, -2)
        p = _p.clone()
        p[:, :3] = (p[:, :3].unsqueeze(-2) @ RT).squeeze(-2) + t
        return p

    def rotate(self, _p, R):
        RT = R.transpose(-1, -2)
        p = _p.clone()
        p[:, :3] = (p[:, :3].unsqueeze(-2) @ RT).squeeze(-2)

        return p

    def update_pos(self, rt):
        r, t = rt[:, :3], rt[:, 3:]
        R = rodriguez(r)
        Ti = torch.eye(4).repeat(self.num_voxels,
                                 1, 1).to(self.device).to(self.dtype)
        Ti[:, :3, :3] = R
        Ti[:, :3, 3] = t
        self.T = Ti @ self.T
        self.fpoints = self.transform(self.ref_points, self.T[self.p2v])
        self.bpoints = self.transform(self.ref_points, inverse(self.T)[self.p2v])
        self.bnormals = self.rotate(self.ref_normals, self.T[self.p2v, :3, :3])
        self.fvoxels = self.transform(self.ref_voxels, self.T)
        self.bvoxels = self.transform(self.ref_voxels, inverse(self.T))

    def visualize_frames(self, name, points, normals, voxels):
        ps_ref = self.vis.pointcloud(f'{name}-points', points[:, :3].cpu().numpy())
        ps_ref.add_vector_quantity(f'normals', normals[:, :3].cpu().numpy())
        ps_ref.add_scalar_quantity(f'frame mod 2', points[:, -1].cpu().numpy() % 2, enabled=True)
        ps_ref.add_scalar_quantity(f'frame', points[:, -1].cpu().numpy(), enabled=True)
        
        ps_refv = self.vis.pointcloud(f'{name}-voxels', voxels[:, :3].cpu().numpy())
        ps_refv.add_scalar_quantity(f'frame mod 2', voxels[:, -1].cpu().numpy() % 2, enabled=True)
        ps_refv.add_scalar_quantity(f'frame', voxels[:, -1].cpu().numpy(), enabled=False)
        ps_box = self.vis.boxes(f'{name}-boxes', self.corners, self.classes)

    def solve_dense(self):
        num_points = scatter(torch.ones(self.fpoints.shape[0]), self.p2v,
                             reduce='sum', dim=0, dim_size=self.num_voxels)
        dense_voxel_indices = torch.where(num_points > 5)[0]
        dense_point_indices = torch.where(num_points[self.p2v] > 5)[0]
        ref_points = self.ref_points[dense_point_indices]
        ref_normals = self.ref_normals[dense_point_indices]
        ref_voxels = self.ref_voxels[dense_voxel_indices]
        self.vis.boxes('boxes', self.corners, self.classes)
        self.visualize_object_tracking()
        self.visualize_frames('full', self.ref_points, self.ref_normals,
                              self.ref_voxels)
        self.visualize_frames('dense', ref_points, ref_normals, ref_voxels)

    def visualize_object_tracking(self):
        from det3d.core.utils.visualization import Visualizer
        from det3d.core.bbox import box_np_ops
        self.vis = Visualizer([0.2, 0.2], [-75.2, -75.2])
        object_traces = defaultdict(lambda: {'boxes': [], 'labels': [],
                                             'corners': [], 'frame_ids': []
                                            })
        for i, (box, corner, label, token) in enumerate(zip(\
                self.boxes, self.corners, self.classes, self.object_tokens)):
            trace = object_traces[token]
            trace['boxes'].append(box)
            trace['labels'].append(label)
            trace['corners'].append(corner)
            trace['frame_ids'].append(self.frame_ids[i])
            object_traces[token] = trace
        self.visualize_frames('all', self.ref_points, self.ref_normals, self.ref_voxels)
        import ipdb; ipdb.set_trace()
        for token, trace in object_traces.items():
            boxes = trace['boxes']
            corners = trace['corners']
            boxes = np.stack(boxes, axis=0)
            corners = np.stack(corners, axis=0)
            frame_ids = np.array(trace['frame_ids'])
            labels = np.array(trace['labels']).reshape(-1)
            self.vis.boxes(f'token-trace', corners, labels)
            points_synced = []
            for i in range(boxes.shape[0]):
                frame_id = frame_ids[i]
                Ti = self.anno_T[frame_id]
                mask = self.ref_points[:, -1] == frame_id
                surfaces = box_np_ops.corner_to_surfaces_3d(corners[i:i+1])
                indices = box_np_ops.points_in_convex_polygon_3d_jit(
                        self.ref_points[mask, :3].numpy(), surfaces)[:, 0]
                points = self.ref_points[mask][indices].clone()
                points[:, :3] = (points[:, :3] - Ti[:3, 3]) @ Ti[:3, :3]
                points_back = box_np_ops.points_to_rbbox_system(points[:, :3], boxes[i:i+1])
                corners_back = box_np_ops.points_to_rbbox_system(corners[i], boxes[i:i+1])
                
                points_synced.append(points_back)
                #self.vis.boxes(f'box-{token}-frame-{frame_id}', corners_back.reshape(-1, 8, 3), labels[i:i+1])
            points_synced = np.concatenate(points_synced, axis=0)
            self.vis.pointcloud(f'token-points', points_synced[:, :3])
            self.vis.show()
            import ipdb; ipdb.set_trace()

    def solve(self, max_iter=10000):
        """solve the optimization.
        
        args:
            max_iter (int)

        returns:
            
        """
        self.solve_dense()
        #q = self.ref_points
        #ref_n = self.ref_normals
        #ref_v = self.ref_voxels
        #self.is_active_voxels = torch.tensor(True).repeat(
        #                            self.num_voxels).to(self.device)
        #self.is_active_graph = self.is_active_voxels.clone()
        #self.active_voxels = torch.where(self.is_active_voxels)[0]
        #self.active_voxel_map = torch.zeros(self.num_voxels
        #                                    ).long().to(self.device)
        #self.active_voxel_map[self.active_voxels] = \
        #        torch.arange(self.active_voxels.shape[0],
        #                     dtype=torch.long).to(self.device)
        #self.active_points = torch.where(
        #                         self.is_active_voxels[self.p2v])[0]
        #rt = torch.zeros(self.num_voxels, 6).to(self.device).to(self.dtype)
        #self.H_diag = torch.zeros(self.num_voxels, 6, 6,
        #                          dtype=self.dtype, device=self.device)
        #self.g = torch.zeros(self.num_voxels, 6,
        #                     dtype=self.dtype, device=self.device)

        # visualization
        return 
        H0 = torch.diag(torch.tensor([10,10,10,1,1,10], dtype=torch.float32)
                        ).repeat(self.num_voxels, 1, 1)

        # solve
        for itr in range(max_iter):
            self.energy = dict(rigid=0.0, reg=0.0)
            self.H_diag[:] = 0.0
            self.g[:] = 0.0
            self.H_diag += H0.to(self.device)*self.lamb
            rt[:] = 0.0

            # build dynamic graph
            self.construct_time -= time.time()
            self.corres_f = self.ht.find_corres(
                q, self.fpoints[self.active_points], self.corres_voxel_size, 1)
            self.corres_b = self.ht.find_corres(
                q, self.bpoints[self.active_points], self.corres_voxel_size, -1)
            self.construct_time += time.time()
            self.build_rigid_graph()
            
            # start constructing quadratic problem
            self.solve_reg()
            H_off_diag = self.solve_rigid()
            #graph_index = torch.tensor(self.find_graphs()).long().to(self.device)
            self.solve_time -= time.time()
            spH = SparseMatPrep(self.num_voxels * 6)
            spH.add_off_diag(H_off_diag.cpu(), self.edges0.cpu(), self.edges1.cpu())
            spH.add_off_diag(H_off_diag.cpu().transpose(-1, -2),
                             self.edges1.cpu(), self.edges0.cpu())
            spH.add_block_diag(self.H_diag.cpu())
            H = spH.output()
            rt = scipy.sparse.linalg.spsolve(H, self.g.view(-1).cpu().numpy())
            rt = torch.tensor(rt.reshape(-1, 6), dtype=self.dtype, device=self.device)
            self.solve_time += time.time()

            #if True:
            #    self.solve_time -= time.time()
            #    e0, e1 = self.edges0, self.edges1
            #    num_active_voxels = self.active_voxels.shape[0]
            #    rt = torch.zeros(self.num_voxels, 6, dtype=torch.float,
            #                     device=self.device)
            #    H_diag = self.H_diag[self.active_voxels]
            #    g = self.g[self.active_voxels]
            #    rt_act = rt[self.active_voxels]
            #    y = rt_act.clone()
            #    lr = 5e-6
            #    last_lamb = 0
            #    for inner_itr in range(1000):
            #        rt_act1 = rt_act.unsqueeze(-1)
            #        Q1 = (rt_act1[e1] * (H_off_diag @ rt_act1[e0])).sum()
            #        Q0 = 0.5*((H_diag @ rt_act1) * rt_act1).sum()
            #        Q2 = (g * rt_act).sum()
            #        Q = Q0 + Q1 - Q2
            #        grad = -g
            #        grad += scatter(
            #                    (H_off_diag @ rt_act1[e0]).squeeze(-1),
            #                    e1, dim=0, dim_size=num_active_voxels,
            #                    reduce='sum')
            #        grad += scatter(
            #                    (H_off_diag.transpose(-2, -1)
            #                        @ rt_act1[e1]).squeeze(-1),
            #                    e0, dim=0, dim_size=num_active_voxels,
            #                    reduce='sum')
            #        grad += (H_diag @ rt_act1).squeeze(-1)
            #        
            #        new_lamb = 0.5*(1 + np.sqrt(1+4*last_lamb**2))
            #        gamma = (1.0 - last_lamb) / new_lamb
            #        ysp = rt_act - lr*grad
            #        rt_act = (1-gamma)*ysp + gamma*y
            #        last_lamb = new_lamb
            #        y = ysp
            #        gradnorm = grad.norm(p=2)
            #    voxel_edges = torch.stack([self.edges0, self.edges1], dim=-1)
            #    self.vis.curvenetwork('voxel-graph',
            #                          v[self.active_voxels, :3].cpu().numpy(),
            #                          voxel_edges.cpu().numpy())
            #    rt[self.active_voxels] = rt_act
            #    self.solve_time += time.time()
            #    rtnorm = rt.cpu().norm(p=2, dim=-1)

            #    #self.active_voxels = torch.where(rtnorm > 1e-3)[0]
            #    #self.is_active_graph[:] = False
            #    #self.is_active_graph[graph_index[self.active_voxels]] = True
            #    #self.is_active_voxels[:] = False
            #    #self.is_active_voxels[self.active_voxels] = self.is_active_graph[graph_index[self.active_voxels]]
            #    #self.active_voxel_map[:] = -1
            #    #self.active_voxel_map[self.active_voxels] = \
            #    #        torch.arange(self.active_voxels.shape[0],
            #    #                     dtype=torch.long).to(self.device)
            #    #self.active_points = torch.where(
            #    #                         self.is_active_voxels[self.p2v])[0]

            # apply transformations
            self.construct_time -= time.time()
            self.update_pos(rt)

            # visualization
            ps_mp = self.vis.pointcloud('moving-f', self.fpoints[:, :3].cpu().numpy())
            ps_mp.add_scalar_quantity('frame', self.fpoints[:, -1].cpu().numpy() % 2, enabled=True)
            ps_mp.add_scalar_quantity('active', self.is_active_voxels[self.p2v].cpu().float(), enabled=True)
            ps_mp.add_scalar_quantity('mvdist', (self.fpoints - self.ref_points).norm(p=2, dim=-1).cpu(), enabled=True)
            
            ps_mp = self.vis.pointcloud('moving-b', self.bpoints[:, :3].cpu().numpy())
            ps_mp.add_scalar_quantity('frame', self.bpoints[:, -1].cpu().numpy() % 2, enabled=True)
            ps_mp.add_scalar_quantity('active', self.is_active_voxels[self.p2v].cpu().float(), enabled=True)
            ps_mp.add_scalar_quantity('mvdist', (self.bpoints - self.ref_points).norm(p=2, dim=-1).cpu(), enabled=True)
            
            ps_mv = self.vis.pointcloud('moving-voxels', self.fvoxels[:, :3].cpu().numpy(), radius=8e-4)
            ps_mv.add_scalar_quantity('frame', self.fvoxels[:, -1].cpu().numpy() % 2, enabled=True)
            ps_mv.add_scalar_quantity('reg loss', self.energy['reg_vec'].cpu().sqrt().numpy(), enabled=True)

            self.construct_time += time.time()
            message = f'iter={itr}, construct={self.construct_time:.4f}, '
            message += f'solve={self.solve_time:.4f}, '
            message += f'act={self.active_voxels.shape[0]}, '
            message += f'rt={rt.norm(p=2, dim=-1).mean():.4f}, '
            message += f'rigid={self.energy["rigid"]:.8f}, '
            message += f'reg={self.energy["reg"]:.8f}, '
            message += f'total={(self.energy["reg"]+self.energy["rigid"]):.8f}'
            print(message)
            if (itr+1) % 100 == 0:
                import ipdb; ipdb.set_trace()
                pass

