from ..registry import PIPELINES
from collections import defaultdict
import numpy as np
import pickle
import os
from torch_scatter import scatter
from torch_cluster import radius_graph
import torch
import scipy
from scipy.sparse import csr_matrix
from det3d.ops.primitives.primitives_cpu import (
    voxelization, voxel_graph,
    query_point_correspondence as query_corres,
)
from det3d.structures import Sequence

@PIPELINES.register_module
class EstimateMotionMask(object):
    def __init__(self,
                 point_cloud_range=None,
                 debug=True,
                 ):
        """
        Args:
            point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max)
        """
        self.counter = 0
        self.stats = [0, 0, 0]
        if point_cloud_range is not None:
            self.pc_range = point_cloud_range
        self.debug = debug

    def undersegment(self, sweep_points, radius=1.0):
        """Compute under-segments of each frame.

        Args:
            sweep_points (list[np.ndarray]): frames of length N
            radius: for computing clusters

        Returns:
            num_clusters (N): number of clusters each frame
            cluster_ids (list[np.ndarray]): cluster indices per frame
        """
        num_clusters, cluster_ids = [], []
        for i, _points in enumerate(sweep_points):
            points = torch.tensor(_points)
            e0, e1 = radius_graph(points, radius, loop=False,
                                  max_num_neighbors=300).numpy()
            adj_matrix = csr_matrix(
                (np.ones_like(e0), (e0, e1)),
                shape=(points.shape[0], points.shape[0]))
            num_comp, comp_labels = \
                scipy.sparse.csgraph.connected_components(
                    adj_matrix, directed=False)
            
            num_clusters.append(num_comp)
            cluster_ids.append(comp_labels)

        return num_clusters, cluster_ids

    def motion_estimate(self, seq, voxel_size=[0.6, 0.6, 0.6, 1],
                        max_iter=10000):
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        points, normals, voxels, vp_edges = \
            seq.temporal_voxelization(voxel_size)

        # iterative optimization
        solver = ARAPSolver(points, normals, voxels, vp_edges,
                            voxel_size, seq)
        solver.solve(max_iter=max_iter)

    def visualize_lidar_lines(self, high_points, sweep_points, _origins):
        """Visualize The LiDAR lines.
        
        """
        from det3d.core.utils.visualization import Visualizer
        
        vis = Visualizer([0.2, 0.2], [-75.2, -75.2], size_factor=1)

        origins = torch.tensor(_origins)

        lines, edges, points = [], [], []
        for i, (_points, _line_points) in enumerate(zip(high_points, sweep_points)):
            line_points = torch.tensor(_line_points, dtype=torch.float32)
            points.append(torch.tensor(_points, dtype=torch.float32))
            origin = origins[i].reshape(-1, 3)
            origin = origin.repeat(line_points.shape[0], 1)
            line = torch.stack([line_points, origin], dim=1).view(-1, 3)
            lines.append(line)
        points = torch.cat(points, dim=0)
        #for i, line in enumerate(lines):
        #    edges = torch.arange(line.shape[0]).view(-1, 2)
        #    vis.curvenetwork(f'lidar-lines-{i}', line, edges)
            #lines = torch.cat(lines, dim=0)
        vis.pointcloud('points', points)
        import ipdb; ipdb.set_trace()

        vis.show()

    def __call__(self, res, info):
        seq = res['lidar_sequence']
        self.motion_estimate(seq)
        
        return res, info
