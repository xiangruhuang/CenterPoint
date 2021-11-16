from ..registry import PIPELINES
import numpy as np
import os
from torch_scatter import scatter
from torch_cluster import radius_graph
import torch
import scipy
from scipy.sparse import csr_matrix
from det3d.ops.primitives.primitives_cpu import (
    voxelization, 
)

@PIPELINES.register_module
class FilterGround(object):
    def __init__(self, rel_threshold=0.5, lamb=10, debug=False):
        self.rel_threshold = rel_threshold
        self.debug = debug
        self.lamb = lamb
        self.voxel_size=[1,1]

    def voxelize(self, points):
        """ voxelize the points

        Args:
            points (N, 3)
        Returns:
            coord_1d (N): unique indices from `coord_2d`
            coord_2d (N, 2): discrete coordinate on (x,y)-axis
            grid_size (3): number of grids in each dimension.
        """
        x, y, z = points.T
        pc_range = torch.tensor(
            [points[:, 0].min(), points[:, 1].min(), 
             points[:, 0].max(), points[:, 1].max()])
        voxel_size = torch.tensor(self.voxel_size)
        grid_size = ((pc_range[2:] - pc_range[:2]) // voxel_size).long()
        x_coord = ((x - pc_range[0]) // voxel_size[0]).long()
        y_coord = ((y - pc_range[1]) // voxel_size[1]).long()
        x_coord[x_coord >= grid_size[0]] = grid_size[0]-1
        y_coord[y_coord >= grid_size[1]] = grid_size[1]-1
        x_coord[x_coord < 0] = 0
        y_coord[y_coord < 0] = 0
        
        coord_1d = (y_coord * grid_size[0] + x_coord).long()
        coord_2d = torch.stack([x_coord, y_coord], axis=-1)

        return coord_1d, coord_2d, grid_size, pc_range, voxel_size
    
    def solve(self, dims, b, w, lamb=100):
        """Solve linear equation: (diag(w) + L) x = b
        L being the laplacian of a grid graph of dimension specified by `dims`

        Args:
            dims (2): grid dimensions, N = dims[0]*dims[1]
            b (N)
            w (N): non-negative diagonal

        Returns:
            x (N): solution of the linear system.
        """
        num_grids = dims[0] * dims[1]
        L = self.get_grid_laplacian(dims)
        W = scipy.sparse.dia_matrix((w, 0), shape=(num_grids, num_grids))
        x = scipy.sparse.linalg.spsolve(W + lamb * L, b)

        return x

    def get_grid_laplacian(self, dims):
        """Get a laplacian of a 2d grid graph
        
        Args:
            dims (2): number of grids in x, y dimension

        Returns:
            L (scipy.sparse.csr_matrix): sparse Laplacian

        """
        num_grids = dims[0] * dims[1]
        grids = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]))
        grids = np.stack(grids, axis=-1).reshape(-1, 2)
        grids = torch.tensor(grids)
        e0, e1 = radius_graph(grids, r=1.5, loop=False)
        grids_1d = grids[:, 1]*dims[0] + grids[:, 0]
        deg = scatter(torch.ones_like(e0), e0, dim=0,
                      dim_size=num_grids, reduce='add')
        
        data, row, col = [], [], []
        # edges
        data.append(-np.ones_like(e0))
        row.append(e0.numpy())
        col.append(e1.numpy())

        # diagonal
        data.append(deg.numpy())
        row.append(np.arange(num_grids))
        col.append(np.arange(num_grids))
        
        # merge
        data = np.concatenate(data)
        row = np.concatenate(row)
        col = np.concatenate(col)
        
        L = csr_matrix((data, (row, col)), shape=(num_grids, num_grids))
        
        return L

    def filter_ground(self, seq, rel_threshold=0.2):
        """Reconstruct the ground of the entire scene.

        Args:
            sweep_points (list[np.ndarray of shape (*, 3)]): list of frames
            rel_threshold : filter points w.r.t. relative height to ground

        Returns:
            valid_masks
            filtered_points

        """
        sweep_points = [f.points for f in seq.frames]
        filename = f'data/Waymo/train/ssl/ground_{self.seq_id}.pt'
        if not os.path.exists(filename):
            _points_all = np.concatenate(sweep_points, axis=0)

            from torch_cluster import grid_cluster
            points_all = torch.tensor(_points_all)
            
            # voxelize points into coordinates on x-y plane
            coord_1d, coord_2d, grid_size, pc_range, voxel_size = \
                self.voxelize(points_all)
            num_grids = grid_size[0]*grid_size[1]
            
            # find unique set of 1D coordinates and an inverse map
            coors, inverse = coord_1d.unique(return_inverse=True)
            unique_x, unique_y = coors % grid_size[0], coors // grid_size[0]
            coors = torch.stack([unique_x, unique_y], axis=-1)
            
            # distribute heights into voxels, find minimum per grid
            grid_heights = scatter(points_all[:, 2],
                                   coord_1d, dim=0,
                                   dim_size=num_grids, reduce='min')

            # solve an optimization for ground height everywhere
            w = np.zeros(num_grids)
            w[coord_1d] = 1.0
            ground_heights = self.solve(grid_size, grid_heights, w, lamb=self.lamb)
            ground_heights = ground_heights.reshape(grid_size[1], grid_size[0]).T
            
            valid_masks = []
            num_points = [s.shape[0] for s in sweep_points]
            for i, s in enumerate(sweep_points):
                x, y, z = torch.tensor(s).T
                x_coord = ((x - pc_range[0]) // voxel_size[0]).long()
                y_coord = ((y - pc_range[1]) // voxel_size[1]).long()
                x_coord[x_coord >= grid_size[0]] = grid_size[0]-1
                y_coord[y_coord >= grid_size[1]] = grid_size[1]-1
                x_coord[x_coord < 0] = 0
                y_coord[y_coord < 0] = 0
                coord_2d = torch.stack([x_coord, y_coord], axis=-1)
                
                rel_height = z - ground_heights[(coord_2d[:, 0], coord_2d[:, 1])]
                valid_mask = torch.where(rel_height > rel_threshold)[0]
                valid_masks.append(valid_mask)
                seq.frames[i].filter(valid_mask)

            save_dict = dict(masks=valid_masks, num_points=num_points,
                             ground=ground_heights, pc_range=pc_range,
                             voxel_size=voxel_size)
            #torch.save(save_dict, filename)
        else:
            print('loading ground filtering masks')
            load_dict = torch.load(filename)
            valid_masks = load_dict['masks']
            num_points = load_dict['num_points']
            ground_heights = load_dict['ground']
            pc_range = load_dict['pc_range']
            voxel_size = load_dict['voxel_size']
            filtered_points = []
            for i, s in enumerate(sweep_points):
                assert s.shape[0] == num_points[i]
                seq.frames[i].filter(valid_masks[i])

        return ground_heights, pc_range, voxel_size 

    def __call__(self, res, info):
        import time
        start_time = time.time()
        seq = res['lidar_sequence']
        self.seq_id = seq.seq_id
        if self.debug:
            from det3d.core.utils.visualization import Visualizer
            p0 = seq.points4d()
            import ipdb; ipdb.set_trace()

        print('filtering ground')
        ground_heights, pc_range, voxel_size = \
                self.filter_ground(seq,
                                   rel_threshold=self.rel_threshold)
        res['lidar_sequence'] = seq
        end_time = time.time()
        if self.debug:
            vis = Visualizer(voxel_size, pc_range, size_factor=1)
            #ps_p = vis.pointcloud('original points', p0[:, :3])
            #vis.pc_scalar('original points', 'frame % 2', p0[:, -1] % 2)
            p1 = seq.points4d()
            import ipdb; ipdb.set_trace()
            ps_p = vis.pointcloud('points', p1[:, :3])
            ps_g = vis.heatmap('ground', ground_heights.T)
            vis.pc_scalar('points', 'frame % 2', p1[:, -1] % 2)
            print(f'filter ground: time={end_time-start_time:.4f}')
            vis.save('/afs/csail.mit.edu/u/x/xrhuang/public_html/filter_ground.pth')
            #vis.show()

        return res, info
