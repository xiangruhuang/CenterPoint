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
from det3d.solver import ARAPGDSolver as ARAPSolver
import open3d as o3d
from det3d.core.bbox import box_np_ops
from det3d.structures import Sequence

#def generate_sequential_colors(c0, c1, num_frames):
#    c0, c1 = np.array(c0), np.array(c1)
#    ratios = np.linspace(0, 1, num_frames)
#    colors = []
#    for r in ratios:
#        colors.append((1-r)*c0 + r*c1)
#
#    return np.stack(colors, axis=0)

@PIPELINES.register_module
class EstimateMotionMask(object):
    def __init__(self,
                 point_cloud_range=None,
                 visualize=True,
                 ):
        """
        Args:
            point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max)
        """
        self.counter = 0
        self.stats = [0, 0, 0]
        if point_cloud_range is not None:
            self.pc_range = point_cloud_range
        self.visualize = visualize

    def load_lidar_sequence(self, res, info):
        """Load points from adjacent frames into the current coordinate system.

        """
        seq = Sequence(info)
        seq.toglobal()
        origins = seq.camera_trajectory()
        #seq.transform()
        #anno = get_pickle(info['anno_path'])
        #T = anno['veh_to_global'].reshape(4, 4)
        #Tinv = np.linalg.inv(T)
        #sweep_points = []
        #origins = []
        #info['sweeps'].append(dict(path=info['path']))
        ## frames in decreasing order
        #for i, s in enumerate(info['sweeps']):
        #    lidar_file = get_pickle(s['path'])['lidars']
        #    anno_dict = get_pickle(s['path'].replace('lidar', 'annos'))
        #    points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 6)[:, :3]
        #    Ti = Tinv @ anno_dict['veh_to_global'].reshape(4, 4)
        #    points = points @ Ti[:3, :3].T + Ti[:3, 3]
        #    sweep_points.append(points)
        #    origin = np.zeros(3)
        #    origin = Ti[:3, :3] @ origin + Ti[:3, 3]
        #    origins.append(origin)
        #sweep_points = sweep_points[::-1]
        #origins = origins[::-1]

        #center_frame_id = len(sweep_points) - 1
        ## frames in increasing order
        #for i, s in enumerate(info['reverse_sweeps']):
        #    lidar_file = get_pickle(s['path'])['lidars']
        #    anno_dict = get_pickle(s['path'].replace('lidar', 'annos'))
        #    points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 6)[:, :3]
        #    Ti = Tinv @ anno_dict['veh_to_global'].reshape(4, 4)
        #    points = points @ Ti[:3, :3].T + Ti[:3, 3]
        #    sweep_points.append(points)
        #    origin = np.zeros(3)
        #    origin = Ti[:3, :3] @ origin + Ti[:3, 3]
        #    origins.append(origin)
        #origins = np.stack(origins, axis=0)

        #return center_frame_id, sweep_points, origins 
        return seq, origins

    #def range_filter_points(self, points):
    #    """Filter points that are not in range of `self.pc_range`

    #    """
    #    masks = []
    #    new_points = []
    #    for pi in points:
    #        mask = None
    #        for i in range(3):
    #            mask_i = (pi[:, i] >= self.pc_range[i]) & (pi[:, i] <= self.pc_range[i+3])
    #            if mask is None:
    #                mask = mask_i
    #            else:
    #                mask = mask & mask_i
    #        masks.append(mask)
    #        new_points.append(pi[mask])
    #    return masks, new_points

    #def filter_lines(self, _points):
    #    """Filter points that are lines or isolated points.
    #    
    #    """
    #    points = torch.as_tensor(_points)
    #    e0, e1 = radius(points, points, r=0.4, max_num_neighbors=64)
    #    diff = (points[e0] - points[e1])
    #    ppT = (diff.unsqueeze(-1) * diff.unsqueeze(-2)).view(-1, 9)
    #    ppT_per_point = scatter(
    #        ppT, e0, dim=0,
    #        dim_size=points.shape[0], reduce='sum').view(-1, 3, 3)
    #    eigvals = np.linalg.eigvalsh(ppT_per_point.numpy())
    #    valid_idx = np.where(eigvals[:, 1] > 0.03)[0]
    #    return valid_idx, _points[valid_idx]

    def voxelize(self, points, voxel_size=[1, 1]):
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
        voxel_size = torch.tensor(voxel_size)
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
            grid_heights = scatter(points_all[:, 2], coord_1d, dim=0,
                                     dim_size=num_grids, reduce='min')

            # solve an optimization for ground height everywhere
            w = np.zeros(num_grids)
            w[coord_1d] = 1.0
            ground_heights = self.solve(grid_size, grid_heights, w, lamb=10)
            ground_heights = ground_heights.reshape(grid_size[1], grid_size[0]).T
            
            valid_masks = []
            filtered_points = []
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
                #filtered_points.append(s[valid_mask])
                #valid_masks.append(valid_mask)
                seq.frames[i].filter(valid_mask)

            save_dict = dict(masks=valid_masks, num_points=num_points)
            torch.save(save_dict, filename)
        else:
            print('loading ground filtering masks')
            load_dict = torch.load(filename)
            valid_masks = load_dict['masks']
            num_points = load_dict['num_points']
            filtered_points = []
            for i, s in enumerate(sweep_points):
                assert s.shape[0] == num_points[i]
                seq.frames[i].filter(valid_masks[i])
                #filtered_points.append(s[valid_masks[i]])
        #return valid_masks, filtered_points

    #def load_and_sync_annos(self, res, info, filter_other=True):
    #    """Load GT boxes, tracking ids from frames (only for debug).

    #    Args:
    #        filter_other: if True, filter ignored boxes

    #    Returns:
    #        annos (dict): containing
    #            - corners (list[np.ndarray(N, 8, 3)]): box corners per frame
    #            - classes (list[np.ndarray(N)]: box classes per frame
    #            - tokens (list[np.array(dtype=str)]): tracking UIDs
    #            - frame_ids (list[np.ndarray(N)]): frame indices
    #    
    #    """
    #    from det3d.datasets.waymo.waymo_common import TYPE_LIST
    #    #boxes = info['gt_boxes']
    #    corners, classes, tokens, frame_ids, boxes = [], [], [], [], []
    #    cls_map = {'VEHICLE':0, 'PEDESTRIAN':1, 'CYCLIST':2}
    #    label_map = {0:-1, 1:0, 2:1, 3:-1, 4:2}
    #    #classes = info['gt_names'].tolist()
    #    #classes = [np.array([cls_map.get(cls, -1) for cls in classes])]
    #    anno = get_pickle(info['anno_path'])
    #    T = anno['veh_to_global'].reshape(4, 4)
    #    Tinv = np.linalg.inv(T)
    #    tm = defaultdict(lambda: np.eye(4))
    #    # frames in decreasing order
    #    for i, s in enumerate(info['sweeps']):
    #        lidar_file = get_pickle(s['path'])['lidars']
    #        anno_dict = get_pickle(s['path'].replace('lidar', 'annos'))
    #        frame_id = int(s['path'].split('/')[-1].split('.')[0].split('_')[3])
    #        Ti = Tinv @ anno_dict['veh_to_global'].reshape(4, 4)
    #        tm[frame_id] = Ti
    #        boxes_i = np.stack([o['box'] for o in anno_dict['objects']], axis=0)
    #        tokens_i = np.array([o['name'] for o in anno_dict['objects']]).astype(str)
    #        cls = [label_map[o['label']] if o['num_points'] > 0 else -1 for o in anno_dict['objects']]
    #        classes.append(np.array(cls))
    #        corners_i = box_np_ops.center_to_corner_box3d(
    #            boxes_i[:, :3], boxes_i[:, 3:6], boxes_i[:, -1], axis=2)
    #        corners_i = (
    #            corners_i.reshape(-1, 3) @ Ti[:3, :3].T + Ti[:3, 3]
    #            ).reshape(-1, 8, 3)
    #        corners.append(corners_i)
    #        tokens.append(tokens_i)
    #        boxes.append(boxes_i)
    #        frame_ids.append(np.array([frame_id for i in range(boxes_i.shape[0])]))
    #    corners = corners[::-1]
    #    classes = classes[::-1]
    #    tokens = tokens[::-1]
    #    frame_ids = frame_ids[::-1]
    #    boxes = boxes[::-1]

    #    # frames in increasing order
    #    for i, s in enumerate(info['reverse_sweeps']):
    #        lidar_file = get_pickle(s['path'])['lidars']
    #        anno_dict = get_pickle(s['path'].replace('lidar', 'annos'))
    #        Ti = Tinv @ anno_dict['veh_to_global'].reshape(4, 4)
    #        frame_id = int(s['path'].split('/')[-1].split('.')[0].split('_')[3])
    #        tm[frame_id] = Ti
    #        boxes_i = np.stack([o['box'] for o in anno_dict['objects']], axis=0)
    #        tokens_i = np.array([o['name'] for o in anno_dict['objects']]).astype(str)
    #        cls = [label_map[o['label']] if o['num_points'] > 0 else -1 for o in anno_dict['objects']]
    #        classes.append(np.array(cls))
    #        corners_i = box_np_ops.center_to_corner_box3d(
    #            boxes_i[:, :3], boxes_i[:, 3:6], boxes_i[:, -1], axis=2)
    #        corners_i = (
    #            corners_i.reshape(-1, 3) @ Ti[:3, :3].T + Ti[:3, 3]
    #            ).reshape(-1, 8, 3)
    #        corners.append(corners_i)
    #        tokens.append(tokens_i)
    #        boxes.append(boxes_i)
    #        frame_ids.append(np.array([frame_id for i in range(boxes_i.shape[0])]))

    #    if filter_other:
    #        for i, (corner, cls) in enumerate(zip(corners, classes)):
    #            mask = (cls != -1)
    #            corners[i] = corner[mask]
    #            classes[i] = cls[mask]
    #            boxes[i] = boxes[i][mask]
    #            tokens[i] = tokens[i][mask]
    #            frame_ids[i] = frame_ids[i][mask]
    #    
    #    annos = dict(
    #        corners=corners,
    #        classes=classes,
    #        tokens=tokens,
    #        boxes=boxes,
    #        frame_ids=frame_ids,
    #        transformations=tm)

    #    return annos

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


    def motion_estimate(self, seq,
                        voxel_size=[0.6, 0.6, 0.6, 1],
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
        self.seq_id = int(info['path'].split('/')[-1].split('_')[1])
        seq, origins = self.load_lidar_sequence(res, info)
        self.filter_ground(seq, rel_threshold=0.5)
        self.motion_estimate(seq)
        
        return res, info
