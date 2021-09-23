import torch
import numpy as np
from torch_scatter import scatter
from sklearn.neighbors import NearestNeighbors as NN

from det3d.core.bbox import box_np_ops
from det3d.core.input.dynamic_voxel import DynamicVoxel
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES

from det3d.ops.primitives import primitives_cpu

@PIPELINES.register_module
class ExtractPrimitives(object):
    def __init__(self,
                 voxel_size,
                 elim_th=0.1,
                 visualize=True,
                 ):
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.visualize = visualize
        self.elim_th = elim_th
        #self.pts_voxel_layer = DynamicVoxel(
        #    voxel_size=self.voxel_size,
        #)
    
    def find_unique_voxels(self, coors):
        """Find unique voxels.

        Args:
            coors (torch.Tensor, shape=[N, 3]): voxel coordinates per point

        Returns:
            num_voxels (int): number of unique voxels
            voxel_indices (shape=[N]): unique voxel ids for each point
            num_points (shape=[M]): number of points in each voxel

        """
        # flip back from [z, y, x] to [x, y, z]
        voxel_dims = torch.flip(coors.max(0)[0] + 1, dims=[0])
        ids = coors[:, 2] * voxel_dims[1]
        ids = (ids + coors[:, 1]) * voxel_dims[2] + coors[:, 0]
        
        unique_voxel_ids, voxel_indices, num_points = \
            torch.unique(ids, return_inverse=True, return_counts=True)
        num_voxels = unique_voxel_ids.shape[0]
        return num_voxels, voxel_indices, num_points

    def get_meshes(self, centers, eigvals, eigvecs):
        """ Prepare corners and faces (for visualization only). """

        v1 = eigvecs[:, :, 1]
        v2 = eigvecs[:, :, 2]
        e1 = eigvals[:, 1].unsqueeze(-1).sqrt()
        e2 = eigvals[:, 2].unsqueeze(-1).sqrt()
        corners = []
        for d1 in [-1, 1]:
            for d2 in [-1, 1]:
                corners.append(centers + d1*v1*e1 + d2*v2*e2)
        num_voxels = centers.shape[0]
        corners = torch.stack(corners, dim=1) # [M, 4, 3]
        faces = [[0, 1, 3, 2]]
        faces = torch.as_tensor(faces, dtype=torch.long)
        faces = faces.repeat(num_voxels, 1, 1)
        faces += torch.arange(num_voxels).unsqueeze(-1).unsqueeze(-1)*4

        return corners.view(-1, 3), faces.view(-1, 4)

    def plane_fitting(self, points, edges, amb_edges,
                      sigma=0.07, eps=1e-6):
        """Fit planes iteratively following RANSAC.

        Args:
            points (shape=[N, 3]): points
            edges (shape=[N, 2]): (voxel, point) relations.
            amb_edges (shape=[N, 2]): (voxel, point) relations,
                including ambient points.

        Returns:
            confidence (shape=[M]): per voxel confidence
            planes (shape=[M, 3]): per voxel normals
        """
        
        e0, e1 = edges.long().T
        num_voxels = edges[:, 0].max()+1
        vs = self.voxel_size
        sigma2 = (vs * vs).sum()*sigma*sigma
        all_edges = torch.cat([edges, amb_edges], dim=0)
        a0, a1 = all_edges.long().T
        weights = torch.ones(all_edges.shape[0], dtype=torch.float)
        for itr in range(10):
            # compute weighted centers
            weights_e = weights.unsqueeze(-1)
            centers = scatter(points[a1]*weights_e, a0, dim=0,
                              reduce='add', dim_size=num_voxels)
            w_sum = scatter(weights_e, a0, dim=0,
                            reduce='add', dim_size=num_voxels)
            centers = centers / w_sum
            xyz_centered = points[a1] - centers[a0]
            
            # compute normals
            weights_e = weights_e.unsqueeze(-1)
            ppT = (xyz_centered.unsqueeze(-1) @ xyz_centered.unsqueeze(-2))
            V = scatter(ppT*weights_e, a0, dim=0,
                        reduce='add', dim_size=num_voxels)
            V = V / w_sum.unsqueeze(-1)
            eigvals, eigvecs = np.linalg.eigh(V)
            normals = eigvecs[:, :, 0]

            # update weights
            residual = np.abs((xyz_centered * normals[a0]).sum(-1))
            weights = sigma2/(sigma2+residual*residual+eps)
        
        #d1 = (xyz_centered * eigvecs[:, :, 1][voxel_indices]).sum(-1)
        #d2 = (xyz_centered * eigvecs[:, :, 2][voxel_indices]).sum(-1)

        #d1 = d1[eliminate == False]
        #d2 = d2[eliminate == False]
        #voxel_indices_s = voxel_indices[eliminate == False]
        #dims = []
        #dims.append(
        #    scatter(d1, voxel_indices_s, dim=0,
        #        reduce='min', dim_size=num_voxels))
        #dims.append(
        #    scatter(d1, voxel_indices_s, dim=0,
        #        reduce='max', dim_size=num_voxels))
        #dims.append(
        #    scatter(d2, voxel_indices_s, dim=0,
        #        reduce='min', dim_size=num_voxels))
        #dims.append(
        #    scatter(d2, voxel_indices_s, dim=0,
        #        reduce='max', dim_size=num_voxels))

        eliminate = (residual < self.elim_th)
        normals = torch.as_tensor(normals)
        eigvecs = torch.as_tensor(eigvecs)
        eigvals = torch.as_tensor(eigvals)
        conf_mean = scatter(eliminate.float(), a0, dim=0,
                            reduce='mean', dim_size=num_voxels)
        conf_sum = scatter(eliminate.float(), a0, dim=0,
                           reduce='sum', dim_size=num_voxels)
        thresholds = np.stack(
            [np.linspace(0.6, 0.8, 5), np.linspace(15, 4, 5)], axis=1)

        #thresholds = [(0.50, 15), (0.45, 13), (0.5, 10), (0.55, 7), (0.6, 4)]
        valid_plane_mask = conf_mean < 0
        for mean_th, sum_th in thresholds:
            valid_plane_mask |= ((conf_mean >= mean_th) & (conf_sum >= sum_th))
        surfels = torch.cat([centers, normals, eigvals[:, 1:],
                             eigvecs[:, :, 1], eigvecs[:, :, 2]], dim=-1)
        valid_surfels = surfels[valid_plane_mask]
        valid_centers = centers[valid_plane_mask]
        valid_normals = normals[valid_plane_mask]

        eliminate = eliminate & valid_plane_mask[a0]
        residual[valid_plane_mask[a0] == False] = 1e5
        residual_points = scatter(residual, a1, dim=0,
                                  reduce='min', dim_size=points.shape[0])
        eliminate_points = scatter(eliminate.float(), a1, dim=0,
                                   reduce='max', dim_size=points.shape[0])

        remain_points = points[eliminate_points < 0.5]
        remain_residual = residual_points[eliminate_points < 0.5]

        if self.visualize:
            corners, faces = self.get_meshes(
                centers[valid_plane_mask],
                eigvals[valid_plane_mask],
                eigvecs[valid_plane_mask])
            import polyscope as ps
            ps.set_up_dir('z_up'); ps.init()
            ps_pa = ps.register_point_cloud('points-all', points, radius=2e-4)
            ps_p = ps.register_point_cloud('points', remain_points, radius=2e-4)
            ps_p.add_scalar_quantity('residual', remain_residual)
            ps.register_surface_mesh('planes', corners.numpy(), faces.numpy())

            ps.show()
            
        return valid_surfels.numpy(), (eliminate_points < 0.5)

    def __call__(self, results, info):
        points = results['lidar']['points']
        points_xyz = torch.from_numpy(points[:, :3])
        # (voxel, point), (voxel, ambient point)
        edges, amb_edges = primitives_cpu.query_voxel_neighbors(
                               points_xyz, self.voxel_size)
        #num_voxels = edges[:, 0].max() + 1
        #e0, e1 = edges.T # (voxel, point)
        #edges = torch.cat([edges, amb_edges], dim=0)
        #a0, a1 = edges.T # ambient included (voxel, point)
        ##coors = self.pts_voxel_layer(points).astype(np.int32)
        ##coors = torch.as_tensor(coors)
        ##num_voxels, voxel_indices, num_points = self.find_unique_voxels(coors)
        #
        ## find voxel-wise geometric centers
        #centers = scatter(points_xyz, e0, dim=0,
        #                  reduce='mean', dim_size=num_voxels)
        #xyz_centered = points_xyz[a1] - centers[a0]
        ## aggregate (p p^T) for each point vector p \in R^3
        #ppT = (xyz_centered.unsqueeze(-1) @ xyz_centered.unsqueeze(-2))
        #V = scatter(ppT, a0, dim=0,
        #            reduce='mean', dim_size=num_voxels)
        #eigvals = np.linalg.eigvalsh(V)
        
        # find possible planes
        surfels, mask = self.plane_fitting(
            points_xyz, edges, amb_edges)
       
        results['lidar']['planes'] = surfels
        results['lidar']['points'] = points[mask]
         
        return results, info

@PIPELINES.register_module
class RadiusGraph(object):
    def __init__(self, radius, visualize=False):
        self.radius = radius
        self.num_graphs = len(radius) + 1
    
    def build_radius_graphs(self, points):
        from sklearn.neighbors import radius_neighbors_graph as RNG
        
        for r in self.radius:
            RNG(points, r)

    def __call__(self, res, info):
        points = res['lidar']['points']
        planes = res['lidar']['planes']
        points_all = np.concatenate([points[:, :3], planes[:, :3]], axis=0)
        self.build_radius_graphs(points_all)


@PIPELINES.register_module
class AssignLabelToPrimitives(object):
    def __init__(self, **kwargs):
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius

    def __call__(self, res, info):
        max_objs = self._max_objs

        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]

        # Calculate output featuremap size
        #grid_size = res["lidar"]["voxels"]["shape"] 
        #pc_range = res["lidar"]["voxels"]["range"]
        #voxel_size = res["lidar"]["voxels"]["size"]

        #feature_map_size = grid_size[:2] // self.out_size_factor
        example = {}

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]

            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)
            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

            draw_gaussian = draw_umich_gaussian

            hms, anno_boxs, inds, masks, cats = [], [], [], [], []

            points = res['lidar']['points']
            planes = res['lidar']['planes']
            for idx, task in enumerate(self.tasks):
                points_hm = np.zeros(
                    (points.shape[0], len(class_names_by_task[idx])),
                    dtype=np.float32)
                planes_hm = np.zeros(
                    (planes.shape[0], len(class_names_by_task[idx])),
                    dtype=np.float32)

                gt_boxes_3d = gt_dict['gt_boxes']
                corners = box_np_ops.rbbox3d_to_corners(gt_boxes_3d[idx])
                point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d[idx])
                plane_indices = box_np_ops.points_in_rbbox(planes, gt_boxes_3d[idx])
                gt_classes = gt_dict['gt_classes'][idx] - 1
                for c in range(points_hm.shape[1]):
                    points_hm[point_indices[:, gt_classes == c].any(1), c] = 1.0
                    planes_hm[plane_indices[:, gt_classes == c].any(1), c] = 1.0
                
                if False:
                    from det3d.core.utils.visualization import Visualizer
                    vis = Visualizer([0.1, 0.1], [-75.2, -75.2])
                    vis.boxes('boxes', corners)
                    ps_points = vis.pointcloud('points', points[:, :3])
                    ps_planes = vis.planes('planes', planes)
                    for c in range(points_hm.shape[1]):
                        ps_points.add_scalar_quantity(
                            f'heatmap-{c}', points_hm[:, c])
                        ps_planes.add_scalar_quantity(
                            f'heatmap-{c}', planes_hm[:, c], defined_on='faces')
                    
                    vis.show()
                    import ipdb; ipdb.set_trace()
                
                if res['type'] == 'NuScenesDataset':
                    # [reg, hei, dim, vx, vy, rots, rotc]
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32)
                elif res['type'] == 'WaymoDataset':
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32) 
                else:
                    raise NotImplementedError("Only Support nuScene for Now!")

                ind = np.zeros((max_objs), dtype=np.int64)
                mask = np.zeros((max_objs), dtype=np.uint8)
                cat = np.zeros((max_objs), dtype=np.int64)

                num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)  

                for k in range(num_objs):
                    cls_id = gt_dict['gt_classes'][idx][k] - 1

                    w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], \
                              gt_dict['gt_boxes'][idx][k][5]

                    w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))

                        # be really careful for the coordinate system of your box annotation. 
                        x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], \
                                  gt_dict['gt_boxes'][idx][k][2]

                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                         (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32)  
                        ct_int = ct.astype(np.int32)

                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                            continue 

                        draw_gaussian(hm[cls_id], ct, radius)

                        new_idx = k
                        x, y = ct_int[0], ct_int[1]

                        cat[new_idx] = cls_id
                        ind[new_idx] = y * feature_map_size[0] + x
                        mask[new_idx] = 1

                        if res['type'] == 'NuScenesDataset': 
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][8]
                            anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        elif res['type'] == 'WaymoDataset':
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][-1]
                            anno_box[new_idx] = np.concatenate(
                            (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                            np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        else:
                            raise NotImplementedError("Only Support Waymo and nuScene for Now")

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)

            # used for two stage code 
            boxes = flatten(gt_dict['gt_boxes'])
            classes = merge_multi_group_label(gt_dict['gt_classes'], num_classes_by_task)

            if res["type"] == "NuScenesDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            elif res['type'] == "WaymoDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            else:
                raise NotImplementedError()

            boxes_and_cls = np.concatenate((boxes, 
                classes.reshape(-1, 1).astype(np.float32)), axis=1)
            num_obj = len(boxes_and_cls)
            assert num_obj <= max_objs
            # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
            boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
            gt_boxes_and_cls[:num_obj] = boxes_and_cls

            example.update({'gt_boxes_and_cls': gt_boxes_and_cls})

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})
        else:
            pass

        res["lidar"]["targets"] = example
