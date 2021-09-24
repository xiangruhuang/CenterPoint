import os.path as osp
import warnings
import numpy as np
from functools import reduce
import torch

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle 
import os 
from ..registry import PIPELINES
from torch_scatter import scatter

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=4, painted=False):
    if painted:
        dir_path = os.path.join(*path.split('/')[:-2], 'painted_'+path.split('/')[-2])
        painted_path = os.path.join(dir_path, path.split('/')[-1]+'.npy')
        points =  np.load(painted_path)
        points = points[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] # remove ring_index from features 
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, painted=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), painted=painted).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T

def read_single_waymo(obj):
    bin_file = obj['lidars']
    points_all = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 6)
    points_xyz = points_all[:, :3]
    points_feature = points_all[:, 3:5]

    #points_xyz = obj["lidars"]["points_xyz"]
    #points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])
    
    bin_file = obj['lidars']
    points_all = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 6)
    points_xyz = points_all[:, :3]
    points_feature = points_all[:, 3:5]

    #points_xyz = obj["lidars"]["points_xyz"]
    #points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T


def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res["painted"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, painted=res["painted"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])
        
        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points

            if nsweeps > 1: 
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadMotionMasks(object):
    def __init__(self,
                 #point_cloud_range = [-75.2, -75.2, 0.3, 75.2, 75.2, 4],
                 point_cloud_range = [-74.88, -74.88, 0.3, 74.88, 74.88, 4],
                 visualize=False,
                 interval=1,
                 **kwargs):
        self.point_cloud_range = point_cloud_range
        self.visualize = visualize
        self.interval = interval

    def __call__(self, res, info):

        if res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            path = info['path']
            tokens = path.split('/')[-1].split('.')[0].split('_')
            seq_id, frame_id = int(tokens[1]), int(tokens[3])
            if seq_id % self.interval != 0:
                pth_file = os.path.join(
                    'data/Waymo/train/motion_masks/',
                    f'0{seq_id:03d}{frame_id:03d}.pth',
                )
                if os.path.exists(pth_file):
                    try:
                        motion_dict = torch.load(pth_file)
                        points = res['lidar']['points'][:, :3]
                        in_range = ((points > self.point_cloud_range[:3]) & (points < self.point_cloud_range[3:]))
                        in_range = in_range.all(axis=-1)
                        points = points[in_range]
                        valid_points = points[motion_dict['valid_idx']]
                        object_points = valid_points[motion_dict['obj_idx']]
                        moving_points = valid_points[motion_dict['moving']]
                        moving_clusters = motion_dict['point2cluster'][motion_dict['moving']]
                        centers = scatter(
                                    torch.from_numpy(moving_points),
                                    torch.from_numpy(moving_clusters).long(),
                                    dim=0, dim_size=moving_clusters.max()+1,
                                    reduce='mean')
                        res['lidar']['moving_points'] = centers[np.unique(moving_clusters)]
                        res['lidar']['using_motion_mask'] = np.array(True).astype(np.bool).reshape(1)
                    except Exception as e:
                        print(f'error loading {pth_file}')
                        print(e)
                        res['lidar']['using_motion_mask'] = np.array(False).astype(np.bool).reshape(1)
                        moving_points = np.zeros(shape=(0, 3), dtype=np.float32)
                else:
                    res['lidar']['using_motion_mask'] = np.array(False).astype(np.bool).reshape(1)
                    moving_points = np.zeros(shape=(0, 3), dtype=np.float32)
            else:
                res['lidar']['using_motion_mask'] = np.array(False).astype(np.bool).reshape(1)
                moving_points = np.zeros(shape=(0, 3), dtype=np.float32)

            if self.visualize:
                import polyscope as ps
                ps.set_up_dir('z_up')
                ps.init()
                ps.register_point_cloud('points', points, radius=2e-4)
                ps.register_point_cloud('valid points', valid_points, radius=2e-4)
                ps.register_point_cloud('objects', object_points, radius=3e-4)
                ps.register_point_cloud('moving', moving_points, radius=3e-4)
                ps.show() 
           
            res['lidar']['moving_points'] = moving_points
        else:
            pass

        return res, info

@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
        else:
            pass 

        return res, info
