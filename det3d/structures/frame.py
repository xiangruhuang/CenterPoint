import pickle
import numpy as np
from det3d.core.bbox import box_np_ops
import torch
import open3d as o3d

def get_pickle(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)

def get_frame_id(path):
    return int(path.split('/')[-1].split('.')[0].split('_')[3])

class Frame:
    def __init__(self, path):
        self.path = path
        self.frame_id = get_frame_id(path)
        self.load_annos()
        self.load_points()
        self.pose = np.eye(4)
        self.camera_loc = np.eye(3)
    
    def transform(self, T):
        self.pose = T @ self.pose
        self.points = self.points @ T[:3, :3].T + T[:3, 3]
        self.normals = self.normals @ T[:3, :3].T
        self.corners = (
                self.corners.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
                ).reshape(-1, 8, 3)
        self.camera_loc = T[:3, :3] @ self.camera_loc + T[:3, 3]
    
    def load_points(self):
        self.lidar_file = get_pickle(self.path)['lidars']
        self.points = np.fromfile(self.lidar_file,
                                  dtype=np.float32).reshape(-1, 6)[:, :3]
        self.mask = np.ones(self.points.shape[0], dtype=bool)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                         radius=0.5, max_nn=30))
        self.normals = np.array(pcd.normals)

    def load_annos(self):
        anno_dict = get_pickle(self.path.replace('lidar', 'annos'))
        self.T = anno_dict['veh_to_global'].reshape(4, 4)
        self.frame_name = anno_dict['frame_name']
        self.scene_name = anno_dict['scene_name']
        cls_map = {'VEHICLE':0, 'PEDESTRIAN':1, 'CYCLIST':2}
        label_map = {0:-1, 1:0, 2:1, 3:-1, 4:2}
        self.boxes = np.array([o['box'] for o in anno_dict['objects']])
        self.global_speed = np.array([o['global_speed']
                                     for o in anno_dict['objects']])
        self.global_accel = np.array([o['global_accel']
                                     for o in anno_dict['objects']])
        #if self.boxes.shape[0] > 0:
        #    self.boxes[:, -1] *= -1
        self.tokens = np.array([o['name'] for o in anno_dict['objects']]
                               ).astype(str)
        cls = [label_map[o['label']] if o['num_points'] > 0 else -1 \
               for o in anno_dict['objects']]
        self.classes = np.array(cls)
        if self.boxes.shape[0] > 0:
            self.corners = box_np_ops.center_to_corner_box3d(
                self.boxes[:, :3], self.boxes[:, 3:6], -self.boxes[:, -1], axis=2)
        else:
            self.corners = np.zeros((0, 8, 3))
        mask = (self.classes != -1)
        self.corners = self.corners[mask]
        self.classes = self.classes[mask]
        self.boxes = self.boxes[mask]
        self.tokens = self.tokens[mask]

    def toglobal(self):
        T = self.T @ np.linalg.inv(self.pose)
        self.transform(T)

    def tolocal(self):
        T = np.linalg.inv(self.pose)
        self.transform(T)

    def filter(self, mask):
        premask = self.mask == True
        self.mask[premask][:] = False
        self.mask[premask][mask] = True
        self.points = self.points[mask]
        self.normals = self.normals[mask]

