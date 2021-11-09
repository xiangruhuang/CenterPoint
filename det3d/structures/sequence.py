import numpy as np
from .frame import Frame, get_frame_id

class Sequence:
    def __init__(self, info):
        frame_paths = [info['path']]
        for s in info['sweeps']:
            frame_paths.append(s['path'])
        for s in info['reverse_sweeps']:
            frame_paths.append(s['path'])
        frame_paths = sorted(frame_paths,
                             key=lambda s: get_frame_id(s))
        self.seq_id = int(info['path'].split('/')[-1].split('_')[1])
        self.frames = []
        for frame_path in frame_paths:
            self.frames.append(Frame(frame_path))
   
    def toglobal(self):
        for frame in self.frames:
            frame.toglobal()

    def camera_trajectory(self):
        traj = []
        for frame in self.frames:
            traj.append(frame.camera_loc)

    def points4d(self):
        points = []
        for f in self.frames:
            s = f.points
            frame_id = np.ones((s.shape[0], 1)) * f.frame_id
            s = np.concatenate([s, frame_id], axis=-1)
            points.append(s)
        points = np.concatenate(points, axis=0)
        return points

    def normals(self):
        normals = []
        for f in self.frames:
            normals.append(f.normals)

        normals = np.concatenate(normals, axis=0)
        return normals

    def corners(self):
        corners = []
        for f in self.frames:
            corners.append(f.corners)
        corners = np.concatenate(corners, axis=0)
        return corners
    
    def classes(self):
        classes = []
        for f in self.frames:
            classes.append(f.classes)
        classes = np.concatenate(classes, axis=0)
        return classes
