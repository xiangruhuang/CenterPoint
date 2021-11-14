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
            try:
                self.frames.append(Frame(frame_path))
            except Exception as e:
                print(e)
                import ipdb; ipdb.set_trace()
                pass
   
    def toglobal(self):
        for frame in self.frames:
            frame.toglobal()
    
    def tolocal(self):
        for frame in self.frames:
            frame.tolocal()

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
    
    def origin_classes(self):
        classes = []
        for f in self.frames:
            classes.append(f.origin_classes)
        classes = np.concatenate(classes, axis=0)
        return classes

    def tokens(self):
        tokens = []
        for f in self.frames:
            tokens.append(f.tokens)
        tokens = np.concatenate(tokens, axis=0)
        return tokens

    def object_traces(self):
        object_pool = {}
        for fid, f in enumerate(self.frames):
            for tid, token in enumerate(f.tokens):
                obj_box = (f.frame_id,
                           f.boxes[tid],
                           f.corners[tid],
                           f.origin_classes[tid],
                           f.global_speed[tid],
                           f.global_accel[tid],
                           token)
                trace = object_pool.get(token, [])
                trace.append(obj_box)
                object_pool[token] = trace

        object_traces = []
        for token, trace in object_pool.items():
            frame_ids, boxes, corners, classes, global_speed, global_accel, \
                tokens = [[] for i in range(7)]
            for t in trace:
                frame_ids.append(t[0])
                boxes.append(t[1])
                corners.append(t[2])
                classes.append(t[3])
                global_speed.append(t[4])
                global_accel.append(t[5])
                tokens.append(t[6])
            frame_ids = np.array(frame_ids)
            boxes = np.stack(boxes, axis=0)
            corners = np.stack(corners, axis=0)
            classes = np.array(classes)
            tokens = np.array(tokens).astype(str)
            global_speed = np.stack(global_speed, axis=0)
            global_accel = np.stack(global_accel, axis=0)
            trace_dict = dict(
                frame_ids=frame_ids,
                boxes=boxes,
                corners=corners,
                classes=classes,
                global_speed=global_speed,
                global_accel=global_accel,
                tokens=tokens)
            object_traces.append(trace_dict)

        return object_traces
