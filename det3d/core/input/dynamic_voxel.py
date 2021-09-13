import numpy as np

class DynamicVoxel:
    def __init__(self, voxel_size):
        voxel_size = np.array(voxel_size, dtype=np.float32)
        self._voxel_size = voxel_size

    def generate(self, points):
        coors_range = points.min(0)[:3]
        coors = []
        for j in range(3):
            c = np.floor((points[:, j] - coors_range[j]) / self._voxel_size[j])
            coors.append(c)
        coors = np.stack(coors, axis=-1)

        return coors
    
    def __call__(self, points):
        return self.generate(points)
    
    @property
    def voxel_size(self):
        return self._voxel_size
