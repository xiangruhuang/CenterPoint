import torch
from torch import nn
from torch.nn import functional as F

from ..registry import PREPROCESSORS

@PREPROCESSORS.register_module
class Preprocessor(nn.Module):
    def __init__(self, voxel_size, pc_range, **kwargs):
        super(Preprocessor, self).__init__()
        self.voxel_size = torch.tensor(voxel_size)
        self.pc_range = torch.tensor(pc_range)

    def voxelize(self, points, batch_size, batch):
        pc_range = self.pc_range.cuda()
        voxel_size = self.voxel_size.cuda()
        
        coors = torch.zeros(points.shape[0], 4).to(batch)
        coors[:, 0] = batch + batch_size
        coors[:, 1:] = torch.div(points[:, 0, :3] - pc_range[:3], voxel_size, rounding_mode='floor').flip(-1)

        return coors

    def forward(self, data):
        features = data['features']
        coors = data['coors'].long()
        input_shape = data['input_shape']
        batch_size = data['batch_size']
        num_voxels = data['num_voxels']
        thetas = (torch.rand(batch_size) - 0.5) * (3.1415926 / 2)
        R = torch.stack([thetas.cos(), -thetas.sin(), thetas.sin(), thetas.cos()], dim=0).view(-1, 2, 2).to(features)
        rot_features = features.clone()
        rot_features[:, :, :2] = (R[coors[:, 0]].unsqueeze(1) @ rot_features[:, :, :2].unsqueeze(-1)).squeeze(-1)

        rot_coors = self.voxelize(rot_features[:, :, :3], batch_size, coors[:, 0])
        rot_coors[rot_coors < 0] = 0
        rot_coors[:, 1:] = torch.min(rot_coors[:, 1:],
                                     torch.tensor(input_shape).to(rot_coors).flip(0))
        features = torch.cat([features, rot_features], dim=0)
        coors = torch.cat([coors, rot_coors], dim=0)
        num_voxels = torch.cat([num_voxels, num_voxels.clone()], dim=0)
        data = dict(
            features=features,
            num_voxels=num_voxels,
            coors=coors, 
            batch_size=batch_size*2,
            input_shape=input_shape,
            thetas=thetas,
        )

        return data
