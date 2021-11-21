from ..registry import PIPELINES
import torch
from det3d.core.utils.visualization import Visualizer
from PytorchHashmap.torch_hash import HashTable
import numpy as np
from det3d.solver.learning_schedules_fastai import (
    OneCycle,
    ExponentialDecay,
)
import os
from torch_cluster import fps
from det3d.models.builder import build_neck
from det3d.ops.primitives.primitives_cpu import voxelization
from torch_scatter import scatter

@PIPELINES.register_module
class NeuralRegistration(object):
    def __init__(self,
                 flownet=None,
                 hash_table_size=600000,
                 voxel_size=[2.5, 2.5, 2.5, 1],
                 lr_config=None,
                 resume=False,
                 debug=False):
        self.debug = debug
        self.net = build_neck(flownet).cuda()
        self.hash_table_size = hash_table_size
        self.ht = HashTable(hash_table_size*2)
        self.voxel_size = torch.tensor(voxel_size)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.max_iter=500
        if lr_config is not None:
            self.lr_scheduler = ExponentialDecay(
                                    self.optimizer,
                                    self.max_iter*3,
                                    1e-3,
                                    decay_length=100.0/(self.max_iter*3),
                                    decay_factor=0.8,
                                )
            #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            #                    self.optimizer, step_size=200, gamma=0.8)
            #self.lr_scheduler = OneCycle(self.optimizer, **lr_config)
        self.window_size=5
        self.resume=resume

    def chamfer(self, p, q):
        p_idx, q_idx = self.ht.find_corres(q, p, self.voxel_size, 1)
        q_idx_inv, p_idx_inv = self.ht.find_corres(p, q, self.voxel_size, -1)
        loss = (p[p_idx, :3] - q[q_idx, :3]).square().mean()
        loss += (p[p_idx_inv, :3] - q[q_idx_inv, :3]).square().mean()
        return loss

    def load(self, load_path):
        checkpoint = torch.load(load_path)
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        self.net.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        print(f'loaded model from {load_path}')

    def save(self, save_path):
        model_state_dict = self.net.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        checkpoint = dict(model_state_dict=model_state_dict,
                          optimizer_state_dict=optimizer_state_dict)
        torch.save(checkpoint, save_path)
        print(f'saved model to {save_path}')

    def neural_reg(self, points_xyzt):
        for itr in range(self.max_iter):
            rand_idx = torch.randperm(points_xyzt.shape[0],
                                      dtype=torch.long,
                                      device='cuda:0')[:100000]
            points = points_xyzt[rand_idx]
            
            self.optimizer.zero_grad()
            v = self.net(points)
            pf = points.clone()
            pf[:, :3] += v
            loss_f = self.chamfer(pf, points_xyzt)
            pb = points.clone()
            pb[:, :3] += v
            pb[:, -1] += 1
            vb = -self.net(pb)
            loss_b = (vb + v).square().mean()
            loss = loss_f + loss_b
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(self.itr)
            #lr = self.optimizer.param_groups[-1]['lr']
            print(f'iter={self.itr}, loss_f={loss_f.item():.6f}, '\
                  f'loss_b={loss_b.item():.6f}, lr={self.optimizer.lr:.7f}')
            self.itr += 1
            #if itr % 100 == 0:
            #    frame_offset = points_xyzt[:, -1].min().long().item()
            #    self.visualize(points_xyzt, frame_offset)

    def visualize(self, points_xyzt, frame_offset):
        vis = Visualizer([], [])
        for i in range(self.window_size):
            mask = (points_xyzt[:, -1] == frame_offset+i)
            p = points_xyzt[mask]
            vis.pointcloud(f'p{frame_offset+i}', p[:, :3].cpu())
            if i != self.window_size - 1:
                v = self.net(p)
                p_moved = (p[:, :3] + v)
                vis.pointcloud(f'p{frame_offset+i} moved',
                               p_moved.detach().cpu())

        vis.show()

    def __call__(self, res, info):
        import time
        start_time = time.time()
        seq = res['lidar_sequence']
        num_frames = len(seq.frames)
        for i in range(num_frames-self.window_size):
            points_xyzt = seq.points4d(i, i+self.window_size)
            points_xyzt = torch.tensor(points_xyzt, dtype=torch.float32).cuda()
            save_path = os.path.join('work_dirs', 'motion_estimation',
                                     f'seq_{seq.seq_id}_frame_{i}.pt')
            if self.resume and os.path.exists(save_path):
                self.load(save_path)
            self.itr = 0
            for grid_voxel_size in [[0.6, 0.6, 0.6, 1],
                                    [0.4, 0.4, 0.4, 1],
                                    [0.2, 0.2, 0.2, 1]]:
                ev, ep = voxelization(points_xyzt.cpu(),
                                      torch.tensor(grid_voxel_size), False
                                      )[0].T.long().cuda()
                num_voxels = ev.max().item()+1
                voxels_xyzt = scatter(points_xyzt[ep], ev, dim=0,
                                      dim_size=num_voxels, reduce='mean')
                if voxels_xyzt.shape[0] > self.hash_table_size:
                    rand_idx = torch.randperm(voxels_xyzt.shape[0])
                    rand_idx = rand_idx[:self.hash_table_size].cuda()
                    voxels_xyzt = voxels_xyzt[rand_idx]
                self.neural_reg(voxels_xyzt)
                self.save(save_path)
            #self.visualize(voxels_xyzt, i)
        
        end_time = time.time()
        print(f'Neural Registration: time={end_time-start_time:.4f}')
