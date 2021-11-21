import torch
from det3d.core.utils.visualization import Visualizer
from PytorchHashmap.torch_hash import HashTable
import numpy as np
from det3d.solver.learning_schedules_fastai import OneCycle
import os

def MLP(channels, batch_norm=True):
    module_list = []
    for i in range(1, len(channels)):
        module = torch.nn.Linear(channels[i-1], channels[i])
        if batch_norm:
            module = torch.nn.Sequential(module, torch.nn.ReLU(),
                                         torch.nn.BatchNorm1d(channels[i]))
        module_list.append(module)
    return torch.nn.Sequential(*module_list)

def chamfer(p, q, voxel_size):
    p_idx, q_idx = ht.find_corres(q, p, voxel_size, 1)
    q_idx_inv, p_idx_inv = ht.find_corres(p, q, voxel_size, -1)
    loss = (p[p_idx, :3] - q[q_idx, :3]).square().mean()
    loss += (p[p_idx_inv, :3] - q[q_idx_inv, :3]).square().mean()
    return loss

class NeuralReg(torch.nn.Module):
    """ Predict flow from (x,y,z,t)
    Args:
        points_xyzt (torch.tensor, [N, 4]): temporal point cloud

    Returns:
        velo (torch.tensor, [N, 3]): velocity of each point

    """
    def __init__(self,
                 channels = [(4, 128), (128, 128), (128, 128), (128, 128),
                             (128, 128), (128, 128), (128, 3)],
                 **kwargs):
        super(NeuralReg, self).__init__(**kwargs)
        self.layers = []
        for i, channel in enumerate(channels):
            if i == len(channels) - 1:
                layer = MLP(channel, batch_norm=False)
            else:
                layer = MLP(channel)
            self.layers.append(layer)
            self.__setattr__(f'mlp{i}', self.layers[-1])

    def forward(self, points_xyzt):
        points = points_xyzt
        for layer in self.layers:
            next_points = layer(points)
            if next_points.shape[-1] == points.shape[-1]:
                next_points += points
            points = next_points

        return next_points

if __name__ == '__main__':
    net = NeuralReg()
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    lr_scheduler = OneCycle(optimizer, 10000, lr_max=0.001, moms=[0.95, 0.85],
                            div_factor=10.0, pct_start=0.4)
    import time
    start_time = time.time()
    seq = torch.load('seq_fg8.pt')
    points_xyzt = torch.tensor(seq.points4d(0, 5), dtype=torch.float32).cuda()
    from torch_cluster import fps
    points_xyzt = points_xyzt[fps(points_xyzt, ratio=0.1)]
    end_time = time.time()
    print(f'loaded sequence, time={end_time-start_time}')
    load_path = 'checkpoint1000.pt'
    if os.path.exists(load_path):
        print(f'loading checkpoint from {load_path}')
        checkpoint = torch.load(load_path)
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        #scheduler_state_dict = checkpoint['scheduler_state_dict']
        net.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        start_itr = checkpoint['itr'] + 1
        #min_dist = checkpoint['min_dist']
    else:
        start_itr = 0
        #min_dist = torch.zeros(points_xyzt.shape[0]) + 1e10
        # initialization
        #for init_itr in range(10000):
        #    optimizer.zero_grad()
        #    rand_idx = np.random.permutation(points_xyzt.shape[0])
        #    rand_idx = torch.tensor(rand_idx).long().cuda()
        #    points = points_xyzt[rand_idx]

        #    v = net(points)
        #    loss = v.square().mean()
        #    loss.backward()
        #    print(f'init velocity, loss={loss:.4f}')
        #    optimizer.step()
        #    if loss.item() < 0.001:
        #        break
        #model_state_dict = net.state_dict()
        #optimizer_state_dict = optimizer.state_dict()
        #checkpoint = dict(model_state_dict=model_state_dict,
        #                  optimizer_state_dict=optimizer_state_dict,
        #                  itr=-1, min_dist=min_dist)
        #torch.save(checkpoint, f'checkpoint_init.pt')

    ht = HashTable(points_xyzt.shape[0]*2)
    voxel_size = torch.tensor([2.0, 2.0, 2.0, 1])
    vis = Visualizer([], [])
    #vis.pointcloud('points', points_xyzt[:, :3].cpu())
    #p_idx_all, v_idx_all = ht.find_corres(points_xyzt, points_xyzt,
    #                                      voxel_size, 1)
    #vis.corres('all corres', points_xyzt[p_idx_all, :3].cpu(),
    #                         points_xyzt[v_idx_all, :3].cpu())
    #vis.show()
    
    for itr in range(start_itr, 10000):
        rand_idx = np.random.permutation(points_xyzt.shape[0])[:100000]
        rand_idx = torch.tensor(rand_idx).long().cuda()
        #mask = (min_dist[rand_idx] < 0.1)
        points = points_xyzt[rand_idx]

        #v = net(points)
        #p_moved = points.clone()
        #p_moved[:, :3] += v
        #p_idx, v_idx = ht.find_corres(points_xyzt, p_moved, voxel_size, 1)
        #p_idx, v_idx = ht.find_corres(points_xyzt, p_moved, voxel_size, 1)
        #valid_mask = mask[p_idx]
        #p_idx = p_idx[valid_mask]
        #v_idx = v_idx[valid_mask]
        #valid_mask1 = mask[p_idx1] == False
        #p_idx1 = p_idx1[valid_mask1]
        #v_idx1 = v_idx1[valid_mask1]
        #p_idx = torch.cat([p_idx, p_idx1], dim=0)
        #v_idx = torch.cat([v_idx, v_idx1], dim=0)
        #num_original = valid_mask1.sum().item()
        #num_moving = valid_mask.sum().item()

        #p_moved[:, :3] -= 2*v
        #p_idx, v_idx = ht.find_corres(points_xyzt, p_moved, voxel_size, -1)
        #p_idx_inv, v_idx_inv = ht.find_corres(points_xyzt, p_moved, voxel_size, -1)
        #valid_mask = mask[p_idx]
        #p_idx = p_idx[valid_mask]
        #v_idx = v_idx[valid_mask]
        #valid_mask1 = mask[p_idx1] == False
        #p_idx1 = p_idx1[valid_mask1]
        #v_idx1 = v_idx1[valid_mask1]
        #p_idx = torch.cat([p_idx, p_idx1], dim=0)
        #v_idx = torch.cat([v_idx, v_idx1], dim=0)
        #num_original = valid_mask1.sum().item()
        #num_moving = valid_mask.sum().item()
        
        optimizer.zero_grad()
        v = net(points)
        pf = points.clone()
        pf[:, :3] += v
        loss_f = chamfer(pf, points_xyzt, voxel_size)
        pb = points.clone()
        pb[:, :3] += v
        pb[:, -1] += 1
        vb = -net(pb)
        loss_b = (vb + v).square().mean()
        #pb = points.clone()
        #pb[:, :3] += v + vb
        #pb[:, -1] += 1
        #loss_b = chamfer(points_xyzt, pb, voxel_size)
        loss = loss_f + loss_b
        #loss = (points[p_idx, :3] + v[p_idx] - points_xyzt[v_idx, :3]).square().mean()
        #loss += (points[p_idx_inv, :3] - v[p_idx_inv] - points_xyzt[v_idx_inv, :3]).square().mean()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(itr)
        print(f'iter={itr}, loss_f={loss_f.item():.6f}, loss_b={loss_b.item():.6f}, lr={optimizer.lr:.7f}')
        #dist = (points[p_idx, :3] - points_xyzt[v_idx, :3]).norm(p=2, dim=-1).detach().cpu()
        #min_dist[rand_idx[p_idx]] = min_dist[rand_idx[p_idx]].min(dist)
        if itr % 100 == 0:
            model_state_dict = net.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            checkpoint = dict(model_state_dict=model_state_dict,
                              optimizer_state_dict=optimizer_state_dict,
                              itr=itr)
            torch.save(checkpoint, f'checkpoint{itr}.pt')
            mask0 = (points_xyzt[:, -1] == 0)
            p0 = points_xyzt[mask0]
            mask1 = (points_xyzt[:, -1] == 1)
            p1 = points_xyzt[mask1]
            mask2 = (points_xyzt[:, -1] == 2)
            p2 = points_xyzt[mask2]
            vis.pointcloud('p0', p0[:, :3].cpu())
            vis.pointcloud('p1', p1[:, :3].cpu())
            vis.pointcloud('p2', p2[:, :3].cpu())
            v0 = net(p0)
            p0_moved = (p0[:, :3] + v0)
            vis.pointcloud('p0 moved', p0_moved.detach().cpu())
            v1 = net(p1)
            p1_moved = (p1[:, :3] + v1)
            vis.pointcloud('p1 moved', p1_moved.detach().cpu())
            del p0
            del p1
            del p2
            del p0_moved
            del p1_moved
            del mask0
            del mask1
            del mask2
            del v0
            del v1
            #vis.corres('corres', points[p_idx, :3].cpu(), points_xyzt[v_idx, :3].cpu())
            vis.show()
    
