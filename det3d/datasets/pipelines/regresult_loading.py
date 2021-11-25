from ..registry import PIPELINES
from det3d.structures import Sequence
import os
import torch
from det3d.models.builder import build_neck
import numpy as np
from det3d.core.utils.visualization import Visualizer
from det3d.ops.primitives.primitives_cpu import voxelization
from torch_scatter import scatter

@PIPELINES.register_module
class LoadNeuralRegResult(object):
    def __init__(self,
                 flownet=None,
                 window_size=5,
                 debug=False,
                 load_temp=False):
        self.debug = debug
        self.flownet = flownet
        self.window_size = window_size
        self.grid_voxel_size = [0.2, 0.2, 0.2, 1]
        self.load_temp = load_temp

    def __call__(self, res, info):
        import time
        start_time = time.time()
        seq_id = int(info['path'].split('/')[-1].split('_')[1])
        if self.load_temp and os.path.exists(f'reg_res{seq_id}.pt'):
            print(f'loading from reg_res{seq_id}.pt')
            seq=torch.load(f'reg_res{seq_id}.pt')
            #####BUGGGGGGGGGGG#
            seq.frames.pop()
            ######
            res['lidar_sequence'] = seq
            if self.debug:
                vis = Visualizer([], [])
                center = torch.tensor(points.mean(0)[:3])
                camera = center.clone()
                camera[0] -= 10
                vis.look_at_dir(center, camera, (0,0,1))
                points = seq.points4d()
                ps_p = vis.pointcloud('points', points[:, :3])
                ps_p.add_scalar_quantity('frame % 2', points[:, -1] % 2)
                ps_p.add_scalar_quantity('frame', points[:, -1])
                velocity = seq.velocity()
                velocity = torch.tensor(velocity)
                vnorm = velocity.norm(p=2, dim=-1)
                ps_p.add_scalar_quantity('v-norm >  5 cm', vnorm > 0.05)
                ps_p.add_scalar_quantity('v-norm > 10 cm', vnorm > 0.1)
                ps_p.add_scalar_quantity('v-norm > 15 cm', vnorm > 0.15)
                ps_p.add_scalar_quantity('v-norm', vnorm)
                colors = torch.tensor([[1.,0,0], [0,1.,0]]).view(2, 3)
                vcolors = velocity[:, :2] @ colors
                ps_p.add_color_quantity('v-color', vcolors)
                ps_p.add_scalar_quantity('v-norm', vnorm)

                vis.boxes('boxes', seq.corners(),
                                   seq.classes())
                import ipdb; ipdb.set_trace()
                vis.show()
            return res, info
        
        self.net = build_neck(self.flownet).to('cuda:0')
        seq = res['lidar_sequence']
        seq_id = seq.seq_id
        num_frames = len(seq.frames)
        points = torch.tensor(seq.points4d(),
                              dtype=torch.float32).to('cuda:0')
        fusion_weight = [i / (self.window_size - 1) - 0.5
                         for i in range(self.window_size)]
        fusion_weight = np.exp(-(np.array(fusion_weight)**2)/2.0)
        fusion_weight = fusion_weight / fusion_weight.sum()
        print(f'fusion weight: {fusion_weight}')
        for f in seq.frames:
            f.velocity = torch.zeros(f.points.shape[0], 3)
            f.vweight = 0.0
        for i in range(num_frames - self.window_size + 1):
            load_path = os.path.join(
                            'work_dirs/motion_estimation',
                            f'seq_{seq_id}_frame_{i}.pt',
                        )
            assert os.path.exists(load_path)
            checkpoint = torch.load(load_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            points_xyzt = seq.points4d(i, i+self.window_size)
            points_xyzt = torch.tensor(points_xyzt, dtype=torch.float32).cuda()
            ev, ep = voxelization(points_xyzt.cpu(),
                                  torch.tensor(self.grid_voxel_size), False
                                 )[0].T.long().cuda()
            num_voxels = ev.max().item()+1
            voxels_xyzt = scatter(points_xyzt[ep], ev, dim=0,
                                  dim_size=num_voxels, reduce='mean')
            vv = self.net(voxels_xyzt)
            v = torch.zeros(points_xyzt.shape[0], 3).float().cuda()
            v[ep] = vv[ev]
            
            for d in range(self.window_size):
                mask = (points_xyzt[:, -1] == i+d)
                weight_d = fusion_weight[d]
                vd = v[mask]
                seq.frames[i+d].velocity += vd.detach().cpu().numpy()*weight_d
                seq.frames[i+d].vweight += weight_d


        res['lidar_sequence'] = seq
        end_time = time.time()
        import ipdb; ipdb.set_trace()

        if self.debug:
            print(f'load neural reg results: time={end_time-start_time:.4f}')

        del self.net

        return res, info
