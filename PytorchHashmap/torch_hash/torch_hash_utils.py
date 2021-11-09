import torch
from .torch_hash_cuda import (
    hash_insert_gpu,
    correspondence
)

class HashTable:
    def __init__(self, size=2 ** 20, dtype=torch.float32):
        self.size = size
        self.keys = torch.zeros(size, dtype=torch.long) - 1
        self.values = torch.zeros(size, 4, dtype=dtype)
        self.corres = torch.zeros(size, dtype=torch.long) - 1
        self.reverse_indices = torch.zeros(size, dtype=torch.long)
        self.device = 'cuda:0'
        self.qmin = torch.zeros(4, dtype=torch.int32).to(self.device) - 1
        self.qmax = torch.zeros(4, dtype=torch.int32).to(self.device) + 1
        self.keys = self.keys.to(self.device)
        self.values = self.values.to(self.device)
        self.corres = self.corres.to(self.device)
        self.reverse_indices = self.reverse_indices.to(self.device)

        rp = torch.tensor([999269, 999437, 999377], dtype=torch.long)
        self.rp0, self.rp1, self.rp2 = rp

    #def __del__(self):
    #    del self.keys
    #    del self.values
    #    del self.reverse_indices
    #    del self.corres
    #    del self.qmin
    
    def hash(self, voxel_coors):
        """

        Args:
            voxel_coors [N, 4]: voxel coordinates

        Returns:
            keys: [N] integers
            hash_indices: [N] integers 

        """
        insert_keys = torch.zeros_like(voxel_coors[:, 0])
        indices = torch.zeros_like(insert_keys)
        for i in range(voxel_coors.shape[1]):
            insert_keys = insert_keys * self.dims[i] + voxel_coors[:, i]
            indices = indices * self.dims[i] + voxel_coors[:, i]
            indices = (indices * self.rp0 + self.rp1) % self.rp2

        indices = indices % self.size
        return insert_keys.to(torch.long), indices.to(torch.long)

    def find_voxels(self, keys):
        """

        Args:
            keys [N]: integers

        Returns:
            voxel_coors [N, 4]: voxel coordinates
        
        """
        voxel_coors = []
        for i in range(3, -1, -1):
            voxel_coors.append(keys % dims[i])
            keys = keys / dims[i]
        voxel_coors = torch.stack(voxel_coors, dim=-1)
        return voxel_coors

    @torch.no_grad()
    def find_corres(self, ref_points, query_points, voxel_size, temporal_offset):
        """

        Args:
            ref_points (N, D): reference points
            query_points (M, D): query points
            temporal_offset: offset in temporal (last) dimension.

        Returns:
            corres (2, M): corresponding point index pairs.

        """
        ref_points = ref_points.cuda()
        query_points = query_points.cuda()
        voxel_size = voxel_size.cuda()

        points = torch.cat([ref_points, query_points], dim=0)
        voxel_coors = torch.round((points-points.min(0)[0]) / voxel_size
                                 ).long() + 1
        self.dims = (voxel_coors.max(0)[0] - voxel_coors.min(0)[0]) + 3
        
        self.keys[:] = -1
        # hash points into hash table
        hash_insert_gpu(self.keys, self.values, self.reverse_indices, self.dims,
                        voxel_coors[:ref_points.shape[0]], ref_points)
        
        self.qmin[-1] = temporal_offset
        self.qmax[-1] = temporal_offset

        corres = torch.zeros(query_points.shape[0], dtype=torch.long,
                             device=self.device) - 1

        # look up points from hash table
        correspondence(self.keys, self.values, self.reverse_indices,
                       self.dims, voxel_coors[ref_points.shape[0]:],
                       query_points, self.qmin, self.qmax, corres)

        #corres = corres.cpu()
        mask = (corres != -1)
        corres0 = torch.where(mask)[0]
        corres = torch.stack([corres0, corres[mask]], dim=0)

        return corres

if __name__ == '__main__':
    for i in range(100):
        points = torch.randn(1000000, 4) * 1000
        ht = HashTable(2**21)
        voxel_size = torch.tensor([0.2, 0.2, 0.2, 0.2])
        corres = ht.find_corres(points, points, voxel_size, 0)
        assert (corres[0] - corres[1]).abs().sum() < 1e-5
        print(corres)
