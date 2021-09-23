#ifndef PRIMITIVES_CPU_H
#define PRIMITIVES_CPU_H

#include <torch/serialize/tensor.h>
#include <vector>

std::vector<torch::Tensor> query_voxel_neighbors(
    at::Tensor point_tensor, at::Tensor voxel_size
    );

#endif
