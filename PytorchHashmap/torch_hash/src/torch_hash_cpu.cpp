#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CHECK_CUDA(x) do { \
    if (!x.type().is_cuda()) { \
          fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
          exit(-1); \
        } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
    if (!x.is_contiguous()) { \
          fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
          exit(-1); \
        } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

//torch::Tensor query_corres_gpu(
//                at::Tensor ref_points, at::Tensor query_points,
//                at::Tensor voxel_size) {
//  CHECK_INPUT(ref_points);
//  CHECK_INPUT(query_points);
//  CHECK_INPUT(voxel_size);
//
//  const float* ref_points_data = ref_points.data<float>();
//  const float* query_points_data = query_points.data<float>();
//  const float* voxel_size_data = voxel_size.data<float>();
//
//  query_corres_launcher(ref_points_data, query_points_data, voxel_size_data);
//
//  torch::Tensor edges = torch::zeros({5, 3}, torch::dtype(torch::kInt32));
//  return edges;
//}
