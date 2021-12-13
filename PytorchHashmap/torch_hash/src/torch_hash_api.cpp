#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "torch_hash.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hash_insert_gpu", &hash_insert_gpu, "hashtable insert in GPU");
  m.def("correspondence", &correspondence, "compute correspondence");
  m.def("voxel_graph_gpu", &voxel_graph_gpu, "compute graph between voxels");
  m.def("track_graphs_gpu", &track_graphs_gpu, "track graph clusters");
}
