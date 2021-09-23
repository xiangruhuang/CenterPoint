#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>

#include "primitives_cpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("query_voxel_neighbors", &query_voxel_neighbors, "query voxel neighbors");
}
