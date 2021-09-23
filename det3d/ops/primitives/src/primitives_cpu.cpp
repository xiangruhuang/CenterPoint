#include <stdio.h>
#include <math.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include "primitives_cpu.h"
#include <unordered_map>

struct Voxel {
  int x, y, z;
  Voxel() {}
  Voxel(int _x, int _y, int _z) {
    x = _x; y = _y; z = _z;
  }
  Voxel(double _x, double _y, double _z) {
    x = floor(_x); y = floor(_y); z = floor(_z);
  }
  inline bool operator =(const Voxel &v) const{
    return (x == v.x) && (y == v.y) && (z == v.z);
  }
};

struct Point {
  float x, y, z;
  Point() {};
  Point(double _x, double _y, double _z) {
    x = _x; y = _y; z = _z;
  }
  inline Point operator /(const Point &p) const{
    return Point(x/p.x, y/p.y, z/p.z);
  }
};


using namespace std;

vector<torch::Tensor> query_voxel_neighbors(
    at::Tensor point_tensor, at::Tensor voxel_size) {
  unordered_map<int, vector<int>> umap;
  int N = point_tensor.size(0);
  float* vs = voxel_size.data_ptr<float>();
  vector<Voxel> voxels;
  int maxx=-1e10, maxy=-1e10, maxz=-1e10;
  int minx=1e10, miny=1e10, minz=1e10;
  
  auto points = point_tensor.accessor<float, 2>();

  for (int i = 0; i < N; i++) {
    Point p;
    p.x = points[i][0];
    p.y = points[i][1];
    p.z = points[i][2];
    Voxel vi(p.x/vs[0], p.y/vs[1], p.z/vs[2]);
    if (vi.x < minx) {
      minx = vi.x;
    }
    if (vi.y < miny) {
      miny = vi.y;
    }
    if (vi.z < minz) {
      minz = vi.z;
    }
    if (vi.x > maxx) {
      maxx = vi.x;
    }
    if (vi.y > maxy) {
      maxy = vi.y;
    }
    if (vi.z > maxz) {
      maxz = vi.z;
    }
  }
  minx -= 1;
  miny -= 1;
  minz -= 1;
  maxx += 1;
  maxy += 1;
  maxz += 1;
  int dimx = maxx - minx + 1;
  int dimy = maxy - miny + 1;
  int dimz = maxz - minz + 1;
  for (int i = 0; i < N; i++) {
    Point p;
    p.x = points[i][0];
    p.y = points[i][1];
    p.z = points[i][2];
    
    Voxel vi(p.x/vs[0], p.y/vs[1], p.z/vs[2]);
    vi.x -= minx;
    vi.y -= miny;
    vi.z -= minz;
    int hash_id = (vi.x*dimy + vi.y)*dimz + vi.z;
    auto it = umap.find(hash_id);
    if (it != umap.end()) {
      it->second.push_back(i);
    } else {
      vector<int> vp;
      vp.push_back(i);
      umap.insert(std::move(make_pair(hash_id, vp)));
    }
  }
  int num_amb_edges = 0;
  int i = 0;
  vector<vector<int>> adjs(umap.size());
  for (auto it = umap.begin();
       i < umap.size(); i++, it++) {
    int hash_id = it->first;
    int vz = hash_id % dimz;
    hash_id = hash_id / dimz;
    int vy = hash_id % dimy;
    int vx = hash_id / dimy;

    for (int dx = -1; dx <= 1; dx++) {
      for (int dy = -1; dy <= 1; dy++) {
        for (int dz = -1; dz <= 1; dz++) {
          if (dx == 0 && dy == 0 && dz == 0) {
            continue;
          }
          int hash_id_ijk = ((vx+dx)*dimy + (vy+dy))*dimz + vz+dz;
          auto it2 = umap.find(hash_id_ijk);
          if (it2 == umap.end()) {
            continue;
          }
          adjs[i].push_back(hash_id_ijk);
          num_amb_edges += it2->second.size();
        }
      }
    }
  }
  torch::Tensor edges_tensor = torch::zeros(
    {N, 2}, torch::dtype(torch::kInt32)
  );
  torch::Tensor amb_edges_tensor = torch::zeros(
    {num_amb_edges, 2}, torch::dtype(torch::kInt32)
  );
  auto edges_acc = edges_tensor.accessor<int, 2>();
  auto amb_edges_acc = amb_edges_tensor.accessor<int, 2>();
  int edge_size = 0, amb_edge_size = 0;
  i = 0;
  for (auto it = umap.begin();
       i < umap.size(); i++, it++) {
    for (int &idx : it->second) {
      edges_acc[edge_size][0] = i;
      edges_acc[edge_size++][1] = idx;
    }
    for (int hash_id_ijk : adjs[i]) {
      auto it2 = umap.find(hash_id_ijk);
      for (int &amb_idx : it2->second) {
        amb_edges_acc[amb_edge_size][0] = i;
        amb_edges_acc[amb_edge_size++][1] = amb_idx;
      }
    }
  }

  return {edges_tensor, amb_edges_tensor};
}
