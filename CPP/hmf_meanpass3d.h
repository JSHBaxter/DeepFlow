// kernel_example.h
#ifndef HMF_MEANPASS3D_H_
#define HMF_MEANPASS3D_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "hmf_trees.h"

template <typename Device>
struct HmfMeanpass3dFunctor {
  void operator()(
      const Device& d,
      int size[7],
      const int* parentage,
      const int* data_index,
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      const float* rz_cost,
      float* out,
      float** buffers_full,
      float** buffers_images);
    int num_buffers_full();
    int num_buffers_branch();
    int num_buffers_data();
    int num_buffers_images();
};

template <typename Device>
struct HmfMeanpass3dGradFunctor {
  void operator()(
      const Device& d,
      int size[7],
      const int* parentage,
      const int* data_index,
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      const float* rz_cost,
      const float* u,
      const float* g,
      float* g_data,
      float* g_rx,
      float* g_ry,
      float* g_rz,
      int* g_par,
      int* g_didx,
      float** buffers_full,
      float** buffers_images);
    int num_buffers_full();
    int num_buffers_branch();
    int num_buffers_data();
    int num_buffers_images();
};

#endif // HMF_MEANPASS3D_H_