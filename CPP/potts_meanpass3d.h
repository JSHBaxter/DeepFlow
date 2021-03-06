// potts_meanpass3d.h
#ifndef POTTS_MEANPASS3D_H_
#define POTTS_MEANPASS3D_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

template <typename Device>
struct PottsMeanpass3dFunctor {
  void operator()(
      const Device& d,
      int size[5],
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      const float* rz_cost,
      const float* init,
      float* out,
      float** buffers_full,
      float** buffers_images);
    int num_buffers_full();
    int num_buffers_images();
};

template <typename Device>
struct PottsMeanpass3dGradFunctor {
  void operator()(
      const Device& d,
      int size[5],
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      const float* rz_cost,
      const float* u,
      const float* g,
      float* g_data,
      float* rx_data,
      float* ry_data,
      float* rz_data,
      float** buffers_full,
      float** buffers_images);
    int num_buffers_full();
    int num_buffers_images();
};

#endif // POTTS_MEANPASS3D_H_