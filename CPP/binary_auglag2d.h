// binary_auglag2d.h
#ifndef BINARY_AUGLAG2D_H_
#define BINARY_AUGLAG2D_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

template <typename Device>
struct BinaryAuglag2dFunctor {
  void operator()(
      const Device& d,
      int size[4],
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      float* out,
      float** buffers_full,
      float** buffers_images);
    int num_buffers_full();
    int num_buffers_images();
};

#endif // BINARY_AUGLAG2D_H_
