// binary_auglag1d.h
#ifndef BINARY_AUGLAG1D_H_
#define BINARY_AUGLAG1D_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

template <typename Device>
struct BinaryAuglag1dFunctor {
  void operator()(
      const Device& d,
      int size[3],
      const float* data_cost,
      const float* rx_cost,
      float* out,
      float** buffers_full,
      float** buffers_images);
    int num_buffers_full();
    int num_buffers_images();
};

#endif // BINARY_AUGLAG1D_H_
