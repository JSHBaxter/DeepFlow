// kernel_example.h
#ifndef HMF_AUGLAG1D_H_
#define HMF_AUGLAG1D_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "hmf_trees.h"

template <typename Device>
struct HmfAuglag1dFunctor {
  void operator()(
      const Device& d,
      int size[5],
      const int* parentage,
      const int* data_index,
      const float* data_cost,
      const float* rx_cost,
      float* out,
      float** buffers_full,
      float** buffers_images);
    int num_buffers_full();
    int num_buffers_branch();
    int num_buffers_data();
    int num_buffers_images();
};

#endif // HMF_AUGLAG1D_H_
