#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "binary_auglag2d_gpu_solver.h"

template <>
struct BinaryAuglag2dFunctor<GPUDevice>{
  void operator()(
	const GPUDevice& d,
	int sizes[4],
	const float* data_cost,
	const float* rx_cost,
	const float* ry_cost,
	float* u,
	float** buffers_full,
	float** buffers_img){

    int n_c = sizes[1];
    int n_s = sizes[2]*sizes[3];
    int n_batches = sizes[0];
    int data_sizes[2] = {sizes[2],sizes[3]};
    for(int b = 0; b < n_batches; b++)
        BINARY_AUGLAG_GPU_SOLVER_2D(d.stream(), b, n_c, data_sizes, 
								   data_cost+b*n_s*n_c,
								   rx_cost+b*n_s*n_c,
								   ry_cost+b*n_s*n_c,
								   u+b*n_s*n_c,
								   buffers_full,
								   buffers_img
								   )();
  }
  
  int num_buffers_full(){ return BINARY_AUGLAG_GPU_SOLVER_2D::num_buffers_full(); }
  int num_buffers_images(){ return BINARY_AUGLAG_GPU_SOLVER_2D::num_buffers_images(); }
};

#endif // GOOGLE_CUDA
