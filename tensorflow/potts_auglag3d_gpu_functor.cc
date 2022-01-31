#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "potts_auglag3d_gpu_solver.h"

template <>
struct PottsAuglag3dFunctor<GPUDevice>{
  void operator()(
	const GPUDevice& d,
	int sizes[5],
	const float* data_cost,
	const float* rx_cost,
	const float* ry_cost,
	const float* rz_cost,
	float* u,
	float** buffers_full,
	float** buffers_img){

    int n_c = sizes[1];
    int n_s = sizes[2]*sizes[3]*sizes[4];
    int n_batches = sizes[0];
    int data_sizes[3] = {sizes[2],sizes[3],sizes[4]};
    for(int b = 0; b < n_batches; b++)
        POTTS_AUGLAG_GPU_SOLVER_3D(d.stream(), b, n_c, data_sizes, 
								   data_cost+b*n_s*n_c,
								   rx_cost+b*n_s*n_c,
								   ry_cost+b*n_s*n_c,
								   rz_cost+b*n_s*n_c,
								   u+b*n_s*n_c,
								   buffers_full,
								   buffers_img
								   )();
  }
  
  int num_buffers_full(){ return POTTS_AUGLAG_GPU_SOLVER_3D::num_buffers_full(); }
  int num_buffers_images(){ return POTTS_AUGLAG_GPU_SOLVER_3D::num_buffers_images(); }
};
#endif // GOOGLE_CUDA

                           