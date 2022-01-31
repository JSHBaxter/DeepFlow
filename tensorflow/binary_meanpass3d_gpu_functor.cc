#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "binary_meanpass3d_gpu_solver.h"

template <>
struct BinaryMeanpass3dFunctor<GPUDevice> {
  void operator()(
	const GPUDevice& d,
	int sizes[5],
	const float* data_cost,
	const float* rx_cost,
	const float* ry_cost,
	const float* rz_cost,
	const float* init_u,
	float* u,
	float** buffers_full,
	float** /*unused image buffers*/){
      
    int n_batches = sizes[0];
	int n_s = sizes[2]*sizes[3]*sizes[4];
	int n_c = sizes[1];
    int data_sizes[3] = {sizes[2],sizes[3],sizes[4]};
    for(int b = 0; b < n_batches; b++)
        BINARY_MEANPASS_GPU_SOLVER_3D(d.stream(), b, n_c, data_sizes,
									  data_cost+ b*n_s*n_c,
									  rx_cost+ b*n_s*n_c,
									  ry_cost+ b*n_s*n_c,
									  rz_cost+ b*n_s*n_c,
									  init_u + (init_u ? b*n_s*n_c : 0),
									  u+ b*n_s*n_c,
									  buffers_full)();
      
  }
  int num_buffers_full(){ return BINARY_MEANPASS_GPU_SOLVER_3D::num_buffers_full(); }
  int num_buffers_images(){ return 0; }
};

template <>
struct BinaryMeanpass3dGradFunctor<GPUDevice>{

    void operator()(
		const GPUDevice& d,
		int sizes[5],
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
		float** buffers_full,
		float** /*unused image buffers*/
    ){

		int n_batches = sizes[0];
		int n_s = sizes[2]*sizes[3]*sizes[4];
		int n_c = sizes[1];
        int data_sizes[3] = {sizes[2],sizes[3],sizes[4]};
		for(int b = 0; b < n_batches; b++)
			BINARY_MEANPASS_GPU_GRADIENT_3D(d.stream(), b, n_c, data_sizes,
										  u+b*n_s*n_c,
										  g+b*n_s*n_c,
										  rx_cost+b*n_s*n_c,
										  ry_cost+b*n_s*n_c,
										  rz_cost+b*n_s*n_c,
										  g_data+b*n_s*n_c,
										  g_rx+b*n_s*n_c,
										  g_ry+b*n_s*n_c,
										  g_rz+b*n_s*n_c,
										  buffers_full)();
      
	}
    int num_buffers_full(){ return BINARY_MEANPASS_GPU_GRADIENT_3D::num_buffers_full(); }
    int num_buffers_images(){ return 0; }
    
};

#endif // GOOGLE_CUDA
