#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "potts_meanpass1d_gpu_solver.h"

template <>
struct PottsMeanpass1dFunctor<GPUDevice> {
  void operator()(
	const GPUDevice& d,
	int sizes[3],
	const float* data_cost,
	const float* rx_cost,
    const float* init_u,
	float* u,
	float** buffers_full,
	float** /*unused image buffers*/){
      
    int n_batches = sizes[0];
	int n_s = sizes[2];
	int n_c = sizes[1];
    int data_sizes[1] = {sizes[2]};
    for(int b = 0; b < n_batches; b++)
        POTTS_MEANPASS_GPU_SOLVER_1D(d.stream(), b, n_c, data_sizes,
									  data_cost+ b*n_s*n_c,
									  rx_cost+ b*n_s*n_c,
									  init_u + (init_u ? b*n_s*n_c : 0),
									  u+ b*n_s*n_c,
									  buffers_full)();
      
  }
  int num_buffers_full(){ return POTTS_MEANPASS_GPU_SOLVER_1D::num_buffers_full(); }
  int num_buffers_images(){ return POTTS_MEANPASS_GPU_SOLVER_1D::num_buffers_images(); }
};

template <>
struct PottsMeanpass1dGradFunctor<GPUDevice>{

    void operator()(
		const GPUDevice& d,
		int sizes[3],
		const float* data_cost,
		const float* rx_cost,
		const float* u,
		const float* g,
		float* g_data,
		float* g_rx,
		float** buffers_full,
		float** /*unused image buffers*/
    ){

		int n_batches = sizes[0];
		int n_s = sizes[2];
		int n_c = sizes[1];
        int data_sizes[1] = {sizes[2]};
		for(int b = 0; b < n_batches; b++)
			POTTS_MEANPASS_GPU_GRADIENT_1D(d.stream(), b, n_c, data_sizes,
										  u+b*n_s*n_c,
										  g+b*n_s*n_c,
										  rx_cost+b*n_s*n_c,
										  g_data+b*n_s*n_c,
										  g_rx+b*n_s*n_c,
										  buffers_full)();
      
	}

    int num_buffers_full(){ return POTTS_MEANPASS_GPU_GRADIENT_1D::num_buffers_full(); }
    int num_buffers_images(){ return POTTS_MEANPASS_GPU_GRADIENT_1D::num_buffers_images(); }
    
};

#endif // GOOGLE_CUDA
