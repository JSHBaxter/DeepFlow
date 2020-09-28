#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "binary_auglag_gpu_solver.h"
#include "gpu_kernels.h"

class BINARY_AUGLAG_GPU_SOLVER_1D : public BINARY_AUGLAG_GPU_SOLVER_BASE
{
private:
    const int n_x;
    const float* rx;
	float* px;

protected:
    virtual int min_iter_calc(){
		return n_x;
	}
	
    virtual void clear_spatial_flows(){
        clear_buffer(dev, px, n_c*n_s);
	}
	
    virtual void update_spatial_flow_calc(){
		update_spatial_flows(dev, g, div, px, rx, n_x, n_s*n_c);
	}

public:
	BINARY_AUGLAG_GPU_SOLVER_1D(
		const GPUDevice & dev,
        const int batch,
        const int sizes[3],
        const float* data_cost,
        const float* rx_cost,
        float* u,
		float** buffers_full,
		float** buffers_img
	):
	BINARY_AUGLAG_GPU_SOLVER_BASE(dev, batch, sizes[2], sizes[1], data_cost, u, buffers_full, buffers_img),
    n_x(sizes[2]),
    rx(rx_cost),
	px(buffers_full[4])
	{}
};

template <>
struct BinaryAuglag1dFunctor<GPUDevice>{
  void operator()(
	const GPUDevice& d,
	int sizes[3],
	const float* data_cost,
	const float* rx_cost,
	float* u,
	float** buffers_full,
	float** buffers_img){

    int n_c = sizes[1];
    int n_s = sizes[2];
    int n_batches = sizes[0];
    for(int b = 0; b < n_batches; b++)
        BINARY_AUGLAG_GPU_SOLVER_1D(d, b, sizes, 
								   data_cost+b*n_s*n_c,
								   rx_cost+b*n_s*n_c,
								   u+b*n_s*n_c,
								   buffers_full,
								   buffers_img
								   )();
  }
  
  int num_buffers_full(){ return 5; }
  int num_buffers_images(){ return 0; }
};
#endif // GOOGLE_CUDA
