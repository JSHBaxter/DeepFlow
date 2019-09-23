#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "potts_auglag_gpu_solver.h"
#include "gpu_kernels.h"
#include <algorithm>

class POTTS_AUGLAG_GPU_SOLVER_2D : public POTTS_AUGLAG_GPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const float* rx;
    const float* ry;
	float* px;
	float* py;

protected:
    virtual int min_iter_calc(){
	}
	
    virtual void clear_spatial_flows(){
        clear_buffer(dev, px, n_c*n_s);
        clear_buffer(dev, py, n_c*n_s);
	}
	
    virtual void update_spatial_flow_calc(){
		update_spatial_flows(dev, g, div, px, py, rx, ry, n_x, n_y, n_s*n_c);
	}

public:
	POTTS_AUGLAG_GPU_SOLVER_2D(
		const GPUDevice & dev,
        const int batch,
        const int sizes[4],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        float* u,
		float** buffers_full,
		float** buffers_img
	):
	POTTS_AUGLAG_GPU_SOLVER_BASE(dev, batch, sizes[2]*sizes[3], sizes[1], data_cost, u, buffers_full, buffers_img),
    n_x(sizes[2]),
    n_y(sizes[3]),
    rx(rx_cost),
    ry(ry_cost),
	px(buffers_full[3]),
	py(buffers_full[4])
	{}
};

template <>
struct PottsAuglag2dFunctor<GPUDevice>{
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
    for(int b = 0; b < n_batches; b++)
        POTTS_AUGLAG_GPU_SOLVER_2D(d, b, sizes, 
								   data_cost+b*n_s*n_c,
								   rx_cost+b*n_s*n_c,
								   ry_cost+b*n_s*n_c,
								   u+b*n_s*n_c,
								   buffers_full,
								   buffers_img
								   )();
  }
  
  int num_buffers_full(){ return 5; }
  int num_buffers_images(){ return 1; }
};
#endif // GOOGLE_CUDA

                           