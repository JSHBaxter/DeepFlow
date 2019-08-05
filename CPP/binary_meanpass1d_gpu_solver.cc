#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "binary_meanpass_gpu_solver.h"
#include "gpu_kernels.h"

class BINARY_MEANPASS_GPU_SOLVER_1D : public BINARY_MEANPASS_GPU_SOLVER_BASE
{
private:
    const int n_x;
    const float* const rx;
	
protected:
    int min_iter_calc(){
		return n_x;
	}
    void init_vars(){}
    void calculate_regularization(){
		get_effective_reg(dev, r_eff, u, rx, n_x, n_c);
	}
    void clean_up(){}
	
public:
	BINARY_MEANPASS_GPU_SOLVER_1D(
        const GPUDevice & dev,
        const int batch,
        const int sizes[3],
        const float* data_cost,
        const float* rx_cost,
        float* u,
		float** buffers_full
	):
	BINARY_MEANPASS_GPU_SOLVER_BASE(dev, batch, sizes[2], sizes[1], data_cost, u, buffers_full),
	n_x(sizes[2]),
	rx(rx_cost)
	{}
};


class BINARY_MEANPASS_GPU_GRADIENT_1D : public BINARY_MEANPASS_GPU_GRADIENT_BASE
{
private:
    const int n_x;
    const float* const rx;
    float* const g_rx;
	
protected:
    int min_iter_calc(){
		return n_x;
	}
    void init_vars(){
		clear_buffer(dev, g_rx, n_c*n_s);
	}
	void get_reg_gradients_and_push(float tau){
		populate_reg_mean_gradients_and_add(dev, d_y, u, g_rx, n_x, n_c);
		get_gradient_for_u(dev, d_y, d_y, rx, n_x, n_c);
	}
    void clean_up(){}

public:
	BINARY_MEANPASS_GPU_GRADIENT_1D(
        const GPUDevice & dev,
        const int batch,
        const int sizes[3],
        const float* u,
        const float* g,
        const float* rx_cost,
        float* g_d,
        float* g_rx,
		float** full_buffs
	) :
	BINARY_MEANPASS_GPU_GRADIENT_BASE(dev, batch, sizes[2], sizes[1], u, g, g_d, full_buffs),
	n_x(sizes[2]),
	rx(rx_cost),
	g_rx(g_rx)
	{}
};

template <>
struct BinaryMeanpass1dFunctor<GPUDevice> {
  void operator()(
	const GPUDevice& d,
	int sizes[3],
	const float* data_cost,
	const float* rx_cost,
	float* u,
	float** buffers_full,
	float** /*unused image buffers*/){
      
    int n_batches = sizes[0];
	int n_s = sizes[2];
	int n_c = sizes[1];
    for(int b = 0; b < n_batches; b++)
        BINARY_MEANPASS_GPU_SOLVER_1D(d, b, sizes,
									  data_cost+ b*n_s*n_c,
									  rx_cost+ b*n_s*n_c,
									  u+ b*n_s*n_c,
									  buffers_full)();
      
  }
  int num_buffers_full(){ return 1; }
  int num_buffers_images(){ return 0; }
};

template <>
struct BinaryMeanpass1dGradFunctor<GPUDevice>{

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
		for(int b = 0; b < n_batches; b++)
			BINARY_MEANPASS_GPU_GRADIENT_1D(d, b, sizes,
										  u+b*n_s*n_c,
										  g+b*n_s*n_c,
										  rx_cost+b*n_s*n_c,
										  g_data+b*n_s*n_c,
										  g_rx+b*n_s*n_c,
										  buffers_full)();
      
	}

    int num_buffers_full(){ return 3; }
    int num_buffers_images(){ return 0; }
    
};

#endif // GOOGLE_CUDA
