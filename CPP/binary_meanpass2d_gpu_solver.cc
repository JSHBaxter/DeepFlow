#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "binary_meanpass_gpu_solver.h"
#include "gpu_kernels.h"

class BINARY_MEANPASS_GPU_SOLVER_2D : public BINARY_MEANPASS_GPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const float* const rx;
    const float* const ry;
	
protected:
    int min_iter_calc(){
		return n_x+n_y;
	}
    void init_vars(){}
    void calculate_regularization(){
		get_effective_reg(dev, r_eff, u, rx, ry, n_x, n_y, n_c);
	}
    void parity_mask_buffer(float* buffer, const int parity){
        parity_mask(dev,buffer,n_x,n_y,n_c,parity);
    }
    void parity_merge_buffer(float* buffer, const float* other, const int parity){
        parity_mask(dev,buffer,other,n_x,n_y,n_c,parity);
    }
    void clean_up(){}
	
public:
	BINARY_MEANPASS_GPU_SOLVER_2D(
        const GPUDevice & dev,
        const int batch,
        const int sizes[4],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
		const float* init_u,
        float* u,
		float** buffers_full
	):
	BINARY_MEANPASS_GPU_SOLVER_BASE(dev, batch, sizes[2]*sizes[3], sizes[1], data_cost, init_u, u, buffers_full),
	n_x(sizes[2]),
	n_y(sizes[3]),
	rx(rx_cost),
	ry(ry_cost)
	{std::cout << rx_cost << std::endl;
	 std::cout << ry_cost << std::endl;}
};


class BINARY_MEANPASS_GPU_GRADIENT_2D : public BINARY_MEANPASS_GPU_GRADIENT_BASE
{
private:
    const int n_x;
    const int n_y;
    const float* const rx;
    const float* const ry;
    float* const g_rx;
    float* const g_ry;
	
protected:
    int min_iter_calc(){
		return n_x+n_y;
	}
    void init_vars(){
		clear_buffer(dev, g_rx, n_c*n_s);
		clear_buffer(dev, g_ry, n_c*n_s);
	}
	void get_reg_gradients_and_push(float tau){
		populate_reg_mean_gradients_and_add(dev, d_y, u, g_rx, g_ry, n_x, n_y, n_c);
		get_gradient_for_u(dev, d_y, d_y, rx, ry, n_x, n_y, n_c);
	}
    void clean_up(){}

public:
	BINARY_MEANPASS_GPU_GRADIENT_2D(
        const GPUDevice & dev,
        const int batch,
        const int sizes[4],
        const float* u,
        const float* g,
        const float* rx_cost,
        const float* ry_cost,
        float* g_d,
        float* g_rx,
        float* g_ry,
		float** full_buffs
	) :
	BINARY_MEANPASS_GPU_GRADIENT_BASE(dev, batch, sizes[2]*sizes[3], sizes[1], u, g, g_d, full_buffs),
	n_x(sizes[2]),
	n_y(sizes[3]),
	rx(rx_cost),
	ry(ry_cost),
	g_rx(g_rx),
	g_ry(g_ry)
	{}
};

template <>
struct BinaryMeanpass2dFunctor<GPUDevice> {
  void operator()(
	const GPUDevice& d,
	int sizes[4],
	const float* data_cost,
	const float* rx_cost,
	const float* ry_cost,
	const float* init_u,
	float* u,
	float** buffers_full,
	float** /*unused image buffers*/){
      
    int n_batches = sizes[0];
	int n_s = sizes[2]*sizes[3];
	int n_c = sizes[1];
    for(int b = 0; b < n_batches; b++)
        BINARY_MEANPASS_GPU_SOLVER_2D(d, b, sizes,
									  data_cost+ b*n_s*n_c,
									  rx_cost+ b*n_s*n_c,
									  ry_cost+ b*n_s*n_c,
									  init_u + (init_u ? b*n_s*n_c : 0),
									  u+ b*n_s*n_c,
									  buffers_full)();
      
  }
  int num_buffers_full(){ return 1; }
  int num_buffers_images(){ return 0; }
};

template <>
struct BinaryMeanpass2dGradFunctor<GPUDevice>{

    void operator()(
		const GPUDevice& d,
		int sizes[4],
		const float* data_cost,
		const float* rx_cost,
		const float* ry_cost,
		const float* u,
		const float* g,
		float* g_data,
		float* g_rx,
		float* g_ry,
		float** buffers_full,
		float** /*unused image buffers*/
    ){

		int n_batches = sizes[0];
		int n_s = sizes[2]*sizes[3];
		int n_c = sizes[1];
		for(int b = 0; b < n_batches; b++)
			BINARY_MEANPASS_GPU_GRADIENT_2D(d, b, sizes,
										  u+b*n_s*n_c,
										  g+b*n_s*n_c,
										  rx_cost+b*n_s*n_c,
										  ry_cost+b*n_s*n_c,
										  g_data+b*n_s*n_c,
										  g_rx+b*n_s*n_c,
										  g_ry+b*n_s*n_c,
										  buffers_full)();
      
	}

    int num_buffers_full(){ return 3; }
    int num_buffers_images(){ return 0; }
    
};

#endif // GOOGLE_CUDA
