#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "potts_meanpass_gpu_solver.h"
#include "gpu_kernels.h"
#include <algorithm>

class POTTS_MEANPASS_GPU_SOLVER_3D : public POTTS_MEANPASS_GPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const int n_z;
    const float* const rx;
    const float* const ry;
    const float* const rz;
	
protected:
    int min_iter_calc(){
		return std::max(std::max(n_x,n_y), n_z);
	}
    void init_vars(){}
    void calculate_regularization(){
		get_effective_reg(dev, r_eff, u, rx, ry, rz, n_x, n_y, n_z, n_c);
	}
    void clean_up(){}
	
public:
	POTTS_MEANPASS_GPU_SOLVER_3D(
        const GPUDevice & dev,
        const int batch,
        const int sizes[5],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        const float* init_u,
        float* u,
		float** buffers_full
	):
	POTTS_MEANPASS_GPU_SOLVER_BASE(dev, batch, sizes[2]*sizes[3]*sizes[4], sizes[1], data_cost, init_u, u, buffers_full),
	n_x(sizes[2]),
	n_y(sizes[3]),
	n_z(sizes[4]),
	rx(rx_cost),
	ry(ry_cost),
	rz(rz_cost)
	{}
};


class POTTS_MEANPASS_GPU_GRADIENT_3D : public POTTS_MEANPASS_GPU_GRADIENT_BASE
{
private:
    const int n_x;
    const int n_y;
    const int n_z;
    const float* const rx;
    const float* const ry;
    const float* const rz;
    float* const g_rx;
    float* const g_ry;
    float* const g_rz;
	
protected:
    int min_iter_calc(){
		return n_x+n_y+n_z;
	}
    void init_vars(){
		clear_buffer(dev, g_rx, n_c*n_s);
		clear_buffer(dev, g_ry, n_c*n_s);
		clear_buffer(dev, g_rz, n_c*n_s);
	}
	void get_reg_gradients_and_push(float tau){
		populate_reg_mean_gradients_and_add(dev, d_y, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_c);
		get_gradient_for_u(dev, d_y, d_y, rx, ry, rz, n_x, n_y, n_z, n_c);
	}
    void clean_up(){}

public:
	POTTS_MEANPASS_GPU_GRADIENT_3D(
        const GPUDevice & dev,
        const int batch,
        const int sizes[5],
        const float* u,
        const float* g,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        float* g_d,
        float* g_rx,
        float* g_ry,
        float* g_rz,
		float** full_buffs
	) :
	POTTS_MEANPASS_GPU_GRADIENT_BASE(dev, batch, sizes[2]*sizes[3]*sizes[4], sizes[1], u, g, g_d, full_buffs),
	n_x(sizes[2]),
	n_y(sizes[3]),
	n_z(sizes[4]),
	rx(rx_cost),
	ry(ry_cost),
	rz(rz_cost),
	g_rx(g_rx),
	g_ry(g_ry),
	g_rz(g_rz)
	{}
};

template <>
struct PottsMeanpass3dFunctor<GPUDevice> {
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
      
    for(int b = 0; b < n_batches; b++)
        POTTS_MEANPASS_GPU_SOLVER_3D(d, b, sizes,
									  data_cost+ b*n_s*n_c,
									  rx_cost+ b*n_s*n_c,
									  ry_cost+ b*n_s*n_c,
									  rz_cost+ b*n_s*n_c,
									  init_u+ b*n_s*n_c,
									  u+ b*n_s*n_c,
									  buffers_full)();
      
  }
  int num_buffers_full(){ return 1; }
  int num_buffers_images(){ return 0; }
};

template <>
struct PottsMeanpass3dGradFunctor<GPUDevice>{

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
		for(int b = 0; b < n_batches; b++)
			POTTS_MEANPASS_GPU_GRADIENT_3D(d, b, sizes,
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

    int num_buffers_full(){ return 3; }
    int num_buffers_images(){ return 0; }
    
};

#endif // GOOGLE_CUDA
