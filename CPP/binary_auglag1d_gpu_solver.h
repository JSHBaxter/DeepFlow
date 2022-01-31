
#ifndef BINARY_AUGLAG_GPU_SOLVER_1D_
#define BINARY_AUGLAG_GPU_SOLVER_1D_

#include "binary_auglag_gpu_solver.h"

class BINARY_AUGLAG_GPU_SOLVER_1D : public BINARY_AUGLAG_GPU_SOLVER_BASE
{
private:
    const int n_x;
    const float* rx;
	float* px;

protected:
    virtual int min_iter_calc();
    virtual void clear_spatial_flows();
    virtual void update_spatial_flow_calc();

public:
	BINARY_AUGLAG_GPU_SOLVER_1D(
		const cudaStream_t & dev,
        const int batch,
        const int n_c,
        const int sizes[1],
        const float* data_cost,
        const float* rx_cost,
        float* u,
		float** buffers_full,
		float** buffers_img
	);

  static int num_buffers_full(){ return 5; }
  static int num_buffers_images(){ return 0; }
};

#endif
