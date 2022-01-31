#ifndef BINARY_AUGLAG2D_GPU_SOLVER_H
#define BINARY_AUGLAG2D_GPU_SOLVER_H

#include "binary_auglag_gpu_solver.h"

class BINARY_AUGLAG_GPU_SOLVER_2D : public BINARY_AUGLAG_GPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const float* rx;
    const float* ry;
	float* px;
	float* py;

protected:
    virtual int min_iter_calc();
    virtual void clear_spatial_flows();
    virtual void update_spatial_flow_calc();
	
public:
	BINARY_AUGLAG_GPU_SOLVER_2D(
		const cudaStream_t & dev,
        const int batch,
        const int n_c,
        const int sizes[2],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        float* u,
		float** buffers_full,
		float** buffers_img
	);

  static int num_buffers_full(){ return 6; }
  static int num_buffers_images(){ return 0; }
};

#endif
