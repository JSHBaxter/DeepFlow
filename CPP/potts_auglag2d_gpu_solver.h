
#ifndef POTTS_AUGLAG_GPU_SOLVER_2D_
#define POTTS_AUGLAG_GPU_SOLVER_2D_

#include "potts_auglag_gpu_solver.h"

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
    virtual int min_iter_calc();
    virtual void clear_spatial_flows();
    virtual void update_spatial_flow_calc();

public:
	POTTS_AUGLAG_GPU_SOLVER_2D(
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
    
    static inline int num_buffers_full(){ return 5; }
    static inline int num_buffers_images(){ return 1; }
};

#endif
