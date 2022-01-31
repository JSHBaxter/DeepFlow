#ifndef POTTS_AUGLAG1D_GPU_SOLVER_H
#define POTTS_AUGLAG1D_GPU_SOLVER_H

#include "potts_auglag_gpu_solver.h"

class POTTS_AUGLAG_GPU_SOLVER_1D : public POTTS_AUGLAG_GPU_SOLVER_BASE
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
	POTTS_AUGLAG_GPU_SOLVER_1D(
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
    
    static inline int num_buffers_full(){ return 4; }
    static inline int num_buffers_images(){ return 1; }
};

#endif
