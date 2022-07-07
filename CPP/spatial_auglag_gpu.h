#ifndef SPATIAL_AUGLAG_GPU_SOLVER_H
#define SPATIAL_AUGLAG_GPU_SOLVER_H

#include <cuda_runtime.h>
#include <cuda.h>

class SPATIAL_AUGLAG_GPU_SOLVER_BASE
{
protected:
    const cudaStream_t & dev,
    const bool channels_first;
    const int n_channels;
    const int img_size;
    
    float* const g;
    float* const div;

public:
	SPATIAL_AUGLAG_GPU_SOLVER_BASE(
        const cudaStream_t & dev,
        const bool channels_first,
        const int n_channels,
        const int img_size
	);
    
    virtual int get_min_iter() = 0;
    virtual int get_max_iter() = 0;
    
    virtual int get_number_buffer_full() = 0;
    
    virtual void init() = 0;
    virtual void run() = 0;
    virtual void deinit() = 0;
};

#endif
