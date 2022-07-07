
#ifndef SPATIAL_AUGLAG_SOLVER_H
#define SPATIAL_AUGLAG_SOLVER_H

#include "algorithm.h"

template<typename DEV>
class SPATIAL_AUGLAG_SOLVER : public MAXFLOW_ALGORITHM<DEV>
{
protected:
    const int n_c;
    const int n_s;
    const int* const n;
    const int dim;
    
    float* g;
    float* div;
    const float *const *const r;
    float** p;

public:
    ~SPATIAL_AUGLAG_SOLVER();
	SPATIAL_AUGLAG_SOLVER(
        const DEV & dev,
        const int n_channels,
        const int *const n,
        const int dim,
        const float *const *const r
	);
    
    virtual int get_min_iter();
    virtual int get_max_iter();
    
    virtual void allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over);
    virtual int get_buffer_size();
    
    virtual void init();
    virtual void run();
    virtual void deinit();
};

template class SPATIAL_AUGLAG_SOLVER<CPU_DEVICE>;
#ifdef USE_CUDA
template class SPATIAL_AUGLAG_SOLVER<CUDA_DEVICE>;
#endif

#endif
