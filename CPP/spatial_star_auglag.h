
#ifndef SPATIAL_STAR_AUGLAG_SOLVER_H
#define SPATIAL_STAR_AUGLAG_SOLVER_H

#include "spatial_auglag.h"

template<typename DEV>
class SPATIAL_STAR_AUGLAG_SOLVER : public SPATIAL_AUGLAG_SOLVER<DEV>
{
protected:
    const float *const *const l;

public:
    ~SPATIAL_STAR_AUGLAG_SOLVER();
	SPATIAL_STAR_AUGLAG_SOLVER(
        const DEV & dev,
        const int n_channels,
        const int *const n,
        const int dim,
        const float *const *const rl
	);
    
    void run();
};

template class SPATIAL_STAR_AUGLAG_SOLVER<CPU_DEVICE>;
#ifdef USE_CUDA
template class SPATIAL_STAR_AUGLAG_SOLVER<CUDA_DEVICE>;
#endif

#endif
