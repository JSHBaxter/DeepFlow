
#include "spatial_star_auglag.h"
#include "common.h"

#include "cpu_kernels.h"
#include "cpu_kernels_auglag.h"
#ifdef USE_CUDA
#include "gpu_kernels.h"
#include "gpu_kernels_auglag.h"
#endif

#include <iostream>

template<typename DEV>
SPATIAL_STAR_AUGLAG_SOLVER<DEV>::SPATIAL_STAR_AUGLAG_SOLVER(
    const DEV & dev,
    const int n_channels,
    const int *const n,
    const int dim,
    const float *const *const rl
):
SPATIAL_AUGLAG_SOLVER<DEV>(dev, n_channels, n, dim, rl),
l(rl+dim)
{
    if(DEBUG_ITER) std::cout << "SPATIAL_STAR_AUGLAG_SOLVER Constructor" << dim << " " << this->n_s << " " << this->n_c << std::endl;
}

template<typename DEV>
void SPATIAL_STAR_AUGLAG_SOLVER<DEV>::run(){
    update_spatial_star_flows(MAXFLOW_ALGORITHM<DEV>::dev, this->g, this->div, this->p, this->r, this->l, this->dim, this->n, this->n_c);
}
