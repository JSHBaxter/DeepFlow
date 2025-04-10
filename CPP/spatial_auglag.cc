
#include "spatial_auglag.h"
#include "common.h"

#include "cpu_kernels.h"
#include "cpu_kernels_auglag.h"
#ifdef USE_CUDA
#include "gpu_kernels.h"
#include "gpu_kernels_auglag.h"
#endif

#include <iostream>

template<typename DEV>
SPATIAL_AUGLAG_SOLVER<DEV>::SPATIAL_AUGLAG_SOLVER(
    const DEV & dev,
    const int n_channels,
    const int *const n,
    const int dim,
    const float *const *const r
):
MAXFLOW_ALGORITHM<DEV>(dev),
n_c(n_channels),
n(n),
dim(dim),
n_s(product(dim,n)),
r(r),
g(0),
div(0),
p(0)
{
    if(DEBUG_ITER) std::cout << "SPATIAL_AUGLAG_SOLVER Constructor" << dim << " " << n_s << " " << n_c << std::endl;
}

template<typename DEV>
void SPATIAL_AUGLAG_SOLVER<DEV>::allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over){
    if(DEBUG_ITER) std::cout << "SPATIAL_AUGLAG_SOLVER Allocate" << std::endl;
    g = carry_over[0];
    div = carry_over[1];
    if(p) delete p;
    p = new float* [dim];
    for(int i = 0; i < dim; i++)
        p[i] = buffer+i*n_s*n_c;
}

template<typename DEV>
int SPATIAL_AUGLAG_SOLVER<DEV>::get_buffer_size(){
    return n_s*n_c*dim; //enough room for the flow buffers
}

template<typename DEV>
SPATIAL_AUGLAG_SOLVER<DEV>::~SPATIAL_AUGLAG_SOLVER(){
    delete p;
}

template<typename DEV>
int SPATIAL_AUGLAG_SOLVER<DEV>::get_min_iter(){
    return 10*dim;
}

template<typename DEV>
int SPATIAL_AUGLAG_SOLVER<DEV>::get_max_iter(){
    //return maximum dimension size
    int retval = 0;
    for(int d = 0; d < dim; d++)
        retval = (n[d] > retval) ? n[d] : retval;
    return n_s;
}

template<typename DEV>
void SPATIAL_AUGLAG_SOLVER<DEV>::init(){
    clear_buffer(MAXFLOW_ALGORITHM<DEV>::dev, p[0], dim*n_c*n_s);
    clear_buffer(MAXFLOW_ALGORITHM<DEV>::dev, div, n_s*n_c);
}

template<typename DEV>
void SPATIAL_AUGLAG_SOLVER<DEV>::run(){
    update_spatial_flows(MAXFLOW_ALGORITHM<DEV>::dev, g, div, p, r, dim, n, n_c);
}

template<typename DEV>
void SPATIAL_AUGLAG_SOLVER<DEV>::deinit(){
    //nothing to do here
}
