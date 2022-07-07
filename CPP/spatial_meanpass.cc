
#include "spatial_meanpass.h"

#include "common.h"

#include "cpu_kernels.h"
#ifdef USE_CUDA
#include "gpu_kernels.h"
#endif

#include <iostream>

template<typename DEV>
SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>::SPATIAL_MEANPASS_FORWARD_SOLVER(
        const DEV & dev,
        const int n_channels,
        const int* const n,
        const int dim,
        const float* const* const r
):
MAXFLOW_ALGORITHM<DEV>(dev),
n_c(n_channels),
n(n),
dim(dim),
n_s(product(dim,n)),
r(r),
u(0),
r_eff(0)
{
    if(DEBUG_ITER) std::cout << "SPATIAL_MEANPASS_FORWARD_SOLVER<DEV> Constructor " << dim << " " << n_s << " " << n_c << std::endl;
}

template<typename DEV>
void SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>::allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over){
    if(DEBUG_ITER) std::cout << "SPATIAL_MEANPASS_FORWARD_SOLVER<DEV> Allocate" << std::endl;
    r_eff = carry_over[0];
    u = c_carry_over[0];
}

template<typename DEV>
int SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>::get_buffer_size(){
    return 0;
}

template<typename DEV>
SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>::~SPATIAL_MEANPASS_FORWARD_SOLVER(){  
}

template<typename DEV>
int SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>::get_min_iter(){
    return 10;
}

template<typename DEV>
int SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>::get_max_iter(){
    //return maximum dimension size
    int retval = 0;
    for(int d = 0; d < dim; d++)
        retval = (n[d] > retval) ? n[d] : retval;
    return retval;
}

template<typename DEV>
void SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>::init(){
    //nothing to do here
}

template<typename DEV>
void SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>::run(){
    get_effective_reg(MAXFLOW_ALGORITHM<DEV>::dev, r_eff, u, r, dim, n, n_c);
}

template<typename DEV>
void SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>::parity(float* buffer, const int n_pc, const int parity){
    parity_mask(MAXFLOW_ALGORITHM<DEV>::dev,buffer,dim,n,n_pc,parity);
}

template<typename DEV>
void SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>::parity(float* buffer, const float* other, const int n_pc, const int parity){
    parity_mask(MAXFLOW_ALGORITHM<DEV>::dev,buffer,other,dim,n,n_pc,parity);
}

template<typename DEV>
void SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>::deinit(){
    //nothing to do here
}


template<typename DEV>
SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>::SPATIAL_MEANPASS_BACKWARD_SOLVER(
    const DEV & dev,
    const int n_channels,
    const int *const n,
    const int dim,
    const float *const *const inputs,
    float *const *const g_r
):
MAXFLOW_ALGORITHM<DEV>(dev),
n_c(n_channels),
n(n),
dim(dim),
n_s(product(dim,n)),
r(inputs),
g_r(g_r),
u(0),
g_u(0),
d_y(0)
{
    if(DEBUG_ITER) std::cout << "SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV> Constructor";
    for(int i = 0; i < dim; i++)
        if(DEBUG_ITER) std::cout << r[i] << " ";
    if(DEBUG_ITER) std::cout << std::endl;
}

template<typename DEV>
void SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>::allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over){
    if(DEBUG_ITER) std::cout << "SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV> Allocate";
    u = c_carry_over[0];
    g_u = carry_over[0];
    d_y = carry_over[1];
    if(DEBUG_ITER) std::cout << " " << u << " " << g_u << " " << d_y << std::endl;
}

template<typename DEV>
int SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>::get_buffer_size(){
    return 0;
}

template<typename DEV>
SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>::~SPATIAL_MEANPASS_BACKWARD_SOLVER(){
   
}

template<typename DEV>
int SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>::get_min_iter(){
    return 10*dim;
}

template<typename DEV>
int SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>::get_max_iter(){
    //return maximum dimension size
    int retval = 0;
    for(int d = 0; d < dim; d++)
        retval = (n[d] > retval) ? n[d] : retval;
    return retval;
}

template<typename DEV>
void SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>::init(){
    for(int i = 0; i < dim; i++)
        clear_buffer(MAXFLOW_ALGORITHM<DEV>::dev, g_r[i], n_c*n_s);
}

template<typename DEV>
void SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>::deinit(){
    //nothing to do here
}

template<typename DEV>
void SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>::run(){
    run(1.0);
}

template<typename DEV>
void SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>::run(float tau){
    populate_reg_mean_gradients_and_add(MAXFLOW_ALGORITHM<DEV>::dev, d_y, u, g_r, dim, n, n_c, tau);
    get_gradient_for_u(MAXFLOW_ALGORITHM<DEV>::dev, d_y, r, g_u, dim, n, n_c, tau);
}

        