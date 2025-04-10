
#include "potts_meanpass_solver.h"

#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "common.h"

#include "cpu_kernels.h"
#include "cpu_kernels_meanpass.h"
#ifdef USE_CUDA
#include "gpu_kernels.h"
#include "gpu_kernels_meanpass.h"
#endif

template<typename DEV>
POTTS_MEANPASS_SOLVER<DEV>::POTTS_MEANPASS_SOLVER(
    const DEV & dev,
    const int dim,
    const int* dims,
    const int n_c,
    const float* const* const inputs,
    float* const u) :
MAXFLOW_ALGORITHM<DEV>(dev),
n_c(n_c),
n_s(product(dim,dims)),
init_u(inputs[0]),
data(inputs[1]),
u(u),
r_eff(0)
{
    if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_SOLVER Constructor " << dim << " " << n_s << " " << n_c << " " << inputs[0] << " " << u << std::endl;
    spatial_flow = new SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>(dev,n_c,dims,dim,inputs+2);
    MAXFLOW_ALGORITHM<DEV>::construct();
}

template<typename DEV>
void POTTS_MEANPASS_SOLVER<DEV>::allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over){
    if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_SOLVER Allocate" << std::endl;
    r_eff = buffer;
    float* carry_over_tmp[] = {r_eff};
    const float* c_carry_over_tmp[] = {u};
    spatial_flow->allocate_buffers(buffer+n_s*n_c,carry_over_tmp,c_carry_over_tmp);
}

template<typename DEV>
int POTTS_MEANPASS_SOLVER<DEV>::get_buffer_size(){
    return n_s * n_c + spatial_flow->get_buffer_size();
}

template<typename DEV>
POTTS_MEANPASS_SOLVER<DEV>::~POTTS_MEANPASS_SOLVER(){
    delete spatial_flow;
}

//perform one iteration of the algorithm
template<typename DEV>
void POTTS_MEANPASS_SOLVER<DEV>::block_iter(const int parity){
    float max_change = 0.0f;
    spatial_flow->run();
    softmax(MAXFLOW_ALGORITHM<DEV>::dev, data, r_eff, r_eff, n_s, n_c);
    spatial_flow->parity(r_eff,u,n_c,parity);
    change_to_diff(MAXFLOW_ALGORITHM<DEV>::dev, u, r_eff, n_s*n_c, tau);
}

template<typename DEV>
void POTTS_MEANPASS_SOLVER<DEV>::run(){
    if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_SOLVER Init" << std::endl;

    //initialize variables
    if(init_u)
        copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, init_u, u, n_s*n_c);
    else
        softmax(MAXFLOW_ALGORITHM<DEV>::dev, data, 0, u, n_s, n_c);
    spatial_flow->init();

    // iterate in blocks
    int min_iter = spatial_flow->get_min_iter();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = spatial_flow->get_max_iter();
    if (max_loop < 10)
        max_loop = 10;
    
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter(iter&1);

        float max_change = max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff, n_c*n_s);
        if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_SOLVER Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter(iter&1);


    //calculate the effective regularization
    spatial_flow->run();
    
    //get final output
    add_then_store(MAXFLOW_ALGORITHM<DEV>::dev, data, r_eff, u, n_s*n_c);
        
    //deallocate temporary buffers
    spatial_flow->deinit();
}


template<typename DEV>
POTTS_MEANPASS_GRADIENT<DEV>::POTTS_MEANPASS_GRADIENT(
    const DEV & dev,
    const int dim,
    const int* dims,
    const int n_c,
    const float *const *const inputs,
    float *const *const outputs) :
MAXFLOW_ALGORITHM<DEV>(dev),
n_c(n_c),
n_s(product(dim,dims)),
grad(inputs[0]),
logits(inputs[1]),
g_data(outputs[0]),
d_y(0),
g_u(0),
u(0)
{
    if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_GRADIENT Constructor " << dim << " " << n_s << " " << n_c << " " << grad << " " << logits << std::endl;
    //std::cout << n_s << " " << n_c << " " << grad << " " << logits << " " << g_data << " " << d_y << " " << g_u << " " << this->u << std::endl;
    spatial_flow = new SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>(MAXFLOW_ALGORITHM<DEV>::dev, n_c, dims, dim, inputs+2, outputs+1);
    MAXFLOW_ALGORITHM<DEV>::construct();
}

template<typename DEV>
POTTS_MEANPASS_GRADIENT<DEV>::~POTTS_MEANPASS_GRADIENT(){
    delete spatial_flow;
}

template<typename DEV>
void POTTS_MEANPASS_GRADIENT<DEV>::allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over){
    if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_GRADIENT Allocate" << std::endl;
    u = buffer;
    g_u = u+n_s*n_c;
    d_y = g_u+n_s*n_c;
    float* carry_over_tmp[] = {g_u, d_y};
    const float* c_carry_over_tmp[] = {u};
    std::cout << "\t" << u << " " << g_u << " " << d_y << std::endl;
    spatial_flow->allocate_buffers(d_y+n_s*n_c,carry_over_tmp,c_carry_over_tmp);
}

template<typename DEV>
int POTTS_MEANPASS_GRADIENT<DEV>::get_buffer_size(){
    return 3*n_s*n_c + spatial_flow->get_buffer_size();
}

//perform one iteration of the algorithm
template<typename DEV>
void POTTS_MEANPASS_GRADIENT<DEV>::block_iter(){
    //untangle softmax derivative and add to data term gradient
    untangle_softmax(MAXFLOW_ALGORITHM<DEV>::dev, g_u, u, d_y, n_s, n_c);
    
    // populate data gradient
    inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, d_y, g_data, n_s*n_c, tau);
    
    //add to regularization gradients and push back
    spatial_flow->run(tau);
}

template<typename DEV>
void POTTS_MEANPASS_GRADIENT<DEV>::run(){
    if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_GRADIENT Init 1" << std::endl;
    
    //get initial maximum value of teh gradient for convgencence purposes
    const float init_grad_max = max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, grad, n_s*n_c);

    //initialize variables
    spatial_flow->init();
    softmax(MAXFLOW_ALGORITHM<DEV>::dev, logits, 0, u, n_s, n_c);
    clear_buffer(MAXFLOW_ALGORITHM<DEV>::dev, g_u, n_s*n_c);
    copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, grad, d_y, n_s*n_c);
    
    //get initial gradient for the data and regularization terms
    copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, grad, g_data, n_s*n_c);
    spatial_flow->run(1.0f);
    if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_GRADIENT Init 6" << std::endl;

    // iterate in blocks
    int min_iter = spatial_flow->get_min_iter();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = spatial_flow->get_max_iter();
    if (max_loop < 10)
        max_loop = 10;
    
    if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_GRADIENT Starting loop" << std::endl;
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        float max_change = max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, g_u, n_s*n_c);
        if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_GRADIENT Iter " << i << ": " << max_change << std::endl;
        if(max_change < beta*init_grad_max)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();
        
    //deallocate temporary buffers
    spatial_flow->deinit();
}
