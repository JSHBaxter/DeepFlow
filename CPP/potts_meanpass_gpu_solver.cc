#include "potts_meanpass_gpu_solver.h"
#include <math.h>
#include <iostream>
#include <limits>

#include "gpu_kernels.h"

POTTS_MEANPASS_GPU_SOLVER_BASE::POTTS_MEANPASS_GPU_SOLVER_BASE(
    const cudaStream_t & dev,
    const int batch,
    const int n_s,
    const int n_c,
    const float* data_cost,
    const float* init_u,
    float* u,
    float** full_buffs) :
dev(dev),
b(batch),
n_c(n_c),
n_s(n_s),
data(data_cost),
r_eff(full_buffs[0]),
u(u)
{
    if(init_u)
        copy_buffer(dev, init_u, u, n_s*n_c);
    else
        softmax(dev, data, 0, u, n_s, n_c);
        
    //std::cout << n_s << " " << n_c << " " << data_cost << " " << r_eff << std::endl;
}

//perform one iteration of the algorithm
void POTTS_MEANPASS_GPU_SOLVER_BASE::block_iter(const int parity){
    float max_change = 0.0f;
    calculate_regularization();
    softmax(dev, data, r_eff, r_eff, n_s, n_c);
    parity_merge_buffer(r_eff,u,parity);
    change_to_diff(dev, u, r_eff, n_s*n_c, tau);
}

void POTTS_MEANPASS_GPU_SOLVER_BASE::operator()(){

    //initialize variables
    init_vars();

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = min_iter_calc();
    if (max_loop < 200)
        max_loop = 200;
    
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter(iter&1);

        float max_change = max_of_buffer(dev, r_eff, n_c*n_s);
        if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_GPU_SOLVER_BASE Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter(iter&1);


    //calculate the effective regularization
    calculate_regularization();
    
    //get final output
    add_then_store(dev, data, r_eff, u, n_s*n_c);
        
    //deallocate temporary buffers
    clean_up();
}

POTTS_MEANPASS_GPU_SOLVER_BASE::~POTTS_MEANPASS_GPU_SOLVER_BASE(){
}


POTTS_MEANPASS_GPU_GRADIENT_BASE::POTTS_MEANPASS_GPU_GRADIENT_BASE(
    const cudaStream_t & dev,
    const int batch,
    const int n_s,
    const int n_c,
    const float* u,
    const float* g,
    float* g_d,
    float** full_buffs) :
dev(dev),
b(batch),
n_c(n_c),
n_s(n_s),
grad(g),
logits(u),
g_data(g_d),
d_y(full_buffs[0]),
g_u(full_buffs[1]),
u(full_buffs[2])
{
    //std::cout << n_s << " " << n_c << " " << grad << " " << logits << " " << g_data << " " << d_y << " " << g_u << " " << this->u << std::endl;
}

//perform one iteration of the algorithm
void POTTS_MEANPASS_GPU_GRADIENT_BASE::block_iter(){
    //untangle softmax derivative and add to data term gradient
    untangle_softmax(dev, g_u, u, d_y, n_s, n_c);
    
    // populate data gradient
    inc_mult_buffer(dev, d_y, g_data, n_s*n_c, tau);
    
    //add to regularization gradients and push back
    get_reg_gradients_and_push(tau);
}

void POTTS_MEANPASS_GPU_GRADIENT_BASE::operator()(){

    //initialize variables
    init_vars();
    softmax(dev, logits, 0, u, n_s, n_c);
    clear_buffer(dev, g_u, n_s*n_c);
    copy_buffer(dev, grad, d_y, n_s*n_c);
    
    //get initial gradient for the data and regularization terms
    copy_buffer(dev, grad, g_data, n_s*n_c);
    get_reg_gradients_and_push(1.0f);

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = min_iter_calc();
    if (max_loop < 200)
        max_loop = 200;
    
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        float max_change = max_of_buffer(dev, g_u, n_s*n_c);
        if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_GPU_GRADIENT_BASE Iter " << i << ": " << max_change << std::endl;
        if (max_change < beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();
        
    //deallocate temporary buffers
    clean_up();
}

POTTS_MEANPASS_GPU_GRADIENT_BASE::~POTTS_MEANPASS_GPU_GRADIENT_BASE(){
}
