
#include "dagmf_meanpass_solver.h"
#include "hmf_trees.h"
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
DAGMF_MEANPASS_SOLVER<DEV>::DAGMF_MEANPASS_SOLVER(
    const DEV & dev,
    DAGNode** bottom_up_list,
    const int dim,
    const int* dims,
    const int n_c,
    const int n_r,
    const float* const* const inputs,
    float* const u) :
MAXFLOW_ALGORITHM<DEV>(dev),
bottom_up_list(bottom_up_list),
n_c(n_c),
n_r(n_r),
n_s(product(dim,dims)),
init_u(inputs[0]),
data(inputs[1]),
u(u),
r_eff(0),
u_full(0)
{
    if(DEBUG_ITER) std::cout << "DAGMF_MEANPASS_SOLVER Constructor " << dim << " " << n_s << " " << n_c << " " << n_r << " " << std::endl;    
    spatial_flow =  new SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>(MAXFLOW_ALGORITHM<DEV>::dev,n_r,dims,dim,inputs+2);
    MAXFLOW_ALGORITHM<DEV>::construct();
}

template<typename DEV>
void DAGMF_MEANPASS_SOLVER<DEV>::allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over){
    if(DEBUG_ITER) std::cout << "DAGMF_MEANPASS_SOLVER Allocate " << std::endl;
    r_eff = buffer;
    u_full = r_eff+ n_s*n_r;
    float* carry_over_tmp[] = {r_eff};
    const float* c_carry_over_tmp[] = {u_full};
    spatial_flow->allocate_buffers(u_full+n_s*n_r,carry_over_tmp,c_carry_over_tmp);
}

template<typename DEV>
int DAGMF_MEANPASS_SOLVER<DEV>::get_buffer_size(){
    return 2 * n_s * n_r + spatial_flow->get_buffer_size();
}

template<typename DEV>
DAGMF_MEANPASS_SOLVER<DEV>::~DAGMF_MEANPASS_SOLVER(){
    delete spatial_flow;
}

template<typename DEV>
void DAGMF_MEANPASS_SOLVER<DEV>::block_iter(const int parity){
    
    if(DEBUG_PRINT){ std::cout << "u_full: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u_full, n_s*n_r);}
    if(DEBUG_PRINT){ std::cout << "r_eff: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff, n_s*n_r);}
    
    //calculate the aggregate probabilities (stored in u_full)
    copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u, u_full+n_s*(n_r-n_c), n_s*n_c);
    clear_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u_full, n_s*(n_r-n_c));
    for (int l = n_c; l < n_r; l++) {
        const DAGNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u_full+n->children[c]->r*n_s, u_full+n->r*n_s, n_s, n->child_weight[c]);
    }
    if(DEBUG_PRINT){ std::cout << "u_full: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u_full, n_s*n_r);}
    //print_buffer(dev, r_eff, n_s*n_r);
    //aggregate_bottom_up(dev, u_ind, u_full, u, n_s, n_c, n_r);

    //calculate the effective regularization (overwrites own r_eff)
    spatial_flow->run();
    if(DEBUG_PRINT){ std::cout << "r_eff: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff, n_s*n_r);}

    //calculate the aggregate effective regularization (overwrites own r_eff)
    for (int l = n_r-1; l >= 0; l--) {
        const DAGNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff+n->r*n_s, r_eff+n->children[c]->r*n_s, n_s, n->child_weight[c]);
    }
    if(DEBUG_PRINT){ std::cout << "r_eff: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff, n_s*n_r);}

    // get new probability estimates, and normalize
    // (copy into u_full as tmp space (prevent race condition) then store answer in r_eff)
    copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff+n_s*(n_r-n_c), u_full, n_s*n_c);
    softmax(MAXFLOW_ALGORITHM<DEV>::dev, data, u_full, r_eff, n_s, n_c);
    if(DEBUG_PRINT){ std::cout << "r_eff: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff, n_s*n_r);}

    //update labels
    spatial_flow->parity(r_eff,u,n_c,parity);
    change_to_diff(MAXFLOW_ALGORITHM<DEV>::dev, u, r_eff, n_s*n_c, tau);
    if(DEBUG_PRINT){ std::cout << "u: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u, n_s*n_c);}
}

template<typename DEV>
void DAGMF_MEANPASS_SOLVER<DEV>::run(){

    //initialise
    spatial_flow->init();
    if(init_u)
        copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, init_u, u, n_s*n_c);
    else
        softmax(MAXFLOW_ALGORITHM<DEV>::dev, data, NULL, u, n_s, n_c);
    
    // iterate in blocks
    int min_iter = spatial_flow->get_min_iter() + n_r - n_c;
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = spatial_flow->get_max_iter();
    if (max_loop < 10)
        max_loop = 10;
    
    for(int i = 0; i < max_loop; i++){    
        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter-1; iter++){
            block_iter(0);
            block_iter(1);
        }

        //Determine if converged
        block_iter(0);
        float max_change_1 = max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff, n_s*n_c);
        block_iter(1);
        float max_change_2 = max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff, n_s*n_c);
        float max_change = (max_change_1 > max_change_2) ? max_change_1 : max_change_2;
        if(DEBUG_ITER) std::cout << "DAGMF_MEANPASS_SOLVER Iter #" << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }
    
    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++){
        block_iter(0);
        block_iter(1);
    }

    //calculate the aggregate probabilities
    copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u, u_full+n_s*(n_r-n_c), n_s*n_c);
    clear_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u_full, n_s*(n_r-n_c));
    for (int l = n_c; l < n_r; l++) {
        const DAGNode* n = bottom_up_list[l];
        if(DEBUG_PRINT){ std::cout << "\t node: " << n->r << "\tnum_children:\t" << n->c << std::endl; }
        for(int c = 0; c < n->c; c++)
            inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u_full+n->children[c]->r*n_s, u_full+n->r*n_s, n_s, n->child_weight[c]);
    }

    //calculate the effective regularization
    if(DEBUG_PRINT){ std::cout << "u_full: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u_full, n_s*n_r);}
    spatial_flow->run();

    //calculate the aggregate effective regularization
    if(DEBUG_PRINT){ std::cout << "r_eff: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff, n_s*n_r);}
    for (int l = n_r-1; l >= 0; l--) {
        const DAGNode* n = bottom_up_list[l];
        if(DEBUG_PRINT){ std::cout << "\t node: " << n->r << "\tnum_children:\t" << n->c << std::endl; }
        for(int c = 0; c < n->c; c++)
            inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff+n->r*n_s, r_eff+n->children[c]->r*n_s, n_s, n->child_weight[c]);
    }
    if(DEBUG_PRINT){ std::cout << "r_eff: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff, n_s*n_r);}

    
    //get final output
    if(DEBUG_PRINT){ std::cout << "u: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u, n_s*n_c);}
    add_then_store(MAXFLOW_ALGORITHM<DEV>::dev, data, r_eff+n_s*(n_r-n_c), u, n_c*n_s);
    if(DEBUG_PRINT){ std::cout << "u: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u, n_s*n_c);}
    
    //clean up
    spatial_flow->deinit();
    
}

template<typename DEV>
DAGMF_MEANPASS_GRADIENT<DEV>::DAGMF_MEANPASS_GRADIENT(
    const DEV & dev,
    DAGNode** bottom_up_list,
    const int dim,
    const int* dims,
    const int n_c,
    const int n_r,
    const float *const *const inputs,
    float *const *const outputs):
MAXFLOW_ALGORITHM<DEV>(dev),
bottom_up_list(bottom_up_list),
n_s(product(dim,dims)),
n_c(n_c),
n_r(n_r),
g_data(outputs[0]),
grad(inputs[0]),
logits(inputs[1]),
u(0),
dy(0),
du(0),
tmp(0)
{
    if(DEBUG_ITER) std::cout << "DAGMF_MEANPASS_GRADIENT Constructor " << dim << " " << n_s << " " << n_c << " " << n_r << std::endl;
    spatial_flow = new SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>(MAXFLOW_ALGORITHM<DEV>::dev, n_r, dims, dim, inputs+2, outputs+1);
    MAXFLOW_ALGORITHM<DEV>::construct();
}

template<typename DEV>
void DAGMF_MEANPASS_GRADIENT<DEV>::allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over){
    if(DEBUG_ITER) std::cout << "DAGMF_MEANPASS_GRADIENT Allocate " << std::endl;
    u = buffer;
    dy = u + n_s*n_r;
    du = dy + n_s*n_r;
    tmp = du + n_s*n_r;
    float* carry_over_tmp[] = {du, dy};
    const float* c_carry_over_tmp[] = {u};
    spatial_flow->allocate_buffers(tmp+n_s*n_r,carry_over_tmp,c_carry_over_tmp);    
}

template<typename DEV>
int DAGMF_MEANPASS_GRADIENT<DEV>::get_buffer_size(){
    return 4 * n_s * n_r + spatial_flow->get_buffer_size();
}

template<typename DEV>
DAGMF_MEANPASS_GRADIENT<DEV>::~DAGMF_MEANPASS_GRADIENT(){
    delete spatial_flow;
}

template<typename DEV>
void DAGMF_MEANPASS_GRADIENT<DEV>::block_iter(){
    int c_offset = n_s*(n_r-n_c);
    if(DEBUG_PRINT){ std::cout << "\ndu: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, du, n_s*n_r);}

    //process gradients 
    untangle_softmax(MAXFLOW_ALGORITHM<DEV>::dev, du+c_offset, u+c_offset, dy+c_offset, n_s, n_c);
    if(DEBUG_PRINT){ std::cout << "dy: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, dy, n_s*n_r);}

    //add into data term gradient
    inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, dy+c_offset, g_data, n_s*n_c, tau);
    if(DEBUG_PRINT){ std::cout << "g_data: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, g_data, n_s*n_c);}

        //expand gradients upwards
    clear_buffer(MAXFLOW_ALGORITHM<DEV>::dev, dy, n_s*(n_r-n_c));
    for (int l = n_c; l < n_r; l++) {
        const DAGNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
        inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, dy+n->children[c]->r*n_s, dy+n->r*n_s, n_s, n->child_weight[c]);
    }
    
    //get gradients for the regularization terms
    spatial_flow->run(tau);

    //collapse back down to leaves
    if(DEBUG_PRINT){ std::cout << "du: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, du, n_s*n_r);}
    for (int l = n_r-1; l >= n_c; l--) {
        const DAGNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, du+n->r*n_s, du+n->children[c]->r*n_s, n_s, n->child_weight[c]);
    }
    if(DEBUG_PRINT){ std::cout << "du: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, du, n_s*n_r);}
}

template<typename DEV>
void DAGMF_MEANPASS_GRADIENT<DEV>::run(){
    if(DEBUG_ITER) std::cout << "DAGMF_MEANPASS_GRADIENT Init" << std::endl;
    int u_tmp_offset = n_s*(n_r-n_c);
    
    spatial_flow->init();
    
    //get initial maximum value of teh gradient for convgencence purposes
    const float init_grad_max = max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, grad, n_s*n_c);
    
    //calculate the aggregate probabilities
    clear_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u, n_s*(n_r-n_c));
    softmax(MAXFLOW_ALGORITHM<DEV>::dev, logits, NULL, u+u_tmp_offset, n_s, n_c);
    for (int l = n_c; l < n_r; l++) {
        const DAGNode* n = bottom_up_list[l];
        if(DEBUG_PRINT){ std::cout << "\t node: " << n->r << "\tnum_children:\t" << n->c << std::endl; }
        for(int c = 0; c < n->c; c++)
            inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u+n->children[c]->r*n_s, u+n->r*n_s, n_s, n->child_weight[c]);
    }

    // populate data gradient
    copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, grad, g_data, n_s*n_c);
    
    //calculate aggregate gradient
    copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, grad, dy+n_s*(n_r-n_c), n_s*n_c);
    clear_buffer(MAXFLOW_ALGORITHM<DEV>::dev, dy, n_s*(n_r-n_c));
    for (int l = n_c; l < n_r; l++) {
        const DAGNode* n = bottom_up_list[l];
        if(DEBUG_PRINT){ std::cout << "\t node: " << n->r << "\tnum_children:\t" << n->c << std::endl; }
        for(int c = 0; c < n->c; c++)
            inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, dy+n->children[c]->r*n_s, dy+n->r*n_s, n_s, n->child_weight[c]);
    }
    
    //and calculate gradients for the rest
    clear_buffer(MAXFLOW_ALGORITHM<DEV>::dev,du,n_s*n_r);
    spatial_flow->run(1.0f);

    //collapse back down to leaves
    for (int l = n_r-1; l >= n_c; l--) {
        const DAGNode* n = bottom_up_list[l];
        if(DEBUG_PRINT){ std::cout << "\t node: " << n->r << "\tnum_children:\t" << n->c << std::endl; }
        for(int c = 0; c < n->c; c++)
            inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, du+n->r*n_s, du+n->children[c]->r*n_s, n_s, n->child_weight[c]);
    }
    
    int min_iter = spatial_flow->get_min_iter() + n_r - n_c;
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = spatial_flow->get_max_iter();
    if (max_loop < 10)
        max_loop = 10;
    
    for(int i = 0; i < max_loop; i++){
        //push gradients back a number of iterations (first iteration has tau=1, the rest a smaller tau)
        for(int iter = 0; iter < min_iter; iter++)
            block_iter();

            float max_change = max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, du+u_tmp_offset, n_c*n_s);
            if(DEBUG_ITER) std::cout << "DAGMF_MEANPASS_GRADIENT Iter " << i << " " << max_change << std::endl;
            if(max_change < beta*init_grad_max)
                break;
    }
    
    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();
        
    //deallocate temporary buffers
    spatial_flow->deinit();
}
