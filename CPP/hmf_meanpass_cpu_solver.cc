#include "hmf_meanpass_cpu_solver.h"
#include "cpu_kernels.h"
#include <iostream>

HMF_MEANPASS_CPU_SOLVER_BASE::HMF_MEANPASS_CPU_SOLVER_BASE(
        TreeNode** bottom_up_list,
        const int batch,
        const int n_s,
        const int n_c,
        const int n_r,
        const float* data_cost,
        float* u):
    bottom_up_list(bottom_up_list),
    b(batch),
    n_s(n_s),
    n_c(n_c),
    n_r(n_r),
    data(data_cost),
    u(u),
    u_tmp(0),
    r_eff(0)
    {
        //std::cout << n_s << " " << n_c << " " << n_r << std::endl;
    }

float HMF_MEANPASS_CPU_SOLVER_BASE::block_iter(){
    aggregate_bottom_up(u,u_tmp,n_s,n_c,n_r,bottom_up_list);
	//print_buffer(u_tmp, n_s*n_r);
    update_spatial_flow_calc();
	//print_buffer(r_eff, n_s*n_r);
    aggregate_top_down(r_eff, n_s, n_r, bottom_up_list);
	//print_buffer(r_eff, n_s*n_r);
    for(int s = 0, i = 0; s < n_s; s++)
    for(int c = 0; c < n_c; c++, i++)
        r_eff[i] = data[i]+r_eff[n_r*s+c];
    float max_change = softmax_with_convergence(r_eff, u, n_s, n_c, tau);
    return max_change;
}

HMF_MEANPASS_CPU_SOLVER_BASE::~HMF_MEANPASS_CPU_SOLVER_BASE(){
}

//perform one iteration of the algorithm
void HMF_MEANPASS_CPU_SOLVER_BASE::operator()(){
    
    // allocate intermediate variables
    float max_change = 0.0f;
    u_tmp = new float[n_s*n_r];
    r_eff = new float[n_s*n_r];

    //initialize variables
    softmax(data, u, n_s, n_c);
    
    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = 200;
    for(int i = 0; i < max_loop; i++){
        
        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            max_change = block_iter();

        //std::cout << "Iter #" << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();
    
    //perform majority of last iteration
    aggregate_bottom_up(u,u_tmp,n_s,n_c,n_r,bottom_up_list);
    update_spatial_flow_calc();
    aggregate_top_down(r_eff, n_s, n_r, bottom_up_list);

    //get final output
    for(int s = 0; s < n_s; s++)
        for(int c = 0; c < n_c; c++)
            u[n_c*s+c] = data[n_c*s+c]+r_eff[n_r*s+c];
    
    //deallocate temporary buffers
    delete u_tmp; u_tmp = 0;
    delete r_eff; r_eff = 0;
}

HMF_MEANPASS_CPU_GRADIENT_BASE::HMF_MEANPASS_CPU_GRADIENT_BASE(
        TreeNode** bottom_up_list,
        const int batch,
        const int n_s,
        const int n_c,
        const int n_r,
        const float* u,
        const float* g,
        float* g_d ) :
    bottom_up_list(bottom_up_list),
    b(batch),
    n_s(n_s),
    n_c(n_c),
    n_r(n_r),
    g_data(g_d),
    logits(u),
    grad(g),
    dy(0),
    u(0),
    g_u(0)
    {}

HMF_MEANPASS_CPU_GRADIENT_BASE::~HMF_MEANPASS_CPU_GRADIENT_BASE(){
}

void HMF_MEANPASS_CPU_GRADIENT_BASE::operator()(){

    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = 200;
    
    //allocate temporary variables
    u = new float[n_s*n_r];
    dy = new float[n_s*n_r];
    g_u = new float[n_s*n_r];
    float* u_small = new float[n_s*n_c];
    
    //transformat logits into labelling and unroll up through hierarchy
    clear(u, n_s*n_r);
    softmax(logits, u_small, n_s, n_c);
    copy(u_small, u, n_s*n_c);
    unfold_buffer(u, n_s, n_c, n_r);
    aggregate_bottom_up(u, n_s, n_r, bottom_up_list);

    //get initial gradient for the data terms and regularization terms
    copy(grad, g_data, n_s*n_c);
    copy(grad, dy, n_s*n_c);
    unfold_buffer(dy, n_s, n_c, n_r);
    update_spatial_flow_calc(false);
    
    //collapse down to leaves
    aggregate_top_down(g_u, n_s, n_r, bottom_up_list);
    refold_buffer(g_u, n_s, n_c, n_r);
    
    for(int i = 0; i < max_loop; i++){
        for(int iter = 0; iter < min_iter; iter++){
            //untangle softmax
            untangle_softmax(g_u, u_small, dy, n_s, n_c);
            
            // populate data gradient
            inc(dy, g_data, tau, n_s*n_c);

            // unfold gradient to full hierarchy
            unfold_buffer(dy, n_s, n_c, n_r);
            
            // push through regularization and energy equations
            update_spatial_flow_calc(true);
    
            //collapse down to leaves
            aggregate_top_down(g_u, n_s, n_r, bottom_up_list);
            refold_buffer(g_u, n_s, n_c, n_r);
        }
        
        //get max of gu and break if converged
        float gu_max = maxabs(g_u, n_s*n_c);
        if( gu_max < beta )
            break;
    }
    
    delete u; u = 0;
    delete dy; dy = 0;
    delete g_u; g_u = 0;
    delete u_small; u_small = 0;
    
}