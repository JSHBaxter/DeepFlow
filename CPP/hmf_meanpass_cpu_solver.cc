#include "hmf_meanpass_cpu_solver.h"
#include "cpu_kernels.h"
#include <iostream>

HMF_MEANPASS_CPU_SOLVER_BASE::HMF_MEANPASS_CPU_SOLVER_BASE(
        const bool channels_first,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_s,
        const int n_c,
        const int n_r,
        const float* data_cost,
        const float* const init_u,
        float* u):
    channels_first(channels_first),
    bottom_up_list(bottom_up_list),
    b(batch),
    n_s(n_s),
    n_c(n_c),
    n_r(n_r),
    data(data_cost),
    u(u),
    data_b(channels_first ? data_cost : transpose(data_cost, new float[n_s*n_c], n_s, n_c) ),
    u_tmp(new float[n_s*n_r]),
    r_eff(new float[n_s*n_r])
{
	//std::cout << n_s << " " << n_c << " " << n_r << std::endl;
	//initialize variables
	if(init_u)
		copy(init_u, u, n_s*n_c);
	else{
        if(channels_first)
            softmax_channels_first(data, u, n_s, n_c);
        else
            softmax(data, u, n_s, n_c);
    }
}

HMF_MEANPASS_CPU_SOLVER_BASE::~HMF_MEANPASS_CPU_SOLVER_BASE(){
    delete[] u_tmp;
    delete[] r_eff;
    if(!channels_first) delete[] data_b;
}

float HMF_MEANPASS_CPU_SOLVER_BASE::block_iter(const int parity, bool last){
    float max_change = 0.0f;
    
    aggregate_bottom_up_channels_first(u,u_tmp,n_s,n_c,n_r,bottom_up_list);
	//print_buffer(u_tmp, n_s*n_r);
    update_spatial_flow_calc();
	//print_buffer(r_eff, n_s*n_r);
    aggregate_top_down_channels_first(r_eff, n_s, n_r, bottom_up_list);
	//print_buffer(r_eff, n_s*n_r);
    inc(data_b, r_eff+n_s*(n_r-n_c), n_s*n_c);
    softmax_channels_first(r_eff+n_s*(n_r-n_c),r_eff,n_s,n_c);
    parity_merge_buffer(r_eff,u,parity);
	if(last)
		max_change = update_with_convergence(u, r_eff, n_s*n_c, tau);
	else
		update(u, r_eff, n_s*n_c, tau);
    return max_change;
}

//perform one iteration of the algorithm
void HMF_MEANPASS_CPU_SOLVER_BASE::operator()(){
    
    // allocate intermediate variables
    u_tmp = new float[n_s*n_r];
    r_eff = new float[n_s*n_r];
    init_reg_info();

    
    // transpose initial labelling
    if(!channels_first){
        for(int s = 0; s < n_s; s++)
            for(int c = 0; c < n_c; c++)
                u_tmp[c*n_s+s] = u[s*n_c+c];
        copy(u_tmp,u,n_s*n_c);
    }
    
    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = min_iter_calc();
    if (max_loop < 200)
        max_loop = 200;
    
    for(int i = 0; i < max_loop; i++){
        
        //run the solver a set block of iterations
        float max_change = 0.0f;
        for (int iter = 0; iter < min_iter; iter++){
            float max_change_1 = block_iter(0,iter==min_iter-1);
            float max_change_2 = block_iter(1,iter==min_iter-1);
            max_change = (max_change_1 > max_change_2) ? max_change_1 : max_change_2;
        }

        //std::cout << "HMF_MEANPASS_CPU_SOLVER_BASE Iter #" << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++){
        block_iter(0,false);
        block_iter(1,false);
    }
    
    //perform majority of last iteration
    aggregate_bottom_up_channels_first(u,u_tmp,n_s,n_c,n_r,bottom_up_list);
    update_spatial_flow_calc();
    aggregate_top_down_channels_first(r_eff, n_s, n_r,bottom_up_list);

    if(!channels_first){
        //get final output (save in data_b as tmp (needed for un-transposition))
        inc(data_b, r_eff+n_s*(n_r-n_c), n_s*n_c);

        // un-transpose final labelling
        for(int s = 0; s < n_s; s++)
            for(int c = 0; c < n_c; c++)
                u[s*n_c+c] = (r_eff+n_s*(n_r-n_c))[c*n_s+s];
    }else{
        inc(data_b, r_eff+n_s*(n_r-n_c), n_s*n_c);
        copy(r_eff+n_s*(n_r-n_c),u,n_s*n_c);
    }
    
    //deallocate temporary buffers
    clean_up();
}

HMF_MEANPASS_CPU_GRADIENT_BASE::HMF_MEANPASS_CPU_GRADIENT_BASE(
        const bool channels_first,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_s,
        const int n_c,
        const int n_r,
        const float* u,
        const float* g,
        float* g_d ) :
    channels_first(channels_first),
    bottom_up_list(bottom_up_list),
    b(batch),
    n_s(n_s),
    n_c(n_c),
    n_r(n_r),
    g_data(g_d),
    logits(u),
    grad(g),
    dy(new float[n_s*n_r]),
    g_u(new float[n_s*n_r]),
    u(new float[n_s*n_r])
    {}

HMF_MEANPASS_CPU_GRADIENT_BASE::~HMF_MEANPASS_CPU_GRADIENT_BASE(){
    delete[] dy;
    delete[] g_u;
    delete[] u;
}

void HMF_MEANPASS_CPU_GRADIENT_BASE::block_iter(){
    
    //untangle softmax
    untangle_softmax_channels_first(g_u+n_s*(n_r-n_c), u+n_s*(n_r-n_c), dy+n_s*(n_r-n_c), n_s, n_c);

    // populate data gradient
    inc(dy+n_s*(n_r-n_c), g_data, tau, n_s*n_c);
    
    //populate gradients up the hierarchy
    clear(dy,n_s*(n_r-n_c));
    aggregate_bottom_up_channels_first(dy, n_s, n_r, bottom_up_list);

    // push through regularization and energy equations
    get_reg_gradients_and_push(tau);

    //collapse down to leaves
    aggregate_top_down_channels_first(g_u, n_s, n_r, bottom_up_list);
    
}

void HMF_MEANPASS_CPU_GRADIENT_BASE::operator()(){
    int u_offset = n_s*(n_r-n_c);

    //get initial maximum value of the gradient for convgencence purposes
    const float init_grad_max = maxabs(grad,n_s*n_c);
        
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = min_iter_calc();
    if (max_loop < 200)
        max_loop = 200;
    
    //init temporary variables
    init_reg_info();
    
    //transform logits into labelling (and change to channels-first) and unroll up through hierarchy (use g_u_l as temp)
    clear(u, n_s*(n_r-n_c));
    if( channels_first ){
        softmax_channels_first(logits, u+u_offset, n_s, n_c);
    }else{
        softmax(logits, u+u_offset, n_s, n_c);
        for(int s = 0; s < n_s; s++)
            for(int c = 0; c < n_c; c++)
                (g_u+u_offset)[c*n_s+s] = (u+u_offset)[s*n_c+c];
        copy(g_u+u_offset,u+u_offset,n_c*n_s);
    }
    aggregate_bottom_up_channels_first(u, n_s, n_r, bottom_up_list);

    //transpose gradient and store initially in g_data in order to initial gradient for the data terms
    if( channels_first ){
        copy(grad, g_data, n_s*n_c);
    }else{
        for(int s = 0; s < n_s; s++)
            for(int c = 0; c < n_c; c++)
                g_data[c*n_s+s] = grad[s*n_c+c];
    }
    copy(g_data, dy+u_offset, n_s*n_c);
    
    //populate gradients up the hierarchy
    clear(dy,n_s*(n_r-n_c));
    aggregate_bottom_up_channels_first(dy, n_s, n_r, bottom_up_list);
    
    //get initial gradient for the regularization terms
    clear(g_u,n_s*n_r);
    get_reg_gradients_and_push(1.0);
    
    //collapse down to leaves
    aggregate_top_down_channels_first(g_u, n_s, n_r, bottom_up_list);
        
    for(int i = 0; i < max_loop; i++){
        for(int iter = 0; iter < min_iter; iter++)
            block_iter();
        
        //get max of gu and break if converged
        float gu_max = maxabs(g_u+u_offset, n_s*n_c);
        //std::cout << "HMF_MEANPASS_CPU_GRADIENT_BASE Iter #" << i << " " << gu_max << std::endl;
        if( gu_max <= beta*init_grad_max )
            break;
    }
    
    //untranspose data terms using g_u as temp storage
    if( !channels_first ){
        for(int s = 0; s < n_s; s++)
            for(int c = 0; c < n_c; c++)
                g_u[s*n_c+c] = g_data[c*n_s+s];
        copy(g_u,g_data,n_s*n_c);
    }
    
    clean_up();
    
}
