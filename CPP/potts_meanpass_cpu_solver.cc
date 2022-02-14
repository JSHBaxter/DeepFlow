#include "potts_meanpass_cpu_solver.h"
#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "cpu_kernels.h"

POTTS_MEANPASS_CPU_SOLVER_BASE::POTTS_MEANPASS_CPU_SOLVER_BASE(
    const bool channels_first,
    const int batch,
    const int n_s,
    const int n_c,
    const float* data_cost,
    const float* init_u,
    float* u ) :
channels_first(channels_first),
b(batch),
n_c(n_c),
n_s(n_s),
data(data_cost),
r_eff(new float[n_s*n_c]),
u(u)
{
    if(init_u)
        copy(init_u, u, n_s*n_c);
    else
        softmax(data, u, n_s, n_c);
    //std::cout << n_s << " " << n_c  << " " << data << " " << r_eff << " " << u  << std::endl;
}

//perform one iteration of the algorithm
float POTTS_MEANPASS_CPU_SOLVER_BASE::block_iter(const int parity, bool last){
	float max_change = 0.0f;
	calculate_regularization();
	inc(data, r_eff, n_s*n_c);
    if(channels_first)
        softmax_channels_first(r_eff,r_eff,n_s,n_c);
    else
        softmax(r_eff,r_eff,n_s,n_c);
    parity_merge_buffer(r_eff,u,parity);
	if(last)
		max_change = update_with_convergence(u, r_eff, n_s*n_c, tau);
		//max_change = softmax_with_convergence(r_eff, u, n_s, n_c, tau);
	else
		update(u, r_eff, n_s*n_c, tau);
		//softmax_update(r_eff, u, n_s, n_c, tau);
	return max_change;
}

void POTTS_MEANPASS_CPU_SOLVER_BASE::operator()(){
        
	// allocate intermediate variables
	float max_change = 0.0f;

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
            max_change = block_iter(iter&1, iter == min_iter-1);

		if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_CPU_SOLVER_BASE Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter(iter&1, false);


	//calculate the effective regularization
	calculate_regularization();
	
	//get final output
	for(int i = 0; i < n_s*n_c; i++)
		u[i] = data[i]+r_eff[i];
        
    //deallocate temporary buffers
    clean_up();
}

POTTS_MEANPASS_CPU_SOLVER_BASE::~POTTS_MEANPASS_CPU_SOLVER_BASE(){
    delete [] r_eff;
}


POTTS_MEANPASS_CPU_GRADIENT_BASE::POTTS_MEANPASS_CPU_GRADIENT_BASE(
    const bool channels_first,
    const int batch,
    const int n_s,
    const int n_c,
	const float* u,
	const float* g,
	float* g_d ) :
channels_first(channels_first),
b(batch),
n_c(n_c),
n_s(n_s),
grad(g),
logits(u),
g_data(g_d),
d_y(new float[n_s*n_c]),
g_u(new float[n_s*n_c]),
u(new float[n_s*n_c])
{
    //std::cout << n_s << " " << n_c  << " " << grad << " " << logits << " " << g_data << " " << d_y << " " << g_u  << " " << this->u << std::endl;
}

//perform one iteration of the algorithm
void POTTS_MEANPASS_CPU_GRADIENT_BASE::block_iter(){
	//untangle softmax derivative
    if(channels_first)
	    untangle_softmax_channels_first(g_u, u, d_y, n_s, n_c);
    else
	    untangle_softmax(g_u, u, d_y, n_s, n_c);
	
	// populate data gradient
	inc(d_y, g_data, tau, n_s*n_c);
	
	//add to regularization gradients and push back
	get_reg_gradients_and_push(tau);
}

void POTTS_MEANPASS_CPU_GRADIENT_BASE::operator()(){

	//initialize variables
	init_vars();
    if(channels_first)
	    softmax_channels_first(logits, u, n_s, n_c);
    else
	    softmax(logits, u, n_s, n_c);
	clear(g_u, g_data, n_s*n_c);
	copy(grad,d_y, n_s*n_c);
	
	//get initial gradient for the data and regularization terms
	copy(grad,g_data,n_s*n_c);
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

		float max_change = maxabs(g_u,n_s*n_c);
		if(DEBUG_ITER) std::cout << "POTTS_MEANPASS_CPU_GRADIENT_BASE Iter " << i << ": " << max_change << std::endl;
        if (max_change < beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();
        
    //deallocate temporary buffers
    clean_up();
}

POTTS_MEANPASS_CPU_GRADIENT_BASE::~POTTS_MEANPASS_CPU_GRADIENT_BASE(){
    delete [] u;
    delete [] d_y;
    delete [] g_u;
}
