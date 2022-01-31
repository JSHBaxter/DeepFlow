#include "binary_meanpass_gpu_solver.h"
#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "gpu_kernels.h"

BINARY_MEANPASS_GPU_SOLVER_BASE::BINARY_MEANPASS_GPU_SOLVER_BASE(
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
    //std::cout << n_s << " " << n_c << std::endl;
	if(init_u)
		copy_buffer(dev, init_u, u, n_s*n_c);
	else
		sigmoid(dev, data, 0, u, n_s*n_c);
}

//perform one iteration of the algorithm
void BINARY_MEANPASS_GPU_SOLVER_BASE::block_iter(int parity){
	calculate_regularization();
	sigmoid(dev, data, r_eff, r_eff, n_s*n_c);
    parity_merge_buffer(r_eff, u, parity);
	change_to_diff(dev, u, r_eff, n_s*n_c, tau);
}

void BINARY_MEANPASS_GPU_SOLVER_BASE::operator()(){

	//initialize variables
	init_vars();

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = 200;
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter-1; iter++){
            block_iter(0);
            block_iter(1);
		}
		block_iter(0);
		float max_change_1 = max_of_buffer(dev, r_eff, n_c*n_s);
		block_iter(1);
		float max_change_2 = max_of_buffer(dev, r_eff, n_c*n_s);
		float max_change = (max_change_1 > max_change_2) ? max_change_1 : max_change_2;
		
		//std::cout << "BINARY_MEANPASS_GPU_SOLVER_BASE Iter " << i << ": " << max_change_1 << " " << max_change_2 << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
	for (int iter = 0; iter < min_iter; iter++){
		block_iter(0);
		block_iter(1);
	}

	//calculate the effective regularization
	calculate_regularization();
	
	//get final output
	add_then_store(dev, data, r_eff, u, n_s*n_c);
        
    //deallocate temporary buffers
    clean_up();
}

BINARY_MEANPASS_GPU_SOLVER_BASE::~BINARY_MEANPASS_GPU_SOLVER_BASE(){
}


BINARY_MEANPASS_GPU_GRADIENT_BASE::BINARY_MEANPASS_GPU_GRADIENT_BASE(
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
    //std::cout << n_s << " " << n_c << std::endl;
}

//perform one iteration of the algorithm
void BINARY_MEANPASS_GPU_GRADIENT_BASE::block_iter(){
	//untangle softmax derivative
	untangle_sigmoid(dev, g_u, u, d_y, n_s*n_c);
	
	//add to data term gradient
	inc_mult_buffer(dev, d_y, g_data, n_s*n_c, tau);
	
	//add to regularization gradients
	get_reg_gradients_and_push(tau);
}

void BINARY_MEANPASS_GPU_GRADIENT_BASE::operator()(){

	//initialize variables
	init_vars();
	sigmoid(dev, logits, 0, u, n_s*n_c);
	clear_buffer(dev, g_u, n_s*n_c);
	copy_buffer(dev, grad, d_y, n_s*n_c);
	
	//get initial gradient for the data and regularization terms
	copy_buffer(dev, grad, g_data, n_s*n_c);
	get_reg_gradients_and_push(1.0f);

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = 200;
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

		float max_change = max_of_buffer(dev, g_u, n_s*n_c);
		//std::cout << "BINARY_MEANPASS_GPU_GRADIENT_BASE Iter " << i << ": " << max_change << " " << beta << std::endl;
        if (max_change < beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();
        
    //deallocate temporary buffers
    clean_up();
}

BINARY_MEANPASS_GPU_GRADIENT_BASE::~BINARY_MEANPASS_GPU_GRADIENT_BASE(){
}
