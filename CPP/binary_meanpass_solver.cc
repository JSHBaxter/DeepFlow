
#include "binary_meanpass_solver.h"

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
BINARY_MEANPASS_SOLVER<DEV>::BINARY_MEANPASS_SOLVER(
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
    if(DEBUG_ITER) std::cout << "BINARY_MEANPASS_SOLVER Constructor " << dim << " " << n_s << " " << n_c << std::endl;
    spatial_flow = new SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>(dev,n_c,dims,dim,inputs+2);
    MAXFLOW_ALGORITHM<DEV>::construct();
}

template<typename DEV>
void BINARY_MEANPASS_SOLVER<DEV>::allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over){
    if(DEBUG_ITER) std::cout << "BINARY_MEANPASS_SOLVER Allocate" << std::endl;
    r_eff = buffer;
    float* carry_over_tmp[] = {r_eff};
    const float* c_carry_over_tmp[] = {u};
    spatial_flow->allocate_buffers(buffer+n_s*n_c,carry_over_tmp,c_carry_over_tmp);
}

template<typename DEV>
int BINARY_MEANPASS_SOLVER<DEV>::get_buffer_size(){
    return n_s * n_c + spatial_flow->get_buffer_size();
}

template<typename DEV>
BINARY_MEANPASS_SOLVER<DEV>::~BINARY_MEANPASS_SOLVER(){
    delete spatial_flow;
}

//perform one iteration of the algorithm
template<typename DEV>
void BINARY_MEANPASS_SOLVER<DEV>::block_iter(int parity){
    if(DEBUG_ITER_EXTREME) print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,r_eff,n_s*n_c);
    if(DEBUG_ITER_EXTREME) print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,u,n_s*n_c);
	spatial_flow->run();
    if(DEBUG_ITER_EXTREME) print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,r_eff,n_s*n_c);
    if(DEBUG_ITER_EXTREME) print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,u,n_s*n_c);
	sigmoid(MAXFLOW_ALGORITHM<DEV>::dev, data, r_eff, r_eff, n_s*n_c);
    if(DEBUG_ITER_EXTREME) print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,r_eff,n_s*n_c);
    if(DEBUG_ITER_EXTREME) print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,u,n_s*n_c);
    spatial_flow->parity(r_eff, u, n_c, parity);
    if(DEBUG_ITER_EXTREME) print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,r_eff,n_s*n_c);
    if(DEBUG_ITER_EXTREME) print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,u,n_s*n_c);
	change_to_diff(MAXFLOW_ALGORITHM<DEV>::dev, u, r_eff, n_s*n_c, tau);
    if(DEBUG_ITER_EXTREME) std::cout << std::endl;
}

template<typename DEV>
void BINARY_MEANPASS_SOLVER<DEV>::run(){
    if(DEBUG_ITER) std::cout << "BINARY_MEANPASS_SOLVER Init" << std::endl;

	//initialize variables
	if(init_u){
        if(DEBUG_ITER_EXTREME) print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,init_u,n_s*n_c);
        copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, init_u, u, n_s*n_c);
	}else
		sigmoid(MAXFLOW_ALGORITHM<DEV>::dev, data, 0, u, n_s*n_c);
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
        for (int iter = 0; iter < min_iter-1; iter++){
            if(DEBUG_ITER_EXTREME) std::cout << "Sub iter " << iter << std::endl;
            block_iter(0);
            block_iter(1);
		}
		block_iter(0);
		float max_change_1 = max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff, n_c*n_s);
		block_iter(1);
		float max_change_2 = max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, r_eff, n_c*n_s);
		float max_change = (max_change_1 > max_change_2) ? max_change_1 : max_change_2;
		
		if(DEBUG_ITER) std::cout << "BINARY_MEANPASS_SOLVER Iter " << i << ": " << max_change_1 << " " << max_change_2 << std::endl;
        if(max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
	for (int iter = 0; iter < min_iter; iter++){
		block_iter(0);
		block_iter(1);
	}

	//calculate the effective regularization
	spatial_flow->run();
	
	//get final output
	add_then_store(MAXFLOW_ALGORITHM<DEV>::dev, data, r_eff, u, n_s*n_c);
        
    //deallocate temporary buffers
    spatial_flow->deinit();
}




template<typename DEV>
BINARY_MEANPASS_GRADIENT<DEV>::BINARY_MEANPASS_GRADIENT(
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
    //std::cout << n_s << " " << n_c << " " << grad << " " << logits << " " << g_data << " " << d_y << " " << g_u << " " << this->u << std::endl;
    if(DEBUG_ITER) std::cout << "BINARY_MEANPASS_GRADIENT Constructor " << dim << " " << n_s << " " << n_c << std::endl;
    spatial_flow = new SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>(dev, n_c, dims, dim, inputs+2, outputs+1);
    MAXFLOW_ALGORITHM<DEV>::construct();
}

template<typename DEV>
BINARY_MEANPASS_GRADIENT<DEV>::~BINARY_MEANPASS_GRADIENT(){
    delete spatial_flow;
}

template<typename DEV>
void BINARY_MEANPASS_GRADIENT<DEV>::allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over){
    if(DEBUG_ITER) std::cout << "BINARY_MEANPASS_CPU_GRADIENT Allocate" << std::endl;
    u = buffer;
    g_u = u+n_s*n_c;
    d_y = g_u+n_s*n_c;
    float* carry_over_tmp[] = {g_u, d_y};
    const float* c_carry_over_tmp[] = {u};
    spatial_flow->allocate_buffers(d_y+n_s*n_c,carry_over_tmp,c_carry_over_tmp);
}

template<typename DEV>
int BINARY_MEANPASS_GRADIENT<DEV>::get_buffer_size(){
    return 3*n_s*n_c + spatial_flow->get_buffer_size();
}

//perform one iteration of the algorithm
template<typename DEV>
void BINARY_MEANPASS_GRADIENT<DEV>::block_iter(){
	//untangle softmax derivative
	untangle_sigmoid(MAXFLOW_ALGORITHM<DEV>::dev, g_u, u, d_y, n_s*n_c);
	
	//add to data term gradient
	inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, d_y, g_data, n_s*n_c, tau);
	
	//add to regularization gradients
	spatial_flow->run(tau);
}

template<typename DEV>
void BINARY_MEANPASS_GRADIENT<DEV>::run(){
    if(DEBUG_ITER) std::cout << "BINARY_MEANPASS_GRADIENT Init" << std::endl;
    
    //get initial maximum value of teh gradient for convgencence purposes
    const float init_grad_max = max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, grad, n_s*n_c);

	//initialize variables
	spatial_flow->init();
	sigmoid(MAXFLOW_ALGORITHM<DEV>::dev, logits, 0, u, n_s*n_c);
	clear_buffer(MAXFLOW_ALGORITHM<DEV>::dev, g_u, n_s*n_c);
	copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, grad, d_y, n_s*n_c);
	
	//get initial gradient for the data and regularization terms
	copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, grad, g_data, n_s*n_c);
	spatial_flow->run(1.0f);

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
            block_iter();

		float max_change = max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, g_u, n_s*n_c);
		if(DEBUG_ITER) std::cout << "BINARY_MEANPASS_GRADIENT Iter " << i << ": " << max_change << " " << beta << std::endl;
        if (max_change < init_grad_max*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();
        
    if(DEBUG_ITER) std::cout << "BINARY_MEANPASS_GRADIENT DeInit" << std::endl;
    spatial_flow->deinit();
}