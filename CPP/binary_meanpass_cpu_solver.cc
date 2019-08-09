#include "binary_meanpass_cpu_solver.h"
#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "cpu_kernels.h"

BINARY_MEANPASS_CPU_SOLVER_BASE::BINARY_MEANPASS_CPU_SOLVER_BASE(
    const int batch,
    const int n_s,
    const int n_c,
    const float* data_cost,
    float* u ) :
b(batch),
n_c(n_c),
n_s(n_s),
data(data_cost),
r_eff(0),
u(u)
{
    //std::cout << n_s << " " << n_c << std::endl;
}

//perform one iteration of the algorithm
float BINARY_MEANPASS_CPU_SOLVER_BASE::block_iter(bool last){
	float max_change = 0.0f;
	calculate_regularization();
	inc(data, r_eff, n_s*n_c);
	if(last)
		max_change = sigmoid_with_convergence(r_eff, u, n_s*n_c, tau);
	else
		sigmoid_update(r_eff, u, n_s*n_c, tau);
	return max_change;
}

void BINARY_MEANPASS_CPU_SOLVER_BASE::operator()(){
        
	// allocate intermediate variables
	float max_change = 0.0f;
	r_eff = new float[n_s*n_c];

	//initialize variables
	init_vars();
	sigmoid(data, u, n_s*n_c);

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = 200;
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            max_change = block_iter(iter == min_iter-1);

		//std::cout << "Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter(false);


	//calculate the effective regularization
	calculate_regularization();
	
	//get final output
	for(int i = 0; i < n_s*n_c; i++)
		u[i] = data[i]+r_eff[i];
        
    //deallocate temporary buffers
    delete r_eff; r_eff = 0;
    clean_up();
}

BINARY_MEANPASS_CPU_SOLVER_BASE::~BINARY_MEANPASS_CPU_SOLVER_BASE(){
}


BINARY_MEANPASS_CPU_GRADIENT_BASE::BINARY_MEANPASS_CPU_GRADIENT_BASE(
    const int batch,
    const int n_s,
    const int n_c,
	const float* u,
	const float* g,
	float* g_d ) :
b(batch),
n_c(n_c),
n_s(n_s),
grad(g),
logits(u),
g_data(g_d),
d_y(0),
g_u(0),
u(0)
{
    //std::cout << n_s << " " << n_c << std::endl;
}

//perform one iteration of the algorithm
void BINARY_MEANPASS_CPU_GRADIENT_BASE::block_iter(){
	//untangle softmax derivative
	untangle_softmax(g_u, u, d_y, n_s, n_c);
	
	// populate data gradient
	inc(d_y, g_data, tau, n_s*n_c);
	
	//add to regularization gradients and push back
	get_reg_gradients_and_push(tau);
}

void BINARY_MEANPASS_CPU_GRADIENT_BASE::operator()(){
        
	// allocate intermediate variables
	u = new float[n_s*n_c];
	d_y = new float[n_s*n_c];
	g_u = new float[n_s*n_c];

	//initialize variables
	init_vars();
	sigmoid(logits, u, n_s*n_c);
	clear(d_y, g_u, g_data, n_s*n_c);
	
	//get initial gradient for the data and regularization terms
	copy(grad,g_data,n_s*n_c);
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

		float max_change = maxabs(g_u,n_s*n_c);
		//std::cout << "Iter " << i << ": " << max_change << std::endl;
        if (max_change < beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();
        
    //deallocate temporary buffers
    delete u; u = 0;
    delete d_y; d_y = 0;
    delete g_u; g_u = 0;
    clean_up();
}

BINARY_MEANPASS_CPU_GRADIENT_BASE::~BINARY_MEANPASS_CPU_GRADIENT_BASE(){
}