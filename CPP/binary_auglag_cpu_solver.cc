#include "binary_auglag_cpu_solver.h"
#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "cpu_kernels.h"

BINARY_AUGLAG_CPU_SOLVER_BASE::BINARY_AUGLAG_CPU_SOLVER_BASE(
    const int batch,
    const int n_s,
    const int n_c,
    const float* data_cost,
    float* u ) :
b(batch),
n_c(n_c),
n_s(n_s),
data(data_cost),
ps(0),
pt(0),
div(0),
g(0),
u(u)
{
    //std::cout << n_s << " " << n_c << std::endl;
}

//perform one iteration of the algorithm
void BINARY_AUGLAG_CPU_SOLVER_BASE::block_iter(){
    
    //calculate the capacity and then update flows
    compute_capacity_binary(g, u, ps, pt, div, n_s*n_c, tau, icc);
    update_spatial_flow_calc();
                 
	//update source flows, sink flows, and multipliers
	compute_source_sink_multipliers_binary( g, u, ps, pt, div, data, cc, icc, n_s*n_c);
}

void BINARY_AUGLAG_CPU_SOLVER_BASE::operator()(){

    //store intermediate information
    ps = new float[n_s*n_c];
    pt = new float[n_s*n_c];
    div = new float[n_s*n_c];
    g = new float[n_s*n_c];

    //initialize variables
	sigmoid(data, u, n_s*n_c);
    //clear(u, n_s*n_c);
	clear(g, div, n_c*n_s);
    clear_spatial_flows();
    clear(pt, n_c*n_s);
    clear(ps, n_c*n_s);
	//init_flows(data, ps, pt, n_s, n_c);

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = 200;
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        float max_change = maxabs(g,n_s*n_c);
		//std::cout << "Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();

    //log output and transpose output back into proper buffer
    log_buffer(u, n_s*n_c);
        
    //deallocate temporary buffers
    delete pt; pt = 0;
    delete ps; ps = 0;
    delete g; g = 0;
    delete div; div = 0;
    clean_up();
}

BINARY_AUGLAG_CPU_SOLVER_BASE::~BINARY_AUGLAG_CPU_SOLVER_BASE(){
}