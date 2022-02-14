#include "binary_auglag_cpu_solver.h"
#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "cpu_kernels.h"

BINARY_AUGLAG_CPU_SOLVER_BASE::BINARY_AUGLAG_CPU_SOLVER_BASE(
    const bool channels_first,
    const int batch,
    const int n_s,
    const int n_c,
    const float * const data_cost,
    float* u ) :
channels_first(channels_first),
b(batch),
n_c(n_c),
n_s(n_s),
data(data_cost),
ps(new float[4*n_s*n_c]),
pt(ps+n_s*n_c),
div(pt+n_s*n_c),
g(div+n_s*n_c),
u(u)
{
    //std::cout << "BINARY_AUGLAG_CPU_SOLVER_BASE\t" << n_s << " " << n_c << " " << data << " " << ps << " " << pt << " " << div << " " << g << " " << u << std::endl;
}

//perform one iteration of the algorithm
void BINARY_AUGLAG_CPU_SOLVER_BASE::block_iter(){
    
    //calculate the capacity and then update flows
    compute_capacity_binary(g, u, ps, pt, div, n_s*n_c, tau, icc);
    print_buffer(g,n_s*n_c);
    update_spatial_flow_calc();
                 
	//update source flows, sink flows, and multipliers
	compute_source_sink_multipliers_binary( g, u, ps, pt, div, data, cc, icc, n_s*n_c);
    print_buffer(ps,n_s*n_c);
    print_buffer(pt,n_s*n_c);
    print_buffer(g,n_s*n_c);
    print_buffer(u,n_s*n_c);
}

void BINARY_AUGLAG_CPU_SOLVER_BASE::operator()(){

    //initialize variables
	sigmoid(data, u, n_s*n_c);
    //clear(u, n_s*n_c);
	clear(g, div, n_c*n_s);
    clear_spatial_flows();
    clear(pt, n_c*n_s);
    clear(ps, n_c*n_s);
	init_flows_binary(data, ps, pt, n_s*n_c);

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    min_iter = 10;
    int max_loop = min_iter_calc();
    if (max_loop < 200)
        max_loop = 200;
    max_loop = 0;
    
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        float max_change = maxabs(g,n_s*n_c);
		if(DEBUG_ITER) std::cout << "BINARY_AUGLAG_CPU_SOLVER_BASE Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();

    //log output and transpose output back into proper buffer
    log_buffer(u, n_s*n_c);
        
    //deallocate temporary buffers
    clean_up();
}

BINARY_AUGLAG_CPU_SOLVER_BASE::~BINARY_AUGLAG_CPU_SOLVER_BASE(){
    delete [] ps;
}
