#include "potts_auglag_cpu_solver.h"
#include <math.h>
#include <iostream>
#include <limits>
#include "cpu_kernels.h"

POTTS_AUGLAG_CPU_SOLVER_BASE::POTTS_AUGLAG_CPU_SOLVER_BASE(
    const bool channels_first,
    const int batch,
    const int n_s,
    const int n_c,
    const float* data_cost,
    float* u ) :
channels_first(channels_first),
b(batch),
n_c(n_c),
n_s(n_s),
data(data_cost),
ps(new float[n_s]),
pt(new float[n_s*n_c]),
div(new float[n_s*n_c]),
g(new float[n_s*n_c]),
u(u)
{
    //std::cout << n_s << " " << n_c << " " << data_cost << " " << ps << " " << pt << " " << div << " " << g << " " << u<< std::endl;
}

//perform one iteration of the algorithm
void POTTS_AUGLAG_CPU_SOLVER_BASE::block_iter(){
    
    //calculate the capacity and then update flows
    if(channels_first)
        compute_capacity_potts_channels_first(g, u, ps, pt, div, n_s, n_c, tau, icc);
    else
        compute_capacity_potts(g, u, ps, pt, div, n_s, n_c, tau, icc);
    update_spatial_flow_calc();
                 
	//update source flows, sink flows, and multipliers
    if(channels_first)
	    compute_source_sink_multipliers_channels_first( g, u, ps, pt, div, data, cc, icc, n_c, n_s);
    else
	    compute_source_sink_multipliers( g, u, ps, pt, div, data, cc, icc, n_c, n_s);
        
}

void POTTS_AUGLAG_CPU_SOLVER_BASE::operator()(){
    
    //initialize flows and labels
    clear(g, div, n_c*n_s);
    clear_spatial_flows();
    //clear(pt, n_c*n_s);
    //clear(ps, n_s);
    if(channels_first)
	    init_flows_channels_first(data, ps, pt, u, n_c, n_s);
    else
	    init_flows(data, ps, pt, u, n_c, n_s);

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

        float max_change = maxabs(g,n_s*n_c);
        if(DEBUG_ITER) std::cout << "POTTS_AUGLAG_CPU_SOLVER_BASE Iter " << i << ": " << max_change << std::endl;
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

POTTS_AUGLAG_CPU_SOLVER_BASE::~POTTS_AUGLAG_CPU_SOLVER_BASE(){
    delete [] ps;
    delete [] pt;
    delete [] g;
    delete [] div;
}
