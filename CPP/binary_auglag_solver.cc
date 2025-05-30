
#include "binary_auglag_solver.h"
#include "spatial_star_auglag.h"
#include "common.h"

#include "cpu_kernels.h"
#include "cpu_kernels_auglag.h"
#ifdef USE_CUDA
#include "gpu_kernels.h"
#include "gpu_kernels_auglag.h"
#endif

#include <iostream>

template<typename DEV>
BINARY_AUGLAG_SOLVER<DEV>::BINARY_AUGLAG_SOLVER(
    const DEV & dev,
    const bool star,
    const int dim,
    const int* dims,
    const int n_c,
    const float * const * const inputs,
    float* u) :
MAXFLOW_ALGORITHM<DEV>(dev),
n_c(n_c),
n_s(product(dim,dims)),
data(inputs[0]),
u(u),
pt(0),
div(0),
g(0),
ps(0)
{
    if(DEBUG_ITER) std::cout << "BINARY_AUGLAG_SOLVER Constructor " << dim << " " << n_s << " " << n_c << std::endl;
    if(star)
        spatial_flow = new SPATIAL_STAR_AUGLAG_SOLVER<DEV>(dev, n_c, dims, dim, inputs+1);
    else
        spatial_flow = new SPATIAL_AUGLAG_SOLVER<DEV>(dev, n_c, dims, dim, inputs+1);
    MAXFLOW_ALGORITHM<DEV>::construct();
}

template<typename DEV>
void BINARY_AUGLAG_SOLVER<DEV>::allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over){
    if(DEBUG_ITER) std::cout << "BINARY_AUGLAG_SOLVER Allocate" << std::endl;
    ps = buffer;
    pt = ps + n_s*n_c;
    div = pt + n_s*n_c;
    g = div + n_s*n_c;
    float* carry_over_tmp[] = {g, div};
    spatial_flow->allocate_buffers(g+n_s*n_c, carry_over_tmp, 0);
}

template<typename DEV>
int BINARY_AUGLAG_SOLVER<DEV>::get_buffer_size(){
    return 4*n_s*n_c + spatial_flow->get_buffer_size();
}

template<typename DEV>
BINARY_AUGLAG_SOLVER<DEV>::~BINARY_AUGLAG_SOLVER<DEV>(){
    delete spatial_flow;
}

template<typename DEV>
void BINARY_AUGLAG_SOLVER<DEV>::block_iter(){
    
    //calculate the capacity and then update flows
    calc_capacity_binary(MAXFLOW_ALGORITHM<DEV>::dev, g, div, ps, pt, u, n_s*n_c, icc, tau);
    if(DEBUG_ITER_EXTREME){ std::cout << "ps: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, ps, n_s*n_c);}
    if(DEBUG_ITER_EXTREME){ std::cout << "pt: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, pt, n_s*n_c);}
    if(DEBUG_ITER_EXTREME){ std::cout << "div: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, div, n_s*n_c);}
    if(DEBUG_ITER_EXTREME){ std::cout << "u: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u, n_s*n_c);}
    if(DEBUG_ITER_EXTREME){ std::cout << "g: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, g, n_s*n_c);}
    
    spatial_flow->run();
    if(DEBUG_ITER_EXTREME){ std::cout << "ps: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, ps, n_s*n_c);}
    if(DEBUG_ITER_EXTREME){ std::cout << "pt: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, pt, n_s*n_c);}
    if(DEBUG_ITER_EXTREME){ std::cout << "div: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, div, n_s*n_c);}
    if(DEBUG_ITER_EXTREME){ std::cout << "u: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u, n_s*n_c);}
    if(DEBUG_ITER_EXTREME){ std::cout << "g: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, g, n_s*n_c);}
	
    //update source flows, sink flows, and multipliers
    update_source_sink_multiplier_binary(MAXFLOW_ALGORITHM<DEV>::dev, ps, pt, div, u, g, data, cc, icc, n_s*n_c);
    if(DEBUG_ITER_EXTREME){ std::cout << "ps: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, ps, n_s*n_c);}
    if(DEBUG_ITER_EXTREME){ std::cout << "pt: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, pt, n_s*n_c);}
    if(DEBUG_ITER_EXTREME){ std::cout << "div: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, div, n_s*n_c);}
    if(DEBUG_ITER_EXTREME){ std::cout << "u: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u, n_s*n_c);}
    if(DEBUG_ITER_EXTREME){ std::cout << "g: " ; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev, g, n_s*n_c);}
}

template<typename DEV>
void BINARY_AUGLAG_SOLVER<DEV>::run(){
    if(DEBUG_ITER) std::cout << "BINARY_AUGLAG_SOLVER Init" << std::endl;

    //initialize variables
    spatial_flow->init();
    init_flows_binary(MAXFLOW_ALGORITHM<DEV>::dev, data, ps, pt, u, n_s*n_c);
    
    // iterate in blocks
    int min_iter = spatial_flow->get_min_iter();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = spatial_flow->get_max_iter();
    if (max_loop < 10)
        max_loop = 10;
    
    for(int i = 0; i < max_loop; i++){    
        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++){
            if(DEBUG_ITER_EXTREME) std::cout << "\nBlock iter " << iter << std::endl;
            block_iter();
        }

        //Determine if converged
        float max_change = max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, g, n_s*n_c);
		if(DEBUG_ITER) std::cout << "BINARY_AUGLAG_SOLVER Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();

    //get final output
    log_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u, u, n_s*n_c);
    
    //deallocate temporary buffers
    spatial_flow->deinit();
}

