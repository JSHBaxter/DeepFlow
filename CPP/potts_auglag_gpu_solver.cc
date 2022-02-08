
#include "potts_auglag_gpu_solver.h"
#include "gpu_kernels.h"
#include <iostream>

POTTS_AUGLAG_GPU_SOLVER_BASE::POTTS_AUGLAG_GPU_SOLVER_BASE(
    const cudaStream_t & dev,
    const int batch,
    const int n_s,
    const int n_c,
    const float* data_cost,
    float* u,
    float** full_buff,
    float** img_buff) :
dev(dev),
b(batch),
n_c(n_c),
n_s(n_s),
data(data_cost),
u(u),
pt(full_buff[0]),
div(full_buff[1]),
g(full_buff[2]),
ps(img_buff[0])
{
    //std::cout << n_s << " " << n_c << " " << data_cost << " " << ps << " " << pt << " " << div << " " << g << " " << u << std::endl;
}

void POTTS_AUGLAG_GPU_SOLVER_BASE::block_iter(){
    
    //calculate the capacity and then update flows
	calc_capacity_potts(dev, g, div, ps, pt, u, n_s, n_c, icc, tau);
    update_spatial_flow_calc();
	
	//update source flows, sink flows, and multipliers
	update_source_sink_multiplier_potts(dev, ps, pt, div, u, g, data, cc, icc, n_c, n_s);
}

void POTTS_AUGLAG_GPU_SOLVER_BASE::operator()(){

    //initialize variables
    clear_spatial_flows();
    clear_buffer(dev, div, n_s*n_c);
    init_flows_potts(dev, data, ps, pt, u, n_s, n_c);
    
    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    //min_iter = 1;
    int max_loop = min_iter_calc();
    if (max_loop < 200)
        max_loop = 200;
    //max_loop = 0;
    
    for(int i = 0; i < max_loop; i++){    
        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        //Determine if converged
        float max_change = max_of_buffer(dev, g, n_s*n_c);
		//std::cout << "POTTS_AUGLAG_GPU_SOLVER_BASE Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();

    //get final output
    log_buffer(dev, u, u, n_s*n_c);

}
