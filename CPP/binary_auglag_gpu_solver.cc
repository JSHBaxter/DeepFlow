#ifdef GOOGLE_CUDA

#include "binary_auglag_gpu_solver.h"
#include "gpu_kernels.h"

BINARY_AUGLAG_GPU_SOLVER_BASE::BINARY_AUGLAG_GPU_SOLVER_BASE(
    const GPUDevice & dev,
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
ps(full_buff[3])
{
    //std::cout << n_s << " " << n_c << std::endl;
}

void BINARY_AUGLAG_GPU_SOLVER_BASE::block_iter(){
    
    //calculate the capacity and then update flows
	calc_capacity_binary(dev, g, div, ps, pt, u, n_s*n_c, icc, tau);
    update_spatial_flow_calc();
	
	//update source flows, sink flows, and multipliers
	update_source_sink_multiplier_binary(dev, ps, pt, div, u, g, data, cc, icc, n_s*n_c);
}

void BINARY_AUGLAG_GPU_SOLVER_BASE::operator()(){

    //initialize variables
	sigmoid(dev, data, 0, u, n_s*n_c);
    //clear_buffer(dev, u, n_s*n_c);
    clear_spatial_flows();
    clear_buffer(dev, div, n_s*n_c);
    clear_buffer(dev, ps, n_s*n_c);
    clear_buffer(dev, pt, n_s*n_c);
	//find_min_constraint(dev, ps, data, n_c, n_s);
	//rep_buffer(dev, ps, pt, n_c, n_s);
    
    // iterate in blocks
    int min_iter = min_iter_calc();
    if( min_iter < 10 )
        min_iter = 10;
    int max_loop = 200;

    for(int i = 0; i < max_loop; i++){    
        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        //Determine if converged
        float max_change = max_of_buffer(dev, g, n_s*n_c);
		//std::cout << "Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();

    //get final output
    log_buffer(dev, u, u, n_s*n_c);

}
#endif
