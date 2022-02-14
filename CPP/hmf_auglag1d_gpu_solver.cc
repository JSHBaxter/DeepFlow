

#include <math.h>
#include <iostream>
#include <limits>
#include "hmf_trees.h"
#include "hmf_auglag1d_gpu_solver.h"
#include "gpu_kernels.h"


int HMF_AUGLAG_GPU_SOLVER_1D::min_iter_calc() {
    return n_r-n_c + (int) std::sqrt(n_x);
}

void HMF_AUGLAG_GPU_SOLVER_1D::clear_spatial_flows(){
    clear_buffer(dev, px, n_s*n_r);
}

void HMF_AUGLAG_GPU_SOLVER_1D::update_spatial_flow_calc(){
    update_spatial_flows(dev, g, div, px, rx_b, n_x, n_r*n_s);
    if(DEBUG_PRINT) print_buffer(dev,px,n_s*n_r);
}

HMF_AUGLAG_GPU_SOLVER_1D::HMF_AUGLAG_GPU_SOLVER_1D(
    const cudaStream_t & dev,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_c,
    const int n_r,
    const int sizes[1],
    const float* data_cost,
    const float* rx_cost,
    float* u,
    float** full_buff,
    float** img_buff) :
HMF_AUGLAG_GPU_SOLVER_BASE(dev,
                           bottom_up_list,
                           batch,
                           sizes[0],
                           n_c,
                           n_r,
                           data_cost,
                           u,
                           full_buff,
                           img_buff),
n_x(sizes[0]),
rx_b(rx_cost),
px(full_buff[4])
{}
