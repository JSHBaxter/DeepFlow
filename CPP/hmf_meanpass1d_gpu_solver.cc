
#include <algorithm>
#include "hmf_trees.h"
#include "hmf_meanpass1d_gpu_solver.h"
#include "gpu_kernels.h"
#include <cmath>

int HMF_MEANPASS_GPU_SOLVER_1D::min_iter_calc(){
    return n_r-n_c + (int) std::sqrt(n_x);
}

void HMF_MEANPASS_GPU_SOLVER_1D::update_spatial_flow_calc(){
    get_effective_reg(dev, r_eff, u_full, rx, n_x, n_r);
}

void HMF_MEANPASS_GPU_SOLVER_1D::parity_mask_buffer(float* buffer, const int parity){
    parity_mask(dev,buffer,n_x,n_c,parity);
}

void HMF_MEANPASS_GPU_SOLVER_1D::parity_merge_buffer(float* buffer, const float* other, const int parity){
    parity_mask(dev,buffer,other,n_x,n_c,parity);
}

HMF_MEANPASS_GPU_SOLVER_1D::HMF_MEANPASS_GPU_SOLVER_1D(
    const cudaStream_t & dev,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_c,
    const int n_r,
    const int sizes[1],
    const float* data_cost,
    const float* rx_cost,
    const float* init_u,
    float* const u,
    float** full_buff,
    float** img_buff) :
HMF_MEANPASS_GPU_SOLVER_BASE(dev,
                             bottom_up_list,
                             batch,
                             sizes[0],
                             n_c,
                             n_r,
                             data_cost,
                             init_u,
                             u,
                             full_buff,
                             img_buff),
n_x(sizes[0]),
rx(rx_cost)
{}

int HMF_MEANPASS_GPU_GRADIENT_1D::min_iter_calc(){
    return n_r-n_c + (int) std::sqrt(n_x);
}

void HMF_MEANPASS_GPU_GRADIENT_1D::clear_variables(){
    clear_buffer(dev, g_rx, n_s*n_r);
}

void HMF_MEANPASS_GPU_GRADIENT_1D::get_reg_gradients_and_push(float tau){
    populate_reg_mean_gradients_and_add(dev, dy, u, g_rx, n_x, n_r, tau);
    get_gradient_for_u(dev, dy, du, rx, n_x, n_r, tau);
}

HMF_MEANPASS_GPU_GRADIENT_1D::HMF_MEANPASS_GPU_GRADIENT_1D(
    const cudaStream_t & dev,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_c,
    const int n_r,
    const int sizes[1],
    const float* rx_cost,
    const float* u,
    const float* g,
    float* g_d,
    float* g_rx,
    float** full_buff) :
HMF_MEANPASS_GPU_GRADIENT_BASE(dev,
                             bottom_up_list,
                             batch,
                             sizes[0],
                             n_c,
                             n_r,
                             u,
                             g,
                             g_d,
                             full_buff),
n_x(sizes[0]),
rx(rx_cost),
g_rx(g_rx)
{}

