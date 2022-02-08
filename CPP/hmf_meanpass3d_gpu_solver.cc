
#include <math.h>
#include <iostream>
#include <limits>
#include "hmf_trees.h"
#include "hmf_meanpass3d_gpu_solver.h"
#include "gpu_kernels.h"
#include <algorithm>
#include <cmath>

int HMF_MEANPASS_GPU_SOLVER_3D::min_iter_calc(){
    return n_r-n_c + (int) std::sqrt(n_x+n_y+n_z);
}

void HMF_MEANPASS_GPU_SOLVER_3D::update_spatial_flow_calc(){
    get_effective_reg(dev, r_eff, u_full, rx, ry, rz, n_x, n_y, n_z, n_r);
}

void HMF_MEANPASS_GPU_SOLVER_3D::parity_mask_buffer(float* buffer, const int parity){
    parity_mask(dev,buffer,n_x,n_y,n_z,n_c,parity);
}

void HMF_MEANPASS_GPU_SOLVER_3D::parity_merge_buffer(float* buffer, const float* other, const int parity){
    parity_mask(dev,buffer,other,n_x,n_y,n_z,n_c,parity);
}

HMF_MEANPASS_GPU_SOLVER_3D::HMF_MEANPASS_GPU_SOLVER_3D(
    const cudaStream_t & dev,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_c,
    const int n_r,
    const int sizes[3],
    const float* data_cost,
    const float* rx_cost,
    const float* ry_cost,
    const float* rz_cost,
    const float* init_u,
    float* const u,
    float** full_buff,
    float** img_buff) :
HMF_MEANPASS_GPU_SOLVER_BASE(dev,
                             bottom_up_list,
                             batch,
                             sizes[0]*sizes[1]*sizes[2],
                             n_c,
                             n_r,
                             data_cost,
                             init_u,
                             u,
                             full_buff,
                             img_buff),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx(rx_cost),
ry(ry_cost),
rz(rz_cost)
{}

int HMF_MEANPASS_GPU_GRADIENT_3D::min_iter_calc(){
    return n_r-n_c + (int) std::sqrt(n_x+n_y+n_z);
}

void HMF_MEANPASS_GPU_GRADIENT_3D::clear_variables(){
    clear_buffer(dev, g_rx, n_s*n_r);
    clear_buffer(dev, g_ry, n_s*n_r);
    clear_buffer(dev, g_rz, n_s*n_r);
}

void HMF_MEANPASS_GPU_GRADIENT_3D::get_reg_gradients_and_push(float tau){
    populate_reg_mean_gradients_and_add(dev, dy, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_r, tau);
    clear_buffer(dev,du,n_s*(n_r-n_c));
    get_gradient_for_u(dev, dy+n_s*(n_r-n_c), du+n_s*(n_r-n_c), rx+n_s*(n_r-n_c), ry+n_s*(n_r-n_c), rz+n_s*(n_r-n_c), n_x, n_y, n_z, n_c, tau);
}

HMF_MEANPASS_GPU_GRADIENT_3D::HMF_MEANPASS_GPU_GRADIENT_3D(
    const cudaStream_t & dev,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_c,
    const int n_r,
    const int sizes[3],
    const float* rx_cost,
    const float* ry_cost,
    const float* rz_cost,
    const float* u,
    const float* g,
    float* g_d,
    float* g_rx,
    float* g_ry,
    float* g_rz,
    float** full_buff) :
HMF_MEANPASS_GPU_GRADIENT_BASE(dev,
                             bottom_up_list,
                             batch,
                             sizes[0]*sizes[1]*sizes[2],
                             n_c,
                             n_r,
                             u,
                             g,
                             g_d,
                             full_buff),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx(rx_cost),
ry(ry_cost),
rz(rz_cost),
g_rx(g_rx),
g_ry(g_ry),
g_rz(g_rz)
{}

