
#include "potts_meanpass3d_gpu_solver.h"
#include "gpu_kernels.h"
#include <algorithm>
#include <cmath>
#include <iostream>

int POTTS_MEANPASS_GPU_SOLVER_3D::min_iter_calc(){
    return (int) std::sqrt(n_x+n_y+n_z);
}
void POTTS_MEANPASS_GPU_SOLVER_3D::init_vars(){}


void POTTS_MEANPASS_GPU_SOLVER_3D::calculate_regularization(){
    get_effective_reg(dev, r_eff, u, rx, ry, rz, n_x, n_y, n_z, n_c);
}

void POTTS_MEANPASS_GPU_SOLVER_3D::parity_mask_buffer(float* buffer, const int parity){
    parity_mask(dev,buffer,n_x,n_y,n_z,n_c,parity);
}

void POTTS_MEANPASS_GPU_SOLVER_3D::parity_merge_buffer(float* buffer, const float* other, const int parity){
    parity_mask(dev,buffer,other,n_x,n_y,n_z,n_c,parity);
}

void POTTS_MEANPASS_GPU_SOLVER_3D::clean_up(){}

POTTS_MEANPASS_GPU_SOLVER_3D::POTTS_MEANPASS_GPU_SOLVER_3D(
    const cudaStream_t & dev,
    const int batch,
    const int n_c,
    const int sizes[3],
    const float* data_cost,
    const float* rx_cost,
    const float* ry_cost,
    const float* rz_cost,
    const float* init_u,
    float* u,
    float** buffers_full
):
POTTS_MEANPASS_GPU_SOLVER_BASE(dev, batch, sizes[0]*sizes[1]*sizes[2], n_c, data_cost, init_u, u, buffers_full),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx(rx_cost),
ry(ry_cost),
rz(rz_cost)
{}

int POTTS_MEANPASS_GPU_GRADIENT_3D::min_iter_calc(){
    return (int) std::sqrt(n_x+n_y+n_z);
}

void POTTS_MEANPASS_GPU_GRADIENT_3D::init_vars(){
    clear_buffer(dev, g_rx, n_c*n_s);
    clear_buffer(dev, g_ry, n_c*n_s);
    clear_buffer(dev, g_rz, n_c*n_s);
}

void POTTS_MEANPASS_GPU_GRADIENT_3D::get_reg_gradients_and_push(float tau){
    populate_reg_mean_gradients_and_add(dev, d_y, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_c, tau);
    get_gradient_for_u(dev, d_y, g_u, rx, ry, rz, n_x, n_y, n_z, n_c, tau);
}

void POTTS_MEANPASS_GPU_GRADIENT_3D::clean_up(){}

POTTS_MEANPASS_GPU_GRADIENT_3D::POTTS_MEANPASS_GPU_GRADIENT_3D(
    const cudaStream_t & dev,
    const int batch,
    const int n_c,
    const int sizes[3],
    const float* u,
    const float* g,
    const float* rx_cost,
    const float* ry_cost,
    const float* rz_cost,
    float* g_d,
    float* g_rx,
    float* g_ry,
    float* g_rz,
    float** full_buffs
) :
POTTS_MEANPASS_GPU_GRADIENT_BASE(dev, batch, sizes[0]*sizes[1]*sizes[2], n_c, u, g, g_d, full_buffs),
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


