
#include "potts_meanpass1d_gpu_solver.h"
#include "gpu_kernels.h"


int POTTS_MEANPASS_GPU_SOLVER_1D::min_iter_calc(){
    return n_x;
}

void POTTS_MEANPASS_GPU_SOLVER_1D::init_vars(){}

void POTTS_MEANPASS_GPU_SOLVER_1D::calculate_regularization(){
    get_effective_reg(dev, r_eff, u, rx, n_x, n_c);
}

void POTTS_MEANPASS_GPU_SOLVER_1D::parity_mask_buffer(float* buffer, const int parity){
    parity_mask(dev,buffer,n_x,n_c,parity);
}

void POTTS_MEANPASS_GPU_SOLVER_1D::parity_merge_buffer(float* buffer, const float* other, const int parity){
    parity_mask(dev,buffer,other,n_x,n_c,parity);
}

void POTTS_MEANPASS_GPU_SOLVER_1D::clean_up(){}

POTTS_MEANPASS_GPU_SOLVER_1D::POTTS_MEANPASS_GPU_SOLVER_1D(
    const cudaStream_t & dev,
    const int batch,
    const int n_c,
    const int sizes[1],
    const float* data_cost,
    const float* rx_cost,
    const float* init_u,
    float* u,
    float** buffers_full
):
POTTS_MEANPASS_GPU_SOLVER_BASE(dev, batch, sizes[0], n_c, data_cost, init_u, u, buffers_full),
n_x(sizes[0]),
rx(rx_cost)
{}

int POTTS_MEANPASS_GPU_GRADIENT_1D::min_iter_calc(){
    return n_x;
}

void POTTS_MEANPASS_GPU_GRADIENT_1D::init_vars(){
    clear_buffer(dev, g_rx, n_c*n_s);
}

void POTTS_MEANPASS_GPU_GRADIENT_1D::get_reg_gradients_and_push(float tau){
    populate_reg_mean_gradients_and_add(dev, d_y, u, g_rx, n_x, n_c, tau);
    get_gradient_for_u(dev, d_y, g_u, rx, n_x, n_c, tau);
}

void POTTS_MEANPASS_GPU_GRADIENT_1D::clean_up(){}

POTTS_MEANPASS_GPU_GRADIENT_1D::POTTS_MEANPASS_GPU_GRADIENT_1D(
    const cudaStream_t & dev,
    const int batch,
    const int n_c,
    const int sizes[1],
    const float* u,
    const float* g,
    const float* rx_cost,
    float* g_d,
    float* g_rx,
    float** full_buffs
) :
POTTS_MEANPASS_GPU_GRADIENT_BASE(dev, batch, sizes[0], n_c, u, g, g_d, full_buffs),
n_x(sizes[0]),
rx(rx_cost),
g_rx(g_rx)
{}
