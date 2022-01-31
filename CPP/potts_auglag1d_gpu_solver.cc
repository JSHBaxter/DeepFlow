#include "potts_auglag1d_gpu_solver.h"
#include "gpu_kernels.h"


int POTTS_AUGLAG_GPU_SOLVER_1D::min_iter_calc(){
    return n_x;
}

void POTTS_AUGLAG_GPU_SOLVER_1D::clear_spatial_flows(){
    clear_buffer(dev, px, n_c*n_s);
}

void POTTS_AUGLAG_GPU_SOLVER_1D::update_spatial_flow_calc(){
    update_spatial_flows(dev, g, div, px, rx, n_x, n_s*n_c);
}

POTTS_AUGLAG_GPU_SOLVER_1D::POTTS_AUGLAG_GPU_SOLVER_1D(
    const cudaStream_t & dev,
    const int batch,
    const int n_c,
    const int sizes[1],
    const float* data_cost,
    const float* rx_cost,
    float* u,
    float** buffers_full,
    float** buffers_img
):
POTTS_AUGLAG_GPU_SOLVER_BASE(dev, batch, sizes[0], n_c, data_cost, u, buffers_full, buffers_img),
n_x(sizes[0]),
rx(rx_cost),
px(buffers_full[3])
{}

