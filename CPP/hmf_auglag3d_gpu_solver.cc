
#include <math.h>
#include <iostream>
#include <limits>
#include "hmf_trees.h"
#include "hmf_auglag3d_gpu_solver.h"
#include "gpu_kernels.h"
#include <algorithm>


int HMF_AUGLAG_GPU_SOLVER_3D::min_iter_calc() {
    return n_r-n_c + (int) std::sqrt(n_x+n_y+n_z);
}

void HMF_AUGLAG_GPU_SOLVER_3D::clear_spatial_flows(){
    clear_buffer(dev, px, n_s*n_r);
    clear_buffer(dev, py, n_s*n_r);
    clear_buffer(dev, pz, n_s*n_r);
}

void HMF_AUGLAG_GPU_SOLVER_3D::update_spatial_flow_calc(){
    update_spatial_flows(dev, g, div, px, py, pz, rx_b, ry_b, rz_b, n_x, n_y, n_z, n_r*n_s);
}

HMF_AUGLAG_GPU_SOLVER_3D::HMF_AUGLAG_GPU_SOLVER_3D(
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
    float* u,
    float** full_buff,
    float** img_buff) :
HMF_AUGLAG_GPU_SOLVER_BASE(dev,
                           bottom_up_list,
                           batch,
                           sizes[0]*sizes[1]*sizes[2],
                           n_c,
                           n_r,
                           data_cost,
                           u,
                           full_buff,
                           img_buff),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx_b(rx_cost),
ry_b(rx_cost),
rz_b(rz_cost),
px(full_buff[4]),
py(full_buff[5]),
pz(full_buff[6])
{
}
