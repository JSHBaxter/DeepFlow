#include <math.h>
#include <iostream>
#include <limits>
#include "hmf_auglag3d_cpu_solver.h"
#include "cpu_kernels.h"
#include "hmf_trees.h"
#include <algorithm>

int HMF_AUGLAG_CPU_SOLVER_3D::min_iter_calc(){
    return std::max(n_x,std::max(n_y,n_z))+n_r-n_c;
}

void HMF_AUGLAG_CPU_SOLVER_3D::clear_spatial_flows(){
    clear(px, 3*n_r*n_s);
}

void HMF_AUGLAG_CPU_SOLVER_3D::update_spatial_flow_calc(){
    compute_flows_channels_first(g, div, px, py, pz, rx_b, ry_b, rz_b, n_r, n_x, n_y, n_z);
}

void HMF_AUGLAG_CPU_SOLVER_3D::clean_up(){
}

HMF_AUGLAG_CPU_SOLVER_3D::~HMF_AUGLAG_CPU_SOLVER_3D(){
    delete px;
}

HMF_AUGLAG_CPU_SOLVER_3D::HMF_AUGLAG_CPU_SOLVER_3D(
    const bool channels_first,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_c,
    const int n_r,
    const int sizes[3],
    const float* data_cost,
    const float* rx_cost,
    const float* ry_cost,
    const float* rz_cost,
    float* u ) :
HMF_AUGLAG_CPU_SOLVER_BASE(channels_first,
                           bottom_up_list,
                           batch,
                           sizes[0]*sizes[1]*sizes[2],
                           n_c,
                           n_r,
                           data_cost,
                           u),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx(rx_cost),
ry(ry_cost),
rz(rz_cost),
px(new float [3*(channels_first ? 1 : 2)*n_s*n_r]),
py(px+n_s*n_r),
pz(py+n_s*n_r),
rx_b( channels_first ? rx : transpose(rx,pz+1*n_r*n_s,n_s,n_r) ),
ry_b( channels_first ? ry : transpose(ry,pz+2*n_r*n_s,n_s,n_r) ),
rz_b( channels_first ? rz : transpose(rz,pz+3*n_r*n_s,n_s,n_r) )
{}
