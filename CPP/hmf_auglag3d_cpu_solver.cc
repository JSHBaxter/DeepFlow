#include <math.h>
#include <iostream>
#include <limits>
#include "hmf_auglag3d_cpu_solver.h"
#include "cpu_kernels.h"
#include "hmf_trees.h"
#include <algorithm>

int HMF_AUGLAG_CPU_SOLVER_3D::min_iter_calc(){
    return n_r-n_c + (int) std::sqrt(n_x+n_y+n_z);
}

void HMF_AUGLAG_CPU_SOLVER_3D::clear_spatial_flows(){
    clear(px, n_r*n_s);
    clear(py, n_r*n_s);
    clear(pz, n_r*n_s);
}

void HMF_AUGLAG_CPU_SOLVER_3D::update_spatial_flow_calc(){
    compute_flows_channels_first(g, div, px, py, pz, rx_b, ry_b, rz_b, n_r, n_x, n_y, n_z);
}

void HMF_AUGLAG_CPU_SOLVER_3D::clean_up(){
}

HMF_AUGLAG_CPU_SOLVER_3D::~HMF_AUGLAG_CPU_SOLVER_3D(){
    delete[] px;
    delete[] py;
    delete[] pz;
    if(!channels_first) delete [] rx_b;
    if(!channels_first) delete [] ry_b;
    if(!channels_first) delete [] rz_b;
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
px(new float [n_s*n_r]),
py(new float [n_s*n_r]),
pz(new float [n_s*n_r]),
rx_b( channels_first ? rx : transpose(rx,new float [n_s*n_r],n_s,n_r) ),
ry_b( channels_first ? ry : transpose(ry,new float [n_s*n_r],n_s,n_r) ),
rz_b( channels_first ? rz : transpose(rz,new float [n_s*n_r],n_s,n_r) )
{}
