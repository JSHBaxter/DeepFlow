
#include <math.h>
#include <iostream>
#include <limits>

#include "hmf_auglag1d_cpu_solver.h"
#include "cpu_kernels.h"
#include "hmf_trees.h"

int HMF_AUGLAG_CPU_SOLVER_1D::min_iter_calc(){
    return n_x+n_r-n_c;
}

void HMF_AUGLAG_CPU_SOLVER_1D::clear_spatial_flows(){
    clear(px, n_r*n_s);
}

void HMF_AUGLAG_CPU_SOLVER_1D::update_spatial_flow_calc(){
    compute_flows_channels_first(g, div, px, rx_b, n_r, n_x);
}

void HMF_AUGLAG_CPU_SOLVER_1D::clean_up(){
}

HMF_AUGLAG_CPU_SOLVER_1D::~HMF_AUGLAG_CPU_SOLVER_1D(){
    delete px;
}

HMF_AUGLAG_CPU_SOLVER_1D::HMF_AUGLAG_CPU_SOLVER_1D(
    const bool channels_first,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_c,
    const int n_r,
    const int sizes[1],
    const float* data_cost,
    const float* rx_cost,
    float* u ) :
HMF_AUGLAG_CPU_SOLVER_BASE(channels_first,
                           bottom_up_list,
                           batch,
                           sizes[0],
                           n_c,
                           n_r,
                           data_cost,
                           u),
n_x(sizes[0]),
rx(rx_cost),
px(new float [(channels_first ? 1 : 2)*n_s*n_r]),
rx_b( channels_first ? rx : transpose(rx,px+n_r*n_s,n_s,n_r) )
{}

