#include <math.h>
#include <iostream>
#include <limits>
#include "hmf_auglag2d_cpu_solver.h"
#include "cpu_kernels.h"
#include "hmf_trees.h"
#include <algorithm>

int HMF_AUGLAG_CPU_SOLVER_2D::min_iter_calc(){
    return n_r-n_c + (int) std::sqrt(n_x+n_y);
}

void HMF_AUGLAG_CPU_SOLVER_2D::clear_spatial_flows(){
    clear(px, n_r*n_s);
    clear(py, n_r*n_s);
}

void HMF_AUGLAG_CPU_SOLVER_2D::update_spatial_flow_calc(){
    compute_flows_channels_first(g, div, px, py, rx_b, ry_b, n_r, n_x, n_y);
    if(DEBUG_PRINT) print_buffer(px,n_s*n_r);
    if(DEBUG_PRINT) print_buffer(py,n_s*n_r);
}

void HMF_AUGLAG_CPU_SOLVER_2D::clean_up(){
}

HMF_AUGLAG_CPU_SOLVER_2D::~HMF_AUGLAG_CPU_SOLVER_2D(){
    delete[] px;
    delete[] py;
    if(!channels_first) delete [] rx_b;
    if(!channels_first) delete [] ry_b;
}

HMF_AUGLAG_CPU_SOLVER_2D::HMF_AUGLAG_CPU_SOLVER_2D(
    const bool channels_first,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_c,
    const int n_r,
    const int sizes[2],
    const float* data_cost,
    const float* rx_cost,
    const float* ry_cost,
    float* u ) :
HMF_AUGLAG_CPU_SOLVER_BASE(channels_first,
                           bottom_up_list,
                           batch,
                           sizes[0]*sizes[1],
                           n_c,
                           n_r,
                           data_cost,
                           u),
n_x(sizes[0]),
n_y(sizes[1]),
rx(rx_cost),
ry(ry_cost),
px(new float[n_s*n_r]),
py(new float[n_s*n_r]),
rx_b( channels_first ? rx : transpose(rx,new float [n_s*n_r],n_s,n_r) ),
ry_b( channels_first ? ry : transpose(ry,new float [n_s*n_r],n_s,n_r) )
{}
