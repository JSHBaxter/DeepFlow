#include <math.h>
#include <iostream>
#include <limits>

#include "potts_auglag1d_cpu_solver.h"
#include "cpu_kernels.h"

int POTTS_AUGLAG_CPU_SOLVER_1D::min_iter_calc(){
    return n_x;
}

void POTTS_AUGLAG_CPU_SOLVER_1D::clear_spatial_flows(){
    px = new float[n_s*n_c];
    clear(px, n_s*n_c);
}

void POTTS_AUGLAG_CPU_SOLVER_1D::update_spatial_flow_calc(){
    if(channels_first)
        compute_flows_channels_first( g, div, px, rx, n_c, n_x);
    else
        compute_flows( g, div, px, rx, n_c, n_x);
}

void POTTS_AUGLAG_CPU_SOLVER_1D::clean_up(){
    if( px ) delete px; px = 0;
}

POTTS_AUGLAG_CPU_SOLVER_1D::POTTS_AUGLAG_CPU_SOLVER_1D(
    const int channels_first,
    const int batch,
    const int n_c,
    const int sizes[1],
    const float* data_cost,
    const float* rx_cost,
    float* u 
):
POTTS_AUGLAG_CPU_SOLVER_BASE(channels_first, batch, sizes[0], n_c, data_cost, u),
n_x(sizes[0]),
rx(rx_cost),
px(0)
{}


