#include <math.h>
#include <iostream>
#include <limits>

#include "potts_auglag3d_cpu_solver.h"
#include "cpu_kernels.h"
#include <algorithm>

int POTTS_AUGLAG_CPU_SOLVER_3D::min_iter_calc(){
    return std::max(std::max(n_x,n_y), n_z);
}

void POTTS_AUGLAG_CPU_SOLVER_3D::clear_spatial_flows(){
    px = new float[n_s*n_c];
    py = new float[n_s*n_c];
    pz = new float[n_s*n_c];
    clear(px, py, pz, n_s*n_c);
}

void POTTS_AUGLAG_CPU_SOLVER_3D::update_spatial_flow_calc(){
    if(channels_first)
        compute_flows_channels_first(g, div, px, py, pz, rx, ry, rz, n_c, n_x, n_y, n_z);
    else
        compute_flows(g, div, px, py, pz, rx, ry, rz, n_c, n_x, n_y, n_z);
}

void POTTS_AUGLAG_CPU_SOLVER_3D::clean_up(){
    if( px ) delete px; px = 0;
    if( py ) delete py; py = 0;
    if( pz ) delete pz; pz = 0;
}

POTTS_AUGLAG_CPU_SOLVER_3D::POTTS_AUGLAG_CPU_SOLVER_3D(
    const bool channels_first,
    const int batch,
    const int n_c,
    const int sizes[3],
    const float* data_cost,
    const float* rx_cost,
    const float* ry_cost,
    const float* rz_cost,
    float* u 
):
POTTS_AUGLAG_CPU_SOLVER_BASE(channels_first, batch, sizes[0]*sizes[1]*sizes[2], n_c, data_cost, u),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx(rx_cost),
ry(ry_cost),
rz(rz_cost),
px(0),
py(0),
pz(0)
{}
