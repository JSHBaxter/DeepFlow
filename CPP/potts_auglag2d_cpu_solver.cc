#include <math.h>
#include <iostream>
#include <limits>

#include "potts_auglag2d_cpu_solver.h"
#include "cpu_kernels.h"
#include <algorithm>

int POTTS_AUGLAG_CPU_SOLVER_2D::min_iter_calc(){
    return (int) std::sqrt(n_x+n_y);
}

void POTTS_AUGLAG_CPU_SOLVER_2D::clear_spatial_flows(){
    clear(px, n_s*n_c);
    clear(py, n_s*n_c);
}

void POTTS_AUGLAG_CPU_SOLVER_2D::update_spatial_flow_calc(){
    if(channels_first)
        compute_flows_channels_first( g, div, px, py, rx, ry, n_c, n_x, n_y);
    else
        compute_flows( g, div, px, py, rx, ry, n_c, n_x, n_y);
}

void POTTS_AUGLAG_CPU_SOLVER_2D::clean_up(){
}

POTTS_AUGLAG_CPU_SOLVER_2D::~POTTS_AUGLAG_CPU_SOLVER_2D(){
    if( px ) delete [] px;
    if( py ) delete [] py;
}

POTTS_AUGLAG_CPU_SOLVER_2D::POTTS_AUGLAG_CPU_SOLVER_2D(
    const bool channels_first,
    const int batch,
    const int n_c,
    const int sizes[2],
    const float* data_cost,
    const float* rx_cost,
    const float* ry_cost,
    float* u 
):
POTTS_AUGLAG_CPU_SOLVER_BASE(channels_first, batch, sizes[0]*sizes[1], n_c, data_cost, u),
n_x(sizes[0]),
n_y(sizes[1]),
rx(rx_cost),
ry(ry_cost),
px(new float[n_s*n_c]),
py(new float[n_s*n_c])
{
    //std::cout << n_x << " " << n_y << " " << n_c << " " << rx << " " << ry << " " << px << " " << py << std::endl;
}

