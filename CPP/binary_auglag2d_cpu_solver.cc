
#include "binary_auglag2d_cpu_solver.h"
#include "cpu_kernels.h"


int BINARY_AUGLAG_CPU_SOLVER_2D::min_iter_calc(){
	return std::max(n_x,n_y);
}

void BINARY_AUGLAG_CPU_SOLVER_2D::clear_spatial_flows(){
	px = new float[n_s*n_c];
	py = new float[n_s*n_c];
	clear(px, py, n_s*n_c);
}

void BINARY_AUGLAG_CPU_SOLVER_2D::update_spatial_flow_calc(){
    if(channels_first)
        compute_flows_channels_first( g, div, px, py, rx, ry, n_c, n_x, n_y);
    else
        compute_flows( g, div, px, py, rx, ry, n_c, n_x, n_y);
}

void BINARY_AUGLAG_CPU_SOLVER_2D::clean_up(){
	if( px ) delete px; px = 0;
	if( py ) delete py; py = 0;
}

BINARY_AUGLAG_CPU_SOLVER_2D::BINARY_AUGLAG_CPU_SOLVER_2D(
    const bool channels_first,
	const int batch,
    const int n_c,
	const int sizes[2],
	const float* data_cost,
	const float* rx_cost,
	const float* ry_cost,
	float* u 
):
BINARY_AUGLAG_CPU_SOLVER_BASE(channels_first, batch, sizes[0]*sizes[1], n_c, data_cost, u),
n_x(sizes[0]),
n_y(sizes[1]),
rx(rx_cost),
ry(ry_cost),
px(0),
py(0)
{}
