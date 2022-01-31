#include <math.h>
#include <iostream>
#include <limits>

#include "binary_meanpass3d_cpu_solver.h"
#include "cpu_kernels.h"

int BINARY_MEANPASS_CPU_SOLVER_3D::min_iter_calc(){
	return n_x+n_y+n_z;
}

void BINARY_MEANPASS_CPU_SOLVER_3D::init_vars(){}

void BINARY_MEANPASS_CPU_SOLVER_3D::calculate_regularization(){
    if(channels_first)
	    calculate_r_eff_channels_first(r_eff, rx, ry, rz, u, n_x, n_y, n_z, n_c);
    else
        calculate_r_eff(r_eff, rx, ry, rz, u, n_x, n_y, n_z, n_c);
}

void BINARY_MEANPASS_CPU_SOLVER_3D::parity_mask_buffer(float* buffer, const int parity){
    if(channels_first)
	    parity_mask_channels_first(buffer,n_x,n_y,n_z,n_c,parity);
    else
	    parity_mask(buffer,n_x,n_y,n_z,n_c,parity);
}

void BINARY_MEANPASS_CPU_SOLVER_3D::parity_merge_buffer(float* buffer, const float* other, const int parity){
    if(channels_first)
	    parity_merge_channels_first(buffer,other,n_x,n_y,n_z,n_c,parity);
    else
	    parity_merge(buffer,other,n_x,n_y,n_z,n_c,parity);
}

void BINARY_MEANPASS_CPU_SOLVER_3D::clean_up(){}

BINARY_MEANPASS_CPU_SOLVER_3D::BINARY_MEANPASS_CPU_SOLVER_3D(
    const bool channels_first,
	const int batch,
    const int n_c,
    const int sizes[3],
	const float* data_cost,
	const float* rx_cost,
	const float* ry_cost,
	const float* rz_cost,
	const float* init_u,
	float* u 
):
BINARY_MEANPASS_CPU_SOLVER_BASE(channels_first, batch, sizes[0]*sizes[1]*sizes[2], n_c, data_cost, init_u, u),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx(rx_cost),
ry(ry_cost),
rz(rz_cost)
{}

int BINARY_MEANPASS_CPU_GRADIENT_3D::min_iter_calc(){
	return n_x+n_y+n_z;
}

void BINARY_MEANPASS_CPU_GRADIENT_3D::init_vars(){
	clear(g_rx, g_ry, g_rz, n_c*n_s);
}

void BINARY_MEANPASS_CPU_GRADIENT_3D::get_reg_gradients_and_push(float tau){
    if(channels_first){
        get_reg_gradients_channels_first(d_y, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_c, tau);
        get_gradient_for_u_channels_first(d_y, rx, ry, rz, g_u, n_x, n_y, n_z, n_c, tau);
    }else{
        get_reg_gradients(d_y, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_c, tau);
        get_gradient_for_u(d_y, rx, ry, rz, g_u, n_x, n_y, n_z, n_c, tau);
    }
}

void BINARY_MEANPASS_CPU_GRADIENT_3D::clean_up(){}

BINARY_MEANPASS_CPU_GRADIENT_3D::BINARY_MEANPASS_CPU_GRADIENT_3D(
    const bool channels_first,
	const int batch,
    const int n_c,
    const int sizes[3],
	const float* u,
	const float* g,
	const float* rx_cost,
	const float* ry_cost,
	const float* rz_cost,
	float* g_d,
	float* g_rx,
	float* g_ry,
	float* g_rz
) :
BINARY_MEANPASS_CPU_GRADIENT_BASE(channels_first, batch, sizes[0]*sizes[1]*sizes[2], n_c, u, g, g_d),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx(rx_cost),
ry(ry_cost),
rz(rz_cost),
g_rx(g_rx),
g_ry(g_ry),
g_rz(g_rz)
{}
