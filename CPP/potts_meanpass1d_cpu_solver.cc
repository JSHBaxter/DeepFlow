
#include "potts_meanpass1d_cpu_solver.h"
#include "cpu_kernels.h"
#include <iostream>
#include <cmath>

int POTTS_MEANPASS_CPU_SOLVER_1D::min_iter_calc(){
    return (int) std::sqrt(n_x);
}

void POTTS_MEANPASS_CPU_SOLVER_1D::init_vars(){}

void POTTS_MEANPASS_CPU_SOLVER_1D::calculate_regularization(){
    if(channels_first)
        calculate_r_eff_channels_first(r_eff, rx, u, n_x, n_c);
    else
        calculate_r_eff(r_eff, rx, u, n_x, n_c);
}

void POTTS_MEANPASS_CPU_SOLVER_1D::parity_mask_buffer(float* buffer, const int parity){
    if(channels_first)
        parity_mask_channels_first(buffer,n_x,n_c,parity);
    else
        parity_mask(buffer,n_x,n_c,parity);
}

void POTTS_MEANPASS_CPU_SOLVER_1D::parity_merge_buffer(float* buffer, const float* other, const int parity){
    if(channels_first)
        parity_merge_channels_first(buffer,other,n_x,n_c,parity);
    else
        parity_merge(buffer,other,n_x,n_c,parity);
}

void POTTS_MEANPASS_CPU_SOLVER_1D::clean_up(){}

POTTS_MEANPASS_CPU_SOLVER_1D::POTTS_MEANPASS_CPU_SOLVER_1D(
    const bool channels_first,
    const int batch,
    const int n_c,
    const int sizes[1],
    const float* data_cost,
    const float* rx_cost,
    const float* init_u,
    float* u 
):
POTTS_MEANPASS_CPU_SOLVER_BASE(channels_first, batch, sizes[0], n_c, data_cost, init_u, u),
n_x(sizes[0]),
rx(rx_cost)
{
    //std::cout << n_x << " " << n_c << " " << rx << std::endl;
}

int POTTS_MEANPASS_CPU_GRADIENT_1D::min_iter_calc(){
    return (int) std::sqrt(n_x);
}

void POTTS_MEANPASS_CPU_GRADIENT_1D::init_vars(){
    clear(g_rx, n_c*n_s);
}

void POTTS_MEANPASS_CPU_GRADIENT_1D::get_reg_gradients_and_push(float tau){
    if(channels_first){
        get_reg_gradients_channels_first(d_y, u, g_rx, n_x, n_c, tau);
        get_gradient_for_u_channels_first(d_y, rx, g_u, n_x, n_c, tau);
    }else{
        get_reg_gradients(d_y, u, g_rx, n_x, n_c, tau);
        get_gradient_for_u(d_y, rx, g_u, n_x, n_c, tau);
    }
}

void POTTS_MEANPASS_CPU_GRADIENT_1D::clean_up(){}

POTTS_MEANPASS_CPU_GRADIENT_1D::POTTS_MEANPASS_CPU_GRADIENT_1D(
    const bool channels_first,
    const int batch,
    const int n_c,
    const int sizes[1],
    const float* u,
    const float* g,
    const float* rx_cost,
    float* g_d,
    float* g_rx
) :
POTTS_MEANPASS_CPU_GRADIENT_BASE(channels_first, batch, sizes[0], n_c, u, g, g_d),
n_x(sizes[0]),
rx(rx_cost),
g_rx(g_rx)
{
    //std::cout << n_x << " " << n_c << " " << rx << " " << g_rx << std::endl;
}

