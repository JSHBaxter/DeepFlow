
#include <algorithm>
#include <iostream>
#include "cpu_kernels.h"
#include "hmf_meanpass1d_cpu_solver.h"

int HMF_MEANPASS_CPU_SOLVER_1D::min_iter_calc(){
    return n_x+n_r-n_c;
}

void HMF_MEANPASS_CPU_SOLVER_1D::init_reg_info(){
}

void HMF_MEANPASS_CPU_SOLVER_1D::clean_up(){
}

void HMF_MEANPASS_CPU_SOLVER_1D::update_spatial_flow_calc(){
    calculate_r_eff_channels_first(r_eff, rx_b, u_tmp, n_x, n_r);
}

void HMF_MEANPASS_CPU_SOLVER_1D::parity_mask_buffer(float* buffer, const int parity){
    parity_mask_channels_first(buffer,n_x,n_c,parity);
}

void HMF_MEANPASS_CPU_SOLVER_1D::parity_merge_buffer(float* buffer, const float* other, const int parity){
    parity_merge_channels_first(buffer,other,n_x,n_c,parity);
}

HMF_MEANPASS_CPU_SOLVER_1D::HMF_MEANPASS_CPU_SOLVER_1D(
    const bool channels_first,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_c,
    const int n_r,
    const int sizes[1],
    const float* data_cost,
    const float* rx_cost,
    const float* init_u,
    float* u ) :
HMF_MEANPASS_CPU_SOLVER_BASE(channels_first,
                             bottom_up_list,batch,
                             sizes[0],
                             n_c,
                             n_r,
                             data_cost,
                             init_u,
                             u),
n_x(sizes[0]),
rx(rx_cost),
rx_b(channels_first ? rx_cost : transpose(rx_cost, new float[n_s*n_r], n_s, n_r))
{}

HMF_MEANPASS_CPU_SOLVER_1D::~HMF_MEANPASS_CPU_SOLVER_1D(){
    if(!channels_first) delete rx_b;
}

int HMF_MEANPASS_CPU_GRADIENT_1D::min_iter_calc(){
    return n_x+n_r-n_c;
}

void HMF_MEANPASS_CPU_GRADIENT_1D::init_reg_info(){
    clear(g_rx,n_s*n_r);
}

void HMF_MEANPASS_CPU_GRADIENT_1D::clean_up(){
    //untranspose gradient, using rx_b as storage
    if( !channels_first){
        float* tmp_space = new float[n_s*n_r];
        for(int s = 0; s < n_s; s++)
            for(int r = 0; r < n_r; r++)
                tmp_space[s*n_r+r] = g_rx[r*n_s+s];
        copy(tmp_space,g_rx,n_s*n_r);
        delete tmp_space;
    }
}

void HMF_MEANPASS_CPU_GRADIENT_1D::get_reg_gradients_and_push(float tau){
    get_reg_gradients_channels_first(dy, u, g_rx, n_x, n_r, tau);
    clear(g_u,n_s*(n_r-n_c));
    get_gradient_for_u_channels_first(dy+n_s*(n_r-n_c), rx_b+n_s*(n_r-n_c), g_u+n_s*(n_r-n_c), n_x, n_c, tau);
}

HMF_MEANPASS_CPU_GRADIENT_1D::~HMF_MEANPASS_CPU_GRADIENT_1D(){
    if(!channels_first) delete rx_b;
}

HMF_MEANPASS_CPU_GRADIENT_1D::HMF_MEANPASS_CPU_GRADIENT_1D(
    const bool channels_first,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_c,
    const int n_r,
    const int sizes[1],
    const float* u,
    const float* g,
    const float* rx_cost,
    float* g_d,
    float* g_rx ) :
HMF_MEANPASS_CPU_GRADIENT_BASE(channels_first,
                             bottom_up_list,batch,
                             sizes[0],
                             n_c,
                             n_r,
                             u,
                             g,
                             g_d),
n_x(sizes[0]),
rx(rx_cost),
rx_b(channels_first ? rx_cost : transpose(rx_cost, new float[n_s*n_r], n_s, n_r)),
g_rx(g_rx)
{}
