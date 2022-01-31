
#include <iostream>
#include <limits>
#include "cpu_kernels.h"
#include "hmf_trees.h"
#include "hmf_meanpass3d_cpu_solver.h"
#include <algorithm>

int HMF_MEANPASS_CPU_SOLVER_3D::min_iter_calc(){
    return std::max(n_x,std::max(n_y,n_z))+n_r-n_c;
}

void HMF_MEANPASS_CPU_SOLVER_3D::init_reg_info(){
}

void HMF_MEANPASS_CPU_SOLVER_3D::clean_up(){
}

void HMF_MEANPASS_CPU_SOLVER_3D::update_spatial_flow_calc(){
    calculate_r_eff_channels_first(r_eff, rx_b, ry_b, rz_b, u_tmp, n_x, n_y, n_z, n_r);
}

void HMF_MEANPASS_CPU_SOLVER_3D::parity_mask_buffer(float* buffer, const int parity){
    parity_mask_channels_first(buffer,n_x,n_y,n_z,n_c,parity);
}

void HMF_MEANPASS_CPU_SOLVER_3D::parity_merge_buffer(float* buffer, const float* other, const int parity){
    parity_merge_channels_first(buffer,other,n_x,n_y,n_z,n_c,parity);
}

HMF_MEANPASS_CPU_SOLVER_3D::~HMF_MEANPASS_CPU_SOLVER_3D(){
    if( !channels_first ) delete rx_b;
}

HMF_MEANPASS_CPU_SOLVER_3D::HMF_MEANPASS_CPU_SOLVER_3D(
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
		const float* init_u,
        float* u ):
HMF_MEANPASS_CPU_SOLVER_BASE(channels_first,
                             bottom_up_list,batch,
                             sizes[0]*sizes[1]*sizes[2],
                             n_c,
                             n_r,
                             data_cost,
                             init_u,
                             u),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx(rx_cost),
ry(ry_cost),
rz(rz_cost),
alloc(channels_first ? 0 : new float[3*n_s*n_r]),
rx_b(channels_first ? rx : transpose(rx, alloc, n_s, n_r)),
ry_b(channels_first ? ry : transpose(ry, alloc+n_s*n_r, n_s, n_r)),
rz_b(channels_first ? rz : transpose(rz, alloc+2*n_s*n_r, n_s, n_r))
{
    alloc = 0;
}

int HMF_MEANPASS_CPU_GRADIENT_3D::min_iter_calc(){
    return std::max(n_x,std::max(n_y,n_z))+n_r-n_c;
}

void HMF_MEANPASS_CPU_GRADIENT_3D::init_reg_info(){
    clear(g_rx,n_s*n_r);
    clear(g_ry,n_s*n_r);
    clear(g_rz,n_s*n_r);
}

void HMF_MEANPASS_CPU_GRADIENT_3D::clean_up(){
    if( !channels_first ){
        //untranspose gradient, using temp as storage
        float* temp = new float[n_s*n_r];
        for(int s = 0; s < n_s; s++)
            for(int r = 0; r < n_r; r++)
                temp[s*n_r+r] = g_rx[r*n_s+s];
        copy(temp,g_rx,n_s*n_r);

        //untranspose gradient, using temp as storage
        for(int s = 0; s < n_s; s++)
            for(int r = 0; r < n_r; r++)
                temp[s*n_r+r] = g_ry[r*n_s+s];
        copy(temp,g_ry,n_s*n_r);

        //untranspose gradient, using temp as storage
        for(int s = 0; s < n_s; s++)
            for(int r = 0; r < n_r; r++)
                temp[s*n_r+r] = g_rz[r*n_s+s];
        copy(temp,g_rz,n_s*n_r);
        delete temp;
    }
}

void HMF_MEANPASS_CPU_GRADIENT_3D::get_reg_gradients_and_push(float tau){
    get_reg_gradients_channels_first(dy, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_r, tau);
    clear(g_u,n_s*(n_r-n_c));
    get_gradient_for_u_channels_first(dy+n_s*(n_r-n_c), rx+n_s*(n_r-n_c), ry+n_s*(n_r-n_c), rz+n_s*(n_r-n_c), g_u+n_s*(n_r-n_c), n_x, n_y, n_z, n_c, tau);
}

HMF_MEANPASS_CPU_GRADIENT_3D::~HMF_MEANPASS_CPU_GRADIENT_3D(){
    if( !channels_first ) delete rx_b;
}

HMF_MEANPASS_CPU_GRADIENT_3D::HMF_MEANPASS_CPU_GRADIENT_3D(
    const bool channels_first,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_c,
    const int n_r,
    const int sizes[3],
    const float* u,
    const float* g,
    const float* rx_cost,
    const float* ry_cost,
    const float* rz_cost,
    float* g_d,
    float* g_rx,
    float* g_ry,
    float* g_rz ) :
HMF_MEANPASS_CPU_GRADIENT_BASE(channels_first,
                             bottom_up_list,batch,
                             sizes[0]*sizes[1]*sizes[2],
                             n_c,
                             n_r,
                             u,
                             g,
                             g_d),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx(rx_cost),
ry(ry_cost),
rz(rz_cost),
alloc(channels_first ? 0 : new float[3*n_s*n_r]),
rx_b(channels_first ? rx : transpose(rx, alloc, n_s, n_r)),
ry_b(channels_first ? ry : transpose(ry, alloc+n_s*n_r, n_s, n_r)),
rz_b(channels_first ? rz : transpose(rz, alloc+2*n_s*n_r, n_s, n_r)),
g_rx(g_rx),
g_ry(g_ry),
g_rz(g_rz)
{
    alloc = 0;
}
