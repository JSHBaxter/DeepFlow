#include "binary_auglag1d_cpu_solver.h"
#include "cpu_kernels.h"
#include <iostream>
#include <cmath>

int BINARY_AUGLAG_CPU_SOLVER_1D::min_iter_calc(){
    return (int) std::sqrt(n_x);
}
	
void BINARY_AUGLAG_CPU_SOLVER_1D::clear_spatial_flows(){
    px = new float[n_s*n_c];
    clear(px, n_s*n_c);
}

void BINARY_AUGLAG_CPU_SOLVER_1D::update_spatial_flow_calc(){
    if(channels_first)
        compute_flows_channels_first(g, div, px, rx, n_c, n_x);
    else
        compute_flows(g, div, px, rx, n_c, n_x);
}

void BINARY_AUGLAG_CPU_SOLVER_1D::clean_up(){
    if( px ) delete [] px; px = 0;
}

BINARY_AUGLAG_CPU_SOLVER_1D::BINARY_AUGLAG_CPU_SOLVER_1D(
    const bool channels_first,
    const int batch,
    const int n_c,
    const int sizes[1],
    const float * const data_cost,
    const float * const rx_cost,
    float* u 
):
BINARY_AUGLAG_CPU_SOLVER_BASE(channels_first,batch, sizes[0], n_c, data_cost, u),
n_x(sizes[0]),
rx(rx_cost),
px(0)
{ 
        //std::cout << "BINARY_AUGLAG_CPU_SOLVER_1D\t" << n_x << " " << n_c << " " << rx << std::endl;
}

#include "binary_auglag2d_cpu_solver.h"
#include "cpu_kernels.h"
#include <cmath>

int BINARY_AUGLAG_CPU_SOLVER_2D::min_iter_calc(){
	return (int) std::sqrt(n_x+n_y);
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
	if( px ) delete [] px; px = 0;
	if( py ) delete [] py; py = 0;
}

BINARY_AUGLAG_CPU_SOLVER_2D::BINARY_AUGLAG_CPU_SOLVER_2D(
    const bool channels_first,
	const int batch,
    const int n_c,
	const int sizes[2],
	const float * const data_cost,
	const float * const rx_cost,
	const float * const ry_cost,
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

#include "binary_auglag3d_cpu_solver.h"
#include "cpu_kernels.h"
#include <cmath> 

int BINARY_AUGLAG_CPU_SOLVER_3D::min_iter_calc(){
	return (int) std::sqrt(n_x+n_y+n_z);
}

void BINARY_AUGLAG_CPU_SOLVER_3D::clear_spatial_flows(){
	px = new float[n_s*n_c];
	py = new float[n_s*n_c];
	pz = new float[n_s*n_c];
	clear(px, py, pz, n_s*n_c);
}

void BINARY_AUGLAG_CPU_SOLVER_3D::update_spatial_flow_calc(){
    if(channels_first)
        compute_flows_channels_first(g, div, px, py, pz, rx, ry, rz, n_c, n_x, n_y, n_z);
    else
        compute_flows(g, div, px, py, pz, rx, ry, rz, n_c, n_x, n_y, n_z);
}

void BINARY_AUGLAG_CPU_SOLVER_3D::clean_up(){
	if( px ) delete [] px; px = 0;
	if( py ) delete [] py; py = 0;
	if( pz ) delete [] pz; pz = 0;
}

BINARY_AUGLAG_CPU_SOLVER_3D::BINARY_AUGLAG_CPU_SOLVER_3D(
    const bool channels_first,
	const int batch,
    const int n_c,
	const int sizes[3],
	const float * const data_cost,
	const float * const rx_cost,
	const float * const ry_cost,
	const float * const rz_cost,
	float* u 
):
BINARY_AUGLAG_CPU_SOLVER_BASE(channels_first,batch, sizes[0]*sizes[1]*sizes[2], n_c, data_cost, u),
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

#include "binary_auglag_cpu_solver.h"
#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "cpu_kernels.h"

BINARY_AUGLAG_CPU_SOLVER_BASE::BINARY_AUGLAG_CPU_SOLVER_BASE(
    const bool channels_first,
    const int batch,
    const int n_s,
    const int n_c,
    const float * const data_cost,
    float* u ) :
channels_first(channels_first),
b(batch),
n_c(n_c),
n_s(n_s),
data(data_cost),
ps(new float[4*n_s*n_c]),
pt(ps+n_s*n_c),
div(pt+n_s*n_c),
g(div+n_s*n_c),
u(u)
{
    //std::cout << "BINARY_AUGLAG_CPU_SOLVER_BASE\t" << n_s << " " << n_c << " " << data << " " << ps << " " << pt << " " << div << " " << g << " " << u << std::endl;
}

//perform one iteration of the algorithm
void BINARY_AUGLAG_CPU_SOLVER_BASE::block_iter(){
    
    //calculate the capacity and then update flows
    compute_capacity_binary(g, u, ps, pt, div, n_s*n_c, tau, icc);
    update_spatial_flow_calc();
                 
	//update source flows, sink flows, and multipliers
	compute_source_sink_multipliers_binary( g, u, ps, pt, div, data, cc, icc, n_s*n_c);
}

void BINARY_AUGLAG_CPU_SOLVER_BASE::operator()(){

    //initialize variables
	sigmoid(data, u, n_s*n_c);
    //clear(u, n_s*n_c);
	clear(g, div, n_c*n_s);
    clear_spatial_flows();
    clear(pt, n_c*n_s);
    clear(ps, n_c*n_s);
	init_flows_binary(data, ps, pt, n_s*n_c);

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = min_iter_calc();
    if (max_loop < 200)
        max_loop = 200;
    
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        float max_change = maxabs(g,n_s*n_c);
		//std::cout << "BINARY_AUGLAG_CPU_SOLVER_BASE Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();

    //log output and transpose output back into proper buffer
    log_buffer(u, n_s*n_c);
        
    //deallocate temporary buffers
    clean_up();
}

BINARY_AUGLAG_CPU_SOLVER_BASE::~BINARY_AUGLAG_CPU_SOLVER_BASE(){
    delete [] ps;
}

#include "binary_auglag1d_gpu_solver.h"
#include "gpu_kernels.h"

#include <iostream>
#include <cmath>

int BINARY_AUGLAG_GPU_SOLVER_1D::min_iter_calc(){
	return (int) std::sqrt(n_x);
}

void BINARY_AUGLAG_GPU_SOLVER_1D::clear_spatial_flows(){
	clear_buffer(dev, px, n_c*n_s);
}

void BINARY_AUGLAG_GPU_SOLVER_1D::update_spatial_flow_calc(){
	update_spatial_flows(dev, g, div, px, rx, n_x, n_s*n_c);
}

BINARY_AUGLAG_GPU_SOLVER_1D::BINARY_AUGLAG_GPU_SOLVER_1D(
	const cudaStream_t & dev,
	const int batch,
    const int n_c,
	const int sizes[1],
	const float * const data_cost,
	const float * const rx_cost,
	float* u,
	float** buffers_full,
	float** buffers_img
):
BINARY_AUGLAG_GPU_SOLVER_BASE(dev, batch, sizes[0], n_c, data_cost, u, buffers_full, buffers_img),
n_x(sizes[0]),
rx(rx_cost),
px(buffers_full[4])
{}


#include "binary_auglag2d_gpu_solver.h"
#include "gpu_kernels.h"
#include <cmath>

int BINARY_AUGLAG_GPU_SOLVER_2D::min_iter_calc(){
	return (int) std::sqrt(n_x+n_y);
}

void BINARY_AUGLAG_GPU_SOLVER_2D::clear_spatial_flows(){
	clear_buffer(dev, px, n_c*n_s);
	clear_buffer(dev, py, n_c*n_s);
}

void BINARY_AUGLAG_GPU_SOLVER_2D::update_spatial_flow_calc(){
	update_spatial_flows(dev, g, div, px, py, rx, ry, n_x, n_y, n_s*n_c);
}

BINARY_AUGLAG_GPU_SOLVER_2D::BINARY_AUGLAG_GPU_SOLVER_2D(
	const cudaStream_t & dev,
	const int batch,
    const int n_c,
	const int sizes[2],
	const float * const data_cost,
	const float * const rx_cost,
	const float * const ry_cost,
	float* u,
	float** buffers_full,
	float** buffers_img
):
BINARY_AUGLAG_GPU_SOLVER_BASE(dev, batch, sizes[0]*sizes[1], n_c, data_cost, u, buffers_full, buffers_img),
n_x(sizes[0]),
n_y(sizes[1]),
rx(rx_cost),
ry(ry_cost),
px(buffers_full[4]),
py(buffers_full[5])
{
	//std::cout << rx << std::endl;
	//std::cout << ry << std::endl;
	//std::cout << px << std::endl;
	//std::cout << py << std::endl;
}


#include "binary_auglag3d_gpu_solver.h"
#include "gpu_kernels.h"
#include <cmath>

int BINARY_AUGLAG_GPU_SOLVER_3D::min_iter_calc(){
	return (int) std::sqrt(n_x+n_y+n_z);
}

void BINARY_AUGLAG_GPU_SOLVER_3D::clear_spatial_flows(){
	clear_buffer(dev, px, n_c*n_s);
	clear_buffer(dev, py, n_c*n_s);
	clear_buffer(dev, pz, n_c*n_s);
}

void BINARY_AUGLAG_GPU_SOLVER_3D::update_spatial_flow_calc(){
	update_spatial_flows(dev, g, div, px, py, pz, rx, ry, rz, n_x, n_y, n_z, n_s*n_c);
}

BINARY_AUGLAG_GPU_SOLVER_3D::BINARY_AUGLAG_GPU_SOLVER_3D(
	const cudaStream_t & dev,
	const int batch,
    const int n_c,
	const int sizes[3],
	const float * const data_cost,
	const float * const rx_cost,
	const float * const ry_cost,
	const float * const rz_cost,
	float* u,
	float** buffers_full,
	float** buffers_img
):
BINARY_AUGLAG_GPU_SOLVER_BASE(dev, batch, sizes[0]*sizes[1]*sizes[2], n_c, data_cost, u, buffers_full, buffers_img),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx(rx_cost),
ry(ry_cost),
rz(rz_cost),
px(buffers_full[4]),
py(buffers_full[5]),
pz(buffers_full[6])
{}
#include "binary_auglag_gpu_solver.h"
#include "gpu_kernels.h"

#include <iostream>

BINARY_AUGLAG_GPU_SOLVER_BASE::BINARY_AUGLAG_GPU_SOLVER_BASE(
    const cudaStream_t & dev,
    const int batch,
    const int n_s,
    const int n_c,
    const float * const data_cost,
    float* u,
    float** full_buff,
    float** img_buff) :
dev(dev),
b(batch),
n_c(n_c),
n_s(n_s),
data(data_cost),
u(u),
pt(full_buff[0]),
div(full_buff[1]),
g(full_buff[2]),
ps(full_buff[3])
{
    //std::cout << "sizes " << n_s << " " << n_c << std::endl;
	//std::cout << "data_cost " << data_cost << std::endl;
	//std::cout << "u " << u << std::endl;
	//std::cout << "pt " << pt << std::endl;
	//std::cout << "div " << div << std::endl;
	//std::cout << "g " << g << std::endl;
	//std::cout << "ps " << ps << std::endl;
}

void BINARY_AUGLAG_GPU_SOLVER_BASE::block_iter(){
    
    //calculate the capacity and then update flows
    calc_capacity_binary(dev, g, div, ps, pt, u, n_s*n_c, icc, tau);
    update_spatial_flow_calc();
	
    //update source flows, sink flows, and multipliers
    update_source_sink_multiplier_binary(dev, ps, pt, div, u, g, data, cc, icc, n_s*n_c);
}

void BINARY_AUGLAG_GPU_SOLVER_BASE::operator()(){

    //initialize variables
	sigmoid(dev, data, 0, u, n_s*n_c);
    //clear_buffer(dev, u, n_s*n_c);
    clear_spatial_flows();
    clear_buffer(dev, div, n_s*n_c);
    clear_buffer(dev, g, n_s*n_c);
    clear_buffer(dev, ps, n_s*n_c);
    clear_buffer(dev, pt, n_s*n_c);
    init_flows_binary(dev, data, ps, pt, n_s*n_c);
    
    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = min_iter_calc();
    if (max_loop < 200)
        max_loop = 200;

    for(int i = 0; i < max_loop; i++){    
        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        //Determine if converged
        float max_change = max_of_buffer(dev, g, n_s*n_c);
		//std::cout << "BINARY_AUGLAG_GPU_SOLVER_BASE Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();

    //get final output
    log_buffer(dev, u, u, n_s*n_c);

}
#include <math.h>
#include <iostream>
#include <limits>

#include "binary_meanpass1d_cpu_solver.h"
#include "cpu_kernels.h"

int BINARY_MEANPASS_CPU_SOLVER_1D::min_iter_calc(){
	return (int) std::sqrt(n_x);
}

void BINARY_MEANPASS_CPU_SOLVER_1D::init_vars(){}

void BINARY_MEANPASS_CPU_SOLVER_1D::calculate_regularization(){
    if(channels_first)
	    calculate_r_eff_channels_first(r_eff, rx, u, n_x, n_c);
    else
	    calculate_r_eff(r_eff, rx, u, n_x, n_c);
}

void BINARY_MEANPASS_CPU_SOLVER_1D::parity_mask_buffer(float* buffer, const int parity){
    if(channels_first)
	    parity_mask_channels_first(buffer,n_x,n_c,parity);
    else
        parity_mask(buffer,n_x,n_c,parity);
}

void BINARY_MEANPASS_CPU_SOLVER_1D::parity_merge_buffer(float* buffer, const float * const other, const int parity){
    if(channels_first)
	    parity_merge_channels_first(buffer,other,n_x,n_c,parity);
    else
        parity_merge(buffer,other,n_x,n_c,parity);
}

void BINARY_MEANPASS_CPU_SOLVER_1D::clean_up(){}

BINARY_MEANPASS_CPU_SOLVER_1D::BINARY_MEANPASS_CPU_SOLVER_1D(
    const bool channels_first,
	const int batch,
    const int n_c,
	const int sizes[1],
	const float * const data_cost,
	const float * const rx_cost,
	const float * const init_u,
	float* u 
):
BINARY_MEANPASS_CPU_SOLVER_BASE(channels_first, batch, sizes[0], n_c, data_cost, init_u, u),
n_x(sizes[0]),
rx(rx_cost)
{
    //std::cout << "BINARY_MEANPASS_CPU_SOLVER_1D\t" << n_x << " " << n_c << " " << rx << " " << std::endl;
}

int BINARY_MEANPASS_CPU_GRADIENT_1D::min_iter_calc(){
	return (int) std::sqrt(n_x);
}

void BINARY_MEANPASS_CPU_GRADIENT_1D::init_vars(){
	clear(g_rx, n_c*n_s);
}

void BINARY_MEANPASS_CPU_GRADIENT_1D::get_reg_gradients_and_push(float tau){
    if(channels_first){
        get_reg_gradients_channels_first(d_y, u, g_rx, n_x, n_c, tau);
        get_gradient_for_u_channels_first(d_y, rx, g_u, n_x, n_c, tau);
    }else{
        get_reg_gradients(d_y, u, g_rx, n_x, n_c, tau);
        get_gradient_for_u(d_y, rx, g_u, n_x, n_c, tau);
    }
}

void BINARY_MEANPASS_CPU_GRADIENT_1D::clean_up(){}

BINARY_MEANPASS_CPU_GRADIENT_1D::BINARY_MEANPASS_CPU_GRADIENT_1D(
    const bool channels_first,
	const int batch,
    const int n_c,
	const int sizes[1],
	const float * const u,
	const float * const g,
	const float * const rx_cost,
	float* g_d,
	float* g_rx
) :
BINARY_MEANPASS_CPU_GRADIENT_BASE(channels_first, batch, sizes[0], n_c, u, g, g_d),
n_x(sizes[0]),
rx(rx_cost),
g_rx(g_rx)
{ 
    //std::cout << "BINARY_MEANPASS_CPU_GRADIENT_1D\t" << n_x << " " << n_c << " " << rx << " " << g_rx << std::endl;
}
#include <math.h>
#include <iostream>
#include <limits>
#include <algorithm>

#include "binary_meanpass2d_cpu_solver.h"
#include "cpu_kernels.h"

int BINARY_MEANPASS_CPU_SOLVER_2D::min_iter_calc(){
	return (int) std::sqrt(n_x+n_y);
}

void BINARY_MEANPASS_CPU_SOLVER_2D::init_vars(){}

void BINARY_MEANPASS_CPU_SOLVER_2D::calculate_regularization(){
    if(channels_first)
        calculate_r_eff_channels_first(r_eff, rx, ry, u, n_x, n_y, n_c);
    else
	    calculate_r_eff(r_eff, rx, ry, u, n_x, n_y, n_c);
}

void BINARY_MEANPASS_CPU_SOLVER_2D::parity_mask_buffer(float* buffer, const int parity){
    if(channels_first)
	    parity_mask_channels_first(buffer,n_x,n_y,n_c,parity);
    else
        parity_mask(buffer,n_x,n_y,n_c,parity);
}

void BINARY_MEANPASS_CPU_SOLVER_2D::parity_merge_buffer(float* buffer, const float * const other, const int parity){
    if(channels_first)
	    parity_merge_channels_first(buffer,other,n_x,n_y,n_c,parity);
    else
	    parity_merge(buffer,other,n_x,n_y,n_c,parity);
}

void BINARY_MEANPASS_CPU_SOLVER_2D::clean_up(){}

BINARY_MEANPASS_CPU_SOLVER_2D::BINARY_MEANPASS_CPU_SOLVER_2D(
    const bool channels_first,
	const int batch,
    const int n_c,
	const int sizes[2],
	const float * const data_cost,
	const float * const rx_cost,
	const float * const ry_cost,
	const float * const init_u,
	float* u 
):
BINARY_MEANPASS_CPU_SOLVER_BASE(channels_first, batch, sizes[0]*sizes[1], n_c, data_cost, init_u, u),
n_x(sizes[0]),
n_y(sizes[1]),
rx(rx_cost),
ry(ry_cost)
{}

int BINARY_MEANPASS_CPU_GRADIENT_2D::min_iter_calc(){
	return (int) std::sqrt(n_x+n_y);
}

void BINARY_MEANPASS_CPU_GRADIENT_2D::init_vars(){
	clear(g_rx, g_ry, n_c*n_s);
}

void BINARY_MEANPASS_CPU_GRADIENT_2D::get_reg_gradients_and_push(float tau){
    if(channels_first){
        get_reg_gradients_channels_first(d_y, u, g_rx, g_ry, n_x, n_y, n_c, tau);
        get_gradient_for_u_channels_first(d_y, rx, ry, g_u, n_x, n_y, n_c, tau);
    }else{
        get_reg_gradients(d_y, u, g_rx, g_ry, n_x, n_y, n_c, tau);
        get_gradient_for_u(d_y, rx, ry, g_u, n_x, n_y, n_c, tau);
    }
}

void BINARY_MEANPASS_CPU_GRADIENT_2D::clean_up(){}

BINARY_MEANPASS_CPU_GRADIENT_2D::BINARY_MEANPASS_CPU_GRADIENT_2D(
    const bool channels_first,
	const int batch,
    const int n_c,
	const int sizes[2],
	const float * const u,
	const float * const g,
	const float * const rx_cost,
	const float * const ry_cost,
	float* g_d,
	float* g_rx,
	float* g_ry
) :
BINARY_MEANPASS_CPU_GRADIENT_BASE(channels_first, batch, sizes[0]*sizes[1], n_c, u, g, g_d),
n_x(sizes[0]),
n_y(sizes[1]),
rx(rx_cost),
ry(ry_cost),
g_rx(g_rx),
g_ry(g_ry)
{}

#include <math.h>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>

#include "binary_meanpass3d_cpu_solver.h"
#include "cpu_kernels.h"

int BINARY_MEANPASS_CPU_SOLVER_3D::min_iter_calc(){
	return (int) std::sqrt(n_x+n_y+n_z);
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

void BINARY_MEANPASS_CPU_SOLVER_3D::parity_merge_buffer(float* buffer, const float * const other, const int parity){
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
	const float * const data_cost,
	const float * const rx_cost,
	const float * const ry_cost,
	const float * const rz_cost,
	const float * const init_u,
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
	return (int) std::sqrt(n_x+n_y+n_z);
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
	const float * const u,
	const float * const g,
	const float * const rx_cost,
	const float * const ry_cost,
	const float * const rz_cost,
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
#include "binary_meanpass_cpu_solver.h"
#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "cpu_kernels.h"

BINARY_MEANPASS_CPU_SOLVER_BASE::BINARY_MEANPASS_CPU_SOLVER_BASE(
    const float channels_first,
    const int batch,
    const int n_s,
    const int n_c,
    const float * const data_cost,
    const float * const init_u,
    float* u ) :
channels_first(channels_first),
b(batch),
n_c(n_c),
n_s(n_s),
data(data_cost),
r_eff(new float[n_s*n_c]),
u(u)
{
    //std::cout << "BINARY_MEANPASS_CPU_SOLVER_BASE\t" << n_s << " " << n_c << " " << data << " " << r_eff << " " << u << std::endl;

	if(init_u)
		copy(init_u, u, n_s*n_c);
	else
		sigmoid(data, u, n_s*n_c);
}

//perform one iteration of the algorithm
float BINARY_MEANPASS_CPU_SOLVER_BASE::block_iter(int parity, bool last){
	float max_change = 0.0f;
	calculate_regularization();
	inc(data, r_eff, n_s*n_c);
	sigmoid(r_eff, r_eff, n_s*n_c);
    parity_merge_buffer(r_eff, u, parity);
	if(last)
		max_change = update_with_convergence(u, r_eff, n_s*n_c, tau);
	else
		update(u, r_eff, n_s*n_c, tau);
	return max_change;
}

void BINARY_MEANPASS_CPU_SOLVER_BASE::operator()(){
        
	// allocate intermediate variables
	float max_change = 0.0f;
	float max_change_1 = 0.0f;
	float max_change_2 = 0.0f;

	//initialize variables
	init_vars();

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = min_iter_calc();
    if (max_loop < 200)
        max_loop = 200;
    
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++){
            max_change_1 = block_iter(0, iter == min_iter-1);
            max_change_2 = block_iter(1, iter == min_iter-1);
			max_change = (max_change_1 > max_change_2) ? max_change_1 : max_change_2;
		}

		//std::cout << "BINARY_MEANPASS_CPU_SOLVER_BASE Iter " << i << ": " << max_change_1 << " " << max_change_2 << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++){
		block_iter(0, false);
		block_iter(1, false);
	}

	//calculate the effective regularization
	calculate_regularization();
	
	//get final output
	for(int i = 0; i < n_s*n_c; i++)
		u[i] = data[i]+r_eff[i];
        
    //deallocate temporary buffers
    clean_up();
}

BINARY_MEANPASS_CPU_SOLVER_BASE::~BINARY_MEANPASS_CPU_SOLVER_BASE(){
    delete [] r_eff; r_eff = 0;
}


BINARY_MEANPASS_CPU_GRADIENT_BASE::BINARY_MEANPASS_CPU_GRADIENT_BASE(
    const float channels_first,
    const int batch,
    const int n_s,
    const int n_c,
    const float * const u,
    const float * const g,
    float* g_d ) :
channels_first(channels_first),
b(batch),
n_c(n_c),
n_s(n_s),
grad(g),
logits(u),
g_data(g_d),
d_y(new float[n_s*n_c]),
g_u(new float[n_s*n_c]),//d_y+n_s*n_c),
u(new float[n_s*n_c])//g_u+n_s*n_c)
{
    //std::cout << "BINARY_MEANPASS_CPU_GRADIENT_BASE\t" << n_s << " " << n_c << " " << grad << " " << logits << " " << g_data << " " << d_y << " " << g_u << " " << this->u << std::endl;
}

//perform one iteration of the algorithm
void BINARY_MEANPASS_CPU_GRADIENT_BASE::block_iter(){
	//untangle sigmoid derivative
	untangle_sigmoid(g_u, u, d_y, n_s*n_c);
	
	// populate data gradient
	inc(d_y, g_data, tau, n_s*n_c);
	
	//add to regularization gradients and push back
	get_reg_gradients_and_push(tau);
}

void BINARY_MEANPASS_CPU_GRADIENT_BASE::operator()(){

	//initialize variables
	init_vars();
	sigmoid(logits, u, n_s*n_c);
	clear(g_u, n_s*n_c);
	copy(grad,d_y,n_s*n_c);
	
	//get initial gradient for the data and regularization terms
	copy(grad,g_data,n_s*n_c);
	get_reg_gradients_and_push(1.0f);

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = min_iter_calc();
    if (max_loop < 200)
        max_loop = 200;
    
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

		float max_change = maxabs(g_u,n_s*n_c);
		//std::cout << "BINARY_MEANPASS_CPU_GRADIENT_BASE Iter " << i << ": " << max_change << " " << beta << std::endl;
        if (max_change < beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();
    
    clean_up();
}

BINARY_MEANPASS_CPU_GRADIENT_BASE::~BINARY_MEANPASS_CPU_GRADIENT_BASE(){
    delete [] d_y; d_y = 0;
    delete [] g_u; g_u = 0;
    delete [] u; u = 0;
}

#include "binary_meanpass1d_gpu_solver.h"
#include "gpu_kernels.h"
#include <cmath> 

int BINARY_MEANPASS_GPU_SOLVER_1D::min_iter_calc(){
	return (int) std::sqrt(n_x);
}

void BINARY_MEANPASS_GPU_SOLVER_1D::init_vars(){}

void BINARY_MEANPASS_GPU_SOLVER_1D::calculate_regularization(){
	get_effective_reg(dev, r_eff, u, rx, n_x, n_c);
}

void BINARY_MEANPASS_GPU_SOLVER_1D::parity_mask_buffer(float* buffer, const int parity){
	parity_mask(dev,buffer,n_x,n_c,parity);
}

void BINARY_MEANPASS_GPU_SOLVER_1D::parity_merge_buffer(float* buffer, const float * const other, const int parity){
	parity_mask(dev,buffer,other,n_x,n_c,parity);
}

void BINARY_MEANPASS_GPU_SOLVER_1D::clean_up(){}

BINARY_MEANPASS_GPU_SOLVER_1D::BINARY_MEANPASS_GPU_SOLVER_1D(
	const cudaStream_t & dev,
	const int batch,
    const int n_c,
	const int sizes[1],
	const float * const data_cost,
	const float * const rx_cost,
	const float * const init_u,
	float* u,
	float** buffers_full
):
BINARY_MEANPASS_GPU_SOLVER_BASE(dev, batch, sizes[0], n_c, data_cost, init_u, u, buffers_full),
n_x(sizes[0]),
rx(rx_cost)
{}

int BINARY_MEANPASS_GPU_GRADIENT_1D::min_iter_calc(){
	return (int) std::sqrt(n_x);
}

void BINARY_MEANPASS_GPU_GRADIENT_1D::init_vars(){
	clear_buffer(dev, g_rx, n_c*n_s);
}

void BINARY_MEANPASS_GPU_GRADIENT_1D::get_reg_gradients_and_push(float tau){
	populate_reg_mean_gradients_and_add(dev, d_y, u, g_rx, n_x, n_c, tau);
	get_gradient_for_u(dev, d_y, g_u, rx, n_x, n_c, tau);
}

void BINARY_MEANPASS_GPU_GRADIENT_1D::clean_up(){}

BINARY_MEANPASS_GPU_GRADIENT_1D::BINARY_MEANPASS_GPU_GRADIENT_1D(
	const cudaStream_t & dev,
	const int batch,
    const int n_c,
	const int sizes[1],
	const float * const u,
	const float * const g,
	const float * const rx_cost,
	float* g_d,
	float* g_rx,
	float** full_buffs
) :
BINARY_MEANPASS_GPU_GRADIENT_BASE(dev, batch, sizes[0], n_c, u, g, g_d, full_buffs),
n_x(sizes[0]),
rx(rx_cost),
g_rx(g_rx)
{}

#include "binary_meanpass2d_gpu_solver.h"
#include "gpu_kernels.h"
#include <iostream>
#include <algorithm>
#include <cmath>

int BINARY_MEANPASS_GPU_SOLVER_2D::min_iter_calc(){
	return (int) std::sqrt(n_x+n_y);
}

void BINARY_MEANPASS_GPU_SOLVER_2D::init_vars(){}

void BINARY_MEANPASS_GPU_SOLVER_2D::calculate_regularization(){
	get_effective_reg(dev, r_eff, u, rx, ry, n_x, n_y, n_c);
}

void BINARY_MEANPASS_GPU_SOLVER_2D::parity_mask_buffer(float* buffer, const int parity){
	parity_mask(dev,buffer,n_x,n_y,n_c,parity);
}

void BINARY_MEANPASS_GPU_SOLVER_2D::parity_merge_buffer(float* buffer, const float * const other, const int parity){
	parity_mask(dev,buffer,other,n_x,n_y,n_c,parity);
}

void BINARY_MEANPASS_GPU_SOLVER_2D::clean_up(){}

BINARY_MEANPASS_GPU_SOLVER_2D::BINARY_MEANPASS_GPU_SOLVER_2D(
	const cudaStream_t & dev,
	const int batch,
    const int n_c,
	const int sizes[2],
	const float * const data_cost,
	const float * const rx_cost,
	const float * const ry_cost,
	const float * const init_u,
	float* u,
	float** buffers_full
):
BINARY_MEANPASS_GPU_SOLVER_BASE(dev, batch, sizes[0]*sizes[1], n_c, data_cost, init_u, u, buffers_full),
n_x(sizes[0]),
n_y(sizes[1]),
rx(rx_cost),
ry(ry_cost)
{
}


int BINARY_MEANPASS_GPU_GRADIENT_2D::min_iter_calc(){
	return (int) std::sqrt(n_x+n_y);
}

void BINARY_MEANPASS_GPU_GRADIENT_2D::init_vars(){
	clear_buffer(dev, g_rx, n_c*n_s);
	clear_buffer(dev, g_ry, n_c*n_s);
}

void BINARY_MEANPASS_GPU_GRADIENT_2D::get_reg_gradients_and_push(float tau){
	populate_reg_mean_gradients_and_add(dev, d_y, u, g_rx, g_ry, n_x, n_y, n_c, tau);
	get_gradient_for_u(dev, d_y, g_u, rx, ry, n_x, n_y, n_c, tau);
}

void BINARY_MEANPASS_GPU_GRADIENT_2D::clean_up(){}

BINARY_MEANPASS_GPU_GRADIENT_2D::BINARY_MEANPASS_GPU_GRADIENT_2D(
	const cudaStream_t & dev,
	const int batch,
    const int n_c,
	const int sizes[2],
	const float * const u,
	const float * const g,
	const float * const rx_cost,
	const float * const ry_cost,
	float* g_d,
	float* g_rx,
	float* g_ry,
	float** full_buffs
) :
BINARY_MEANPASS_GPU_GRADIENT_BASE(dev, batch, sizes[0]*sizes[1], n_c, u, g, g_d, full_buffs),
n_x(sizes[0]),
n_y(sizes[1]),
rx(rx_cost),
ry(ry_cost),
g_rx(g_rx),
g_ry(g_ry)
{}

#include "binary_meanpass3d_gpu_solver.h"
#include "gpu_kernels.h"
#include <algorithm>
#include <cmath>

int BINARY_MEANPASS_GPU_SOLVER_3D::min_iter_calc(){
	return (int) std::sqrt(n_x+n_y+n_z);
}

void BINARY_MEANPASS_GPU_SOLVER_3D::init_vars(){}

void BINARY_MEANPASS_GPU_SOLVER_3D::calculate_regularization(){
	get_effective_reg(dev, r_eff, u, rx, ry, rz, n_x, n_y, n_z, n_c);
}

void BINARY_MEANPASS_GPU_SOLVER_3D::parity_mask_buffer(float* buffer, const int parity){
	parity_mask(dev,buffer,n_x,n_y,n_z,n_c,parity);
}

void BINARY_MEANPASS_GPU_SOLVER_3D::parity_merge_buffer(float* buffer, const float * const other, const int parity){
	parity_mask(dev,buffer,other,n_x,n_y,n_z,n_c,parity);
}

void BINARY_MEANPASS_GPU_SOLVER_3D::clean_up(){}

BINARY_MEANPASS_GPU_SOLVER_3D::BINARY_MEANPASS_GPU_SOLVER_3D(
	const cudaStream_t & dev,
	const int batch,
    const int n_c,
    const int sizes[3],
	const float * const data_cost,
	const float * const rx_cost,
	const float * const ry_cost,
	const float * const rz_cost,
	const float * const init_u,
	float* u,
	float** buffers_full
):
BINARY_MEANPASS_GPU_SOLVER_BASE(dev, batch, sizes[0]*sizes[1]*sizes[2], n_c, data_cost, init_u, u, buffers_full),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx(rx_cost),
ry(ry_cost),
rz(rz_cost)
{}

int BINARY_MEANPASS_GPU_GRADIENT_3D::min_iter_calc(){
	return (int) std::sqrt(n_x+n_y+n_z);
}

void BINARY_MEANPASS_GPU_GRADIENT_3D::init_vars(){
	clear_buffer(dev, g_rx, n_c*n_s);
	clear_buffer(dev, g_ry, n_c*n_s);
	clear_buffer(dev, g_rz, n_c*n_s);
}

void BINARY_MEANPASS_GPU_GRADIENT_3D::get_reg_gradients_and_push(float tau){
	populate_reg_mean_gradients_and_add(dev, d_y, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_c, tau);
	get_gradient_for_u(dev, d_y, g_u, rx, ry, rz, n_x, n_y, n_z, n_c, tau);
}

void BINARY_MEANPASS_GPU_GRADIENT_3D::clean_up(){}

BINARY_MEANPASS_GPU_GRADIENT_3D::BINARY_MEANPASS_GPU_GRADIENT_3D(
	const cudaStream_t & dev,
	const int batch,
    const int n_c,
    const int sizes[3],
	const float * const u,
	const float * const g,
	const float * const rx_cost,
	const float * const ry_cost,
	const float * const rz_cost,
	float* g_d,
	float* g_rx,
	float* g_ry,
	float* g_rz,
	float** full_buffs
) :
BINARY_MEANPASS_GPU_GRADIENT_BASE(dev, batch, sizes[0]*sizes[1]*sizes[2], n_c, u, g, g_d, full_buffs),
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


#include "binary_meanpass_gpu_solver.h"
#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "gpu_kernels.h"

BINARY_MEANPASS_GPU_SOLVER_BASE::BINARY_MEANPASS_GPU_SOLVER_BASE(
	const cudaStream_t & dev,
    const int batch,
    const int n_s,
    const int n_c,
    const float * const data_cost,
    const float * const init_u,
    float* u,
    float** full_buffs) :
dev(dev),
b(batch),
n_c(n_c),
n_s(n_s),
data(data_cost),
r_eff(full_buffs[0]),
u(u)
{
    //std::cout << n_s << " " << n_c << std::endl;
	if(init_u)
		copy_buffer(dev, init_u, u, n_s*n_c);
	else
		sigmoid(dev, data, 0, u, n_s*n_c);
}

//perform one iteration of the algorithm
void BINARY_MEANPASS_GPU_SOLVER_BASE::block_iter(int parity){
	calculate_regularization();
	sigmoid(dev, data, r_eff, r_eff, n_s*n_c);
    parity_merge_buffer(r_eff, u, parity);
	change_to_diff(dev, u, r_eff, n_s*n_c, tau);
}

void BINARY_MEANPASS_GPU_SOLVER_BASE::operator()(){

	//initialize variables
	init_vars();

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = min_iter_calc();
    if (max_loop < 200)
        max_loop = 200;
    
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter-1; iter++){
            block_iter(0);
            block_iter(1);
		}
		block_iter(0);
		float max_change_1 = max_of_buffer(dev, r_eff, n_c*n_s);
		block_iter(1);
		float max_change_2 = max_of_buffer(dev, r_eff, n_c*n_s);
		float max_change = (max_change_1 > max_change_2) ? max_change_1 : max_change_2;
		
		//std::cout << "BINARY_MEANPASS_GPU_SOLVER_BASE Iter " << i << ": " << max_change_1 << " " << max_change_2 << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
	for (int iter = 0; iter < min_iter; iter++){
		block_iter(0);
		block_iter(1);
	}

	//calculate the effective regularization
	calculate_regularization();
	
	//get final output
	add_then_store(dev, data, r_eff, u, n_s*n_c);
        
    //deallocate temporary buffers
    clean_up();
}

BINARY_MEANPASS_GPU_SOLVER_BASE::~BINARY_MEANPASS_GPU_SOLVER_BASE(){
}


BINARY_MEANPASS_GPU_GRADIENT_BASE::BINARY_MEANPASS_GPU_GRADIENT_BASE(
    const cudaStream_t & dev,
    const int batch,
    const int n_s,
    const int n_c,
    const float * const u,
    const float * const g,
    float* g_d,
    float** full_buffs) :
dev(dev),
b(batch),
n_c(n_c),
n_s(n_s),
grad(g),
logits(u),
g_data(g_d),
d_y(full_buffs[0]),
g_u(full_buffs[1]),
u(full_buffs[2])
{
    //std::cout << n_s << " " << n_c << std::endl;
}

//perform one iteration of the algorithm
void BINARY_MEANPASS_GPU_GRADIENT_BASE::block_iter(){
	//untangle softmax derivative
	untangle_sigmoid(dev, g_u, u, d_y, n_s*n_c);
	
	//add to data term gradient
	inc_mult_buffer(dev, d_y, g_data, n_s*n_c, tau);
	
	//add to regularization gradients
	get_reg_gradients_and_push(tau);
}

void BINARY_MEANPASS_GPU_GRADIENT_BASE::operator()(){

	//initialize variables
	init_vars();
	sigmoid(dev, logits, 0, u, n_s*n_c);
	clear_buffer(dev, g_u, n_s*n_c);
	copy_buffer(dev, grad, d_y, n_s*n_c);
	
	//get initial gradient for the data and regularization terms
	copy_buffer(dev, grad, g_data, n_s*n_c);
	get_reg_gradients_and_push(1.0f);

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = min_iter_calc();
    if (max_loop < 200)
        max_loop = 200;
    
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

		float max_change = max_of_buffer(dev, g_u, n_s*n_c);
		//std::cout << "BINARY_MEANPASS_GPU_GRADIENT_BASE Iter " << i << ": " << max_change << " " << beta << std::endl;
        if (max_change < beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();
        
    //deallocate temporary buffers
    clean_up();
}

BINARY_MEANPASS_GPU_GRADIENT_BASE::~BINARY_MEANPASS_GPU_GRADIENT_BASE(){
}
