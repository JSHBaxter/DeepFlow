#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda.h>
#include <cublas_v2.h>
#include <stdio.h>

#include "gpu_kernels.h"

#define NUM_THREADS 256
#define epsilon 0.0001f

#define CHECK_ERRORS false

void get_from_gpu(const Eigen::GpuDevice& dev, const void* source, void* dest, size_t amount){
    cudaStreamSynchronize(dev.stream());
    cudaMemcpy(dest,source,amount,cudaMemcpyDeviceToHost);
}

void check_error(const Eigen::GpuDevice& dev, const char* string){
    cudaError_t cudaerr = cudaStreamSynchronize(dev.stream());
    if (cudaerr != cudaSuccess){
        printf(string);
        printf(" \"%s\".\n", cudaGetErrorString(cudaerr));
    }
}

// Sets variables to 0.0f

void clear_buffer(const Eigen::GpuDevice& dev, float* buffer, const int size){
    cudaMemsetAsync(buffer, 0.0f, size*sizeof(float),dev.stream());
}
void clear_buffer(const Eigen::GpuDevice& dev, int* buffer, const int size){
    cudaMemsetAsync(buffer, 0, size*sizeof(int),dev.stream());
}

// update the source flow
__global__ void update_source_sink_multiplier_potts_kernel(float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_c, const int n_s){
    
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    //update source flow
    float ps_t = icc;
    for(int c = 0; c < n_c; c++)
        ps_t += pt[c*n_s+i]+div[c*n_s+i]-u[c*n_s+i]*icc;
    ps_t /= n_c;
    if( i < n_s )
        ps[i] = ps_t;
    
    //update sink flow
    for(int c = 0; c < n_c; c++){
        float pt_t = ps_t;
        pt_t -= div[c*n_s+i];
        pt_t += u[c*n_s+i]*icc;
        float d_t = -d[c*n_s+i];
        pt_t = (pt_t > d_t) ? d_t : pt_t;
    
        
        //update multiplier
        float erru_t = cc * (ps_t - div[c*n_s+i] - pt_t);
        float u_t = u[c*n_s+i] + erru_t;
        erru_t = (erru_t < 0.0f) ? -erru_t : erru_t;
        
        //output
        if( i < n_s ){
            pt[c*n_s+i] = pt_t;
            u[c*n_s+i] = u_t;
            erru[c*n_s+i] = erru_t;
        }
    }
    
}

void update_source_sink_multiplier_potts(const Eigen::GpuDevice& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_c, const int n_s){
    update_source_sink_multiplier_potts_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(ps, pt, div, u, erru, d, cc, icc, n_c, n_s);
    if(CHECK_ERRORS) check_error(dev, "update_source_sink_multiplier_potts launch failed with error");
}

__global__ void update_source_flows_kernel_channel_first(float* ps, const float* pt, const float* div, const float* u, float icc, const int n_c, const int n_s) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float ps_t = icc;
    for(int c = 0; c < n_c; c++)
        ps_t += pt[c*n_s+i]+div[c*n_s+i]-u[c*n_s+i]*icc;
    ps_t /= (float) n_c;
    if( i < n_s )
        ps[i] = ps_t;
    
}



void update_source_flows(const Eigen::GpuDevice& dev, float* ps, const float* pt, const float* div, const float* u, float icc, const int n_c, const int n_s){
    update_source_flows_kernel_channel_first<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(ps, pt, div, u, icc, n_c, n_s);
    if(CHECK_ERRORS) check_error(dev, "update_source_flows launch failed with error");
}



__global__ void update_sink_flows_kernel(const float* ps, float* pt, const float* div, const float* u, const float* d, float icc, const int b_size, const int n_s) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int ps_i = i % n_s;
    
    float pt_t = ps[ps_i];
    pt_t -= div[i];
    pt_t += icc * u[i];
    float constraint = -d[i];
    pt_t = (pt_t > constraint) ? constraint: pt_t;
    if( i < b_size )
        pt[i] = pt_t;
    
}


void update_sink_flows(const Eigen::GpuDevice& dev, const float* ps, float* pt, const float* div, const float* u, const float* d, float icc, const int n_c, const int n_s){
    update_sink_flows_kernel<<<((n_s*n_c+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(ps, pt, div, u, d, icc, n_c*n_s, n_s );
    if(CHECK_ERRORS) check_error(dev, "update_sink_flows launch failed with error");
}



__global__ void update_multiplier_kernel(const float* ps, const float* pt, const float* div, float* u, float* erru, float cc, const int b_size, const int n_s) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int ps_i = i % n_s;
    
    float erru_t = ps[ps_i];
    erru_t -= div[i];
    erru_t -= pt[i];
    erru_t *= cc;
    float u_t = u[i] + erru_t;
    erru_t = (erru_t < 0.0f) ? -erru_t: erru_t;
    if( i < b_size ){
        u[i] = u_t;
        erru[i] = erru_t;
    }
    
}


void update_multiplier(const Eigen::GpuDevice& dev, const float* ps, const float* pt, const float* div, float* u, float* erru, float cc, const int n_c, const int n_s){
    update_multiplier_kernel<<<((n_s*n_c+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(ps, pt, div, u, erru, cc, n_c*n_s, n_s );
    if(CHECK_ERRORS) check_error(dev, "update_multiplier launch failed with error");
}

//find the maximum in the neg_constraint buffer and then multiply it by -1

__global__ void find_min_constraint_channel_last(float* out, const float* d, const int n_c, const int n_s) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float out_t = d[i*n_c];
    for(int c = 1; c < n_c; c++){
        float contender = d[i*n_c+c];
        out_t = (out_t > contender) ? out_t : contender;
    }
    out_t *= -1.0f;
    if( i < n_s )
        out[i] = out_t;
}


__global__ void find_min_constraint_channel_first(float* out, const float* d, const int n_c, const int n_s) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float out_t = d[i];
    for(int c = 1; c < n_c; c++){
        float contender = d[i+c*n_s];
        out_t = (out_t > contender) ? out_t : contender;
    }
    out_t *= -1.0f;
    if( i < n_s )
        out[i] = out_t;
}


void find_min_constraint(const Eigen::GpuDevice& dev, float* output, const float* neg_constraint, const int n_c, const int n_s){
    find_min_constraint_channel_last<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(output, neg_constraint, n_c, n_s );
    if(CHECK_ERRORS) check_error(dev, "find_min_constraint launch failed with error");
}

//find the maximum in the neg_constraint buffer and then multiply it by -1

__global__ void rep_buffer_channel_last(const float* in, float* out, const int n_c, const int n_s) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float out_t = in[i];
    for(int c = 1; c < n_c; c++)
        if( i < n_s )
            out[i*n_c+c] = out_t;
}

//channel first is a series of memcpy's

void rep_buffer(const Eigen::GpuDevice& dev, const float* input, float* output, const int n_c, const int n_s){
    for(int c = 0; c < n_c; c++)
        cudaMemcpyAsync((void*)(output+c*n_s),input,n_s*sizeof(float),cudaMemcpyDeviceToDevice,dev.stream());
    if(CHECK_ERRORS) check_error(dev, "rep_buffer launch failed with error");
}



__global__ void calc_divergence_kernel(float* div, const float* px, const float* py, const float* pz, const int n_x, const int n_y, const int n_z, const int b_size) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    
    //do z dimension (add just one)
    float div_t = -pz[i];
    float more_flow = pz[i+1];
    div_t += (z < n_z-1) ? more_flow : 0.0f;
    
    //do y dimension (add n_z)
    div_t -= py[i];
    more_flow = py[i+n_z];
    div_t += (y < n_y-1) ? more_flow : 0.0f;
    
    //do x dimension (add n_z*n_y)
    div_t -= px[i];
    more_flow = px[i+n_z*n_y];
    div_t += (x < n_x-1) ? more_flow : 0.0f;
    
    if(i < b_size)
        div[i] = div_t;
}


void calc_divergence(const Eigen::GpuDevice& dev, float* div, const float* px, const float* py, const float* pz, const int n_x, const int n_y, const int n_z, const int n_c){
    int b_size = n_x*n_y*n_z*n_c;
    calc_divergence_kernel<<<((b_size+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(div, px, py, pz, n_x, n_y, n_z, b_size);
    if(CHECK_ERRORS) check_error(dev, "calc_divergence launch failed with error");
}





__global__ void calc_capacity_kernel(float* g, const float* div, const float* ps, const float* pt, const float* u, const int i_size, const int b_size, float icc, float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int ps_i = i % i_size;
    float g_t = div[i] + pt[i] - icc * u[i];
    g_t -= ps[ps_i];
    g_t *= tau;
    if(i < b_size)
        g[i] = g_t;
}



__global__ void update_flows_kernel(const float* g, float* px, float* py, float* pz, const int n_x, const int n_y, const int n_z, const int b_size){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    
    //for z
    float capacity = g[i]-g[i-1];
    float newflow = pz[i] + (z > 0) ? capacity : 0.0f;
    if(i < b_size)
        pz[i] = newflow;
    
    //for y
    capacity = g[i]-g[i-n_y];
    newflow = py[i] + (y > 0) ? capacity : 0.0f;
    if(i < b_size)
        py[i] = newflow;
    
    //for x
    capacity = g[i]-g[i-n_y*n_z];
    newflow = px[i] + (x > 0) ? capacity : 0.0f;
    if(i < b_size)
        px[i] = newflow;
}


void update_spatial_flows(const Eigen::GpuDevice& dev, float* g, const float* div, float* px, float* py, float* pz, const float* ps, const float* pt, const float* u, const int n_x, const int n_y, const int n_z, const int n_c, float icc, float tau){
    int i_size = n_x*n_y*n_z;
    int b_size = i_size*n_c;
    calc_capacity_kernel<<<((b_size+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(g, div, ps, pt, u, i_size, b_size, icc, tau);
    update_flows_kernel<<<((b_size+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(div, px, py, pz, n_x, n_y, n_z, n_c);
    if(CHECK_ERRORS) check_error(dev, "update_spatial_flows launch failed with error");
    
}




__global__ void abs_constrain_kernel(float* b, const float* r, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float value = b[i];
    float constraint = r[i];
    value = (value > constraint) ? constraint: value;
    value = (value < -constraint) ? -constraint: value;
    
    if(i < n_s)
        b[i] = value;
    
}


void abs_constrain(const Eigen::GpuDevice& dev, float* buffer, const float* constrain, const int n_s){
    abs_constrain_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(buffer, constrain, n_s);
    if(CHECK_ERRORS) check_error(dev, "abs_constrain launch failed with error");
    
}

__global__ void log_buffer_kernel(const float* in, float* out, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float value = in[i];
    value = log(value + 0.00001f);
    
    if(i < n_s)
        out[i] = value;
    
}

void log_buffer(const Eigen::GpuDevice& dev, const float* in, float* out, const int n_s){
    log_buffer_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(in, out, n_s);
    if(CHECK_ERRORS) check_error(dev, "log_buffer launch failed with error");
    
}

__global__ void binary_constrain_kernel(float* b, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float value = b[i];
    value = (value > 1.0f) ? 1.0f: value;
    value = (value < 0.0f) ? -0.0f: value;
    
    if(i < n_s)
        b[i] = value;
    
}


void binary_constrain(const Eigen::GpuDevice& dev, float* buffer, const int n_s){
    binary_constrain_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(buffer, n_s);
    if(CHECK_ERRORS) check_error(dev, "binary_constrain launch failed with error");
    
}

__global__ void abs_buffer_kernel(float* buffer, int n) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = buffer[i];
    val = (val > 0.0f) ? val: -val;
    if(i < n)
        buffer[i] = val;
}


__global__ void maxreduce(float* buffer, int j, int n) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val1 = buffer[i];
    float val2 = buffer[i+j];
    float val = (val1 > val2) ? val1: val2;
    if(i + j < n)
        buffer[i] = val;
}


float max_of_buffer(const Eigen::GpuDevice& dev, float* buffer, const int n_s){
    abs_buffer_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(buffer, n_s);
    int n = n_s;
    while(n > 1){
        int j = (n+1)/2;
        maxreduce<<<((n-j+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(buffer, j, n);
        n = j;
    }
    float max_value = 0.0f;
    cudaStreamSynchronize(dev.stream());
    cudaMemcpy(&max_value,(float*)buffer,sizeof(float),cudaMemcpyDeviceToHost);
    if( max_value < 0.0f)
        max_value = -max_value;
    return max_value;
}


__global__ void populate_data_gradient_kernel(const float* g, const float* u, float* output, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float g_t = g[i];
    float u_t = u[i];
    bool not_stigma = ( ((u_t >= 1.0f - epsilon) && (g_t < 0.0f)) || ((u_t <= epsilon) && (g_t > 0.0f)) );
    float grad = not_stigma ? 0.0f: g_t;
    if( i < n_s )
        output[i] = grad;
}


void populate_data_gradient(const Eigen::GpuDevice& dev, const float* g, const float* u, float* output, const int n_s){
    populate_data_gradient_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(g, u, output, n_s);
    if(CHECK_ERRORS) check_error(dev, "populate_data_gradient launch failed with error");
}


__device__ float figure_out_conditions(const float g_t, const float gn_t, const float u_t, const float un_t){
    float sign = 0.0f;
    sign = ( u_t > un_t + epsilon) ?  1.0f: sign;
    sign = ( u_t + epsilon < un_t) ? -1.0f: sign;
    float eq = (sign == 0.0f) ? 1.0f: 0.0f;
    float tau0_0 = (u_t  <= epsilon      && g_t > 0.0f) ? 1.0f: 0.0f;
    float tau1_0 = (u_t  >= 1.0f-epsilon && g_t < 0.0f) ? 1.0f: 0.0f;
    float tau0_1 = (un_t <= epsilon      && gn_t > 0.0f) ? 1.0f: 0.0f;
    float tau1_1 = (un_t >= 1.0f-epsilon && gn_t < 0.0f) ? 1.0f: 0.0f;
    float stigma_0 = ((1.0f - tau0_0) * (1.0f - tau1_0));
    float stigma_1 = ((1.0f - tau0_1) * (1.0f - tau1_1));
    float grad = stigma_0 * stigma_1 * (sign*(gn_t-g_t) + eq*abs(gn_t-g_t))
               + stigma_1 * (tau1_0 - tau0_0) * gn_t
               + stigma_0 * (tau1_1 - tau0_1) * g_t;
    return grad;
}


__global__ void populate_reg_gradient_kernel(const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    
    //do z first
    float g_t = g[i];
    float u_t = u[i];
    float gn_t = g[i+1];
    float un_t = u[i+1];
    float grad = figure_out_conditions(g_t, gn_t, u_t, un_t);
    grad = (z == n_z-1) ? 0.0f: grad;
    if( i < n_s )
        g_rz[i] = grad;
    
    //do y
    g_t = g[i];
    u_t = u[i];
    gn_t = g[i+n_z];
    un_t = u[i+n_z];
    grad = figure_out_conditions(g_t, gn_t, u_t, un_t);
    grad = (y == n_y-1) ? 0.0f: grad;
    if( i < n_s )
        g_ry[i] = grad;
    
    //do x
    g_t = g[i];
    u_t = u[i];
    gn_t = g[i+n_z*n_y];
    un_t = u[i+n_z*n_y];
    grad = figure_out_conditions(g_t, gn_t, u_t, un_t);
    grad = (x == n_x-1) ? 0.0f: grad;
    if( i < n_s )
        g_rx[i] = grad;
}



void populate_reg_gradients(const Eigen::GpuDevice& dev, const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c){
    int n_s = n_x*n_y*n_z*n_c;
    populate_reg_gradient_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(g, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_s);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_gradient launch failed with error");
}


__global__ void softmax_gradient_kernel(const float* g, const float* u, float* g_d, const int n_s, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    for(int c = 0; c < n_c; c++){
        float derivative = 0.0f;
        float u_c1 = u[c*n_s+i];
        for(int c2 = 0; c2 < n_c; c2++){
            float u_c2 = u[c2*n_s+i];
            derivative += u_c1 * ( c == c2 ? (1.0f-u_c2) : (-u_c2) ) * g[c2*n_s+i];
        }
        if(i < n_s)
            g_d[c*n_s+i] = derivative;
    }
}



void softmax_gradient(const Eigen::GpuDevice& dev, const float* g, const float* u, float* g_d, const int n_s, const int n_c){
    softmax_gradient_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(g, u, g_d, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_gradient launch failed with error");
}

__global__ void copy_kernel(const float* source, float* dest, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = source[i];
    if(i < n_s)
        dest[i] = val;
}

void copy_buffer(const Eigen::GpuDevice& dev, const float* source, float* dest, const int n_s){
    copy_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(source, dest, n_s);
    if(CHECK_ERRORS) check_error(dev, "copy_buffer launch failed with error");
}


__global__ void copy_clean_kernel(const float* source, float* dest, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = source[i];
    val = isfinite(val) ? val: 0.0f;
    if(i < n_s)
        dest[i] = val;
}

void copy_buffer_clean(const Eigen::GpuDevice& dev, const float* source, float* dest, const int n_s){
    copy_clean_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(source, dest, n_s);
    if(CHECK_ERRORS) check_error(dev, "copy_buffer_clean launch failed with error");
}

__global__ void copy_clip_kernel(const float* source, float* dest, const int n_s, const float clip){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = source[i];
    val = isfinite(val) ? val: 0.0f;
    val = val < -clip ? -clip: val;
    val = val >  clip ?  clip: val;
    if(i < n_s)
        dest[i] = val;
}

void copy_buffer_clip(const Eigen::GpuDevice& dev, const float* source, float* dest, const int n_s, float clip){
    copy_clip_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(source, dest, n_s, clip);
    if(CHECK_ERRORS) check_error(dev, "copy_buffer_clip launch failed with error");
}

__global__ void inc_kernel(const float* inc, float* acc, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = acc[i];
    float increment = inc[i];
    val += increment;
    if(i < n_s)
        acc[i] = val;
}

void inc_buffer(const Eigen::GpuDevice& dev, const float* inc, float* acc, const int n_s){
    inc_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(inc, acc, n_s);
    if(CHECK_ERRORS) check_error(dev, "inc_buffer launch failed with error");
}

__global__ void inc_mult_kernel(const float* inc, float* acc, const int n_s, const float multi){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = acc[i];
    float increment = multi*inc[i];
    val += increment;
    if(i < n_s)
        acc[i] = val;
}

void inc_mult_buffer(const Eigen::GpuDevice& dev, const float* inc, float* acc, const int n_s, const float multi){
    inc_mult_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(inc, acc, n_s, multi);
    if(CHECK_ERRORS) check_error(dev, "inc_mult_buffer launch failed with error");
}

__global__ void inc2_mult_kernel(const float* inc_m1, const float* inc_m2, float* acc, const int n_s, const float multi){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = acc[i];
    float increment = multi*inc_m1[i]*inc_m2[i];
    val += increment;
    if(i < n_s)
        acc[i] = val;
}

void inc2_mult_buffer(const Eigen::GpuDevice& dev, const float* inc_m1, const float* inc_m2, float* acc, const int n_s, const float multi){
    inc2_mult_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(inc_m1, inc_m2, acc, n_s, multi);
    if(CHECK_ERRORS) check_error(dev, "inc2_mult_buffer launch failed with error");
}


__global__ void div_kernel(const float* div, float* res, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = res[i];
    float divisor = div[i];
    val /= divisor;
    if(i < n_s)
        res[i] = val;
}

void div_buffer(const Eigen::GpuDevice& dev, const float* div, float* res, const int n_s){
    div_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(div, res, n_s);
    if(CHECK_ERRORS) check_error(dev, "div_buffer launch failed with error");
}

__global__ void mult_kernel(const float mult, float* res, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = res[i]*mult;
    if(i < n_s)
        res[i] = val;
}

void mult_buffer(const Eigen::GpuDevice& dev, const float mult, float* res, const int n_s){
    mult_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(mult, res, n_s);
    if(CHECK_ERRORS) check_error(dev, "mult_buffer launch failed with error");
}

__global__ void exp_and_inc_kernel(const float* max, float* cost, float* acc, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = cost[i]-max[i];
    val = exp(val);
    float accu = acc[i];
    accu += val;
    if(i < n_s){
        cost[i] = val;
        acc[i] = accu;
    }
}

void exp_and_inc_buffer(const Eigen::GpuDevice& dev, const float* max, float* cost, float* acc, const int n_s){
    exp_and_inc_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(max, cost, acc, n_s);
    if(CHECK_ERRORS) check_error(dev, "exp_and_inc_buffer launch failed with error");
}

__global__ void add_store_then_max_kernel(const float* comp1, const float* comp2, float* store, float* res, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = res[i];
    float comp_val = comp1[i]+comp2[i];
    val = (val > comp_val) ? val: comp_val;
    if(i < n_s){
        res[i] = val;
        store[i] = comp_val;
    }
}

void add_store_then_max_buffer(const Eigen::GpuDevice& dev, const float* comp1, const float* comp2, float* store, float* res, const int n_s){
    add_store_then_max_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(comp1, comp2, store, res, n_s);
    if(CHECK_ERRORS) check_error(dev, "add_store_then_max_buffer launch failed with error");
}

__global__ void softmax_kernel(const float* e1, float* u, const int n_s, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    //find maxmum cost
    float max_cost = e1[i];
    for(int c = 1; c < n_c; c++){
        float cost = e1[c*n_s+i];
        if(cost > max_cost)
            max_cost = cost;
    }
    
    //find accumulator
    float accum = 0.0f;
    for(int c = 0; c < n_c; c++)
        accum += exp(e1[c*n_s+i]-max_cost);
    
    //apply and normalize
    for(int c = 0; c < n_c; c++){
        float value = exp(e1[c*n_s+i]-max_cost) / accum;
        if(i < n_s)
            u[c*n_s+i] = value;
    }
}

__global__ void softmax_kernel(const float* e1, const float* e2, float* u, const int n_s, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    //find maxmum cost
    float max_cost = e1[i]+e2[i];
    for(int c = 1; c < n_c; c++){
        float cost = e1[c*n_s+i]+e2[c*n_s+i];
        if(cost > max_cost)
            max_cost = cost;
    }
    
    //find accumulator
    float accum = 0.0f;
    for(int c = 0; c < n_c; c++)
        accum += exp(e1[c*n_s+i]+e2[c*n_s+i]-max_cost);
    
    //apply and normalize
    for(int c = 0; c < n_c; c++){
        float value = exp(e1[c*n_s+i]+e2[c*n_s+i]-max_cost) / accum;
        //value = (value < epsilon) ? epsilon : value;
        //value = (value > 1.0f-epsilon) ? 1.0f-epsilon : value;
        if(i < n_s)
            u[c*n_s+i] = value;
    }
}

void softmax(const Eigen::GpuDevice& dev, const float* e1, const float* e2, float* u, const int n_s, const int n_c){
    if(e2 != NULL)
        softmax_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(e1, e2, u, n_s, n_c);
    else
        softmax_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(e1, u, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_gradient launch failed with error");
}


__global__ void populate_reg_mean_gradients_kernel(const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    int n_s = n_x*n_y*n_z;
    for(int c = 0; c < n_c; c++){
        //for z
        float up_contra = u[c*n_s+i+1] * g[c*n_s+i];
        float dn_contra = u[c*n_s+i] * g[c*n_s+i+1];
        float derivative = (z < n_z-1) ? up_contra + dn_contra : 0.0f;
        if(i < n_s)
            g_rz[c*n_s+i] = 0.5f * derivative;
        
        //for y
        up_contra = u[c*n_s+i+n_z] * g[c*n_s+i];
        dn_contra = u[c*n_s+i] * g[c*n_s+i+n_z];
        derivative = (y < n_y-1) ? up_contra + dn_contra : 0.0f;
        if(i < n_s)
            g_ry[c*n_s+i] = 0.5f * derivative;
        
        //for x
        up_contra = u[c*n_s+i+n_z*n_y] * g[c*n_s+i];
        dn_contra = u[c*n_s+i] * g[c*n_s+i+n_z*n_y];
        derivative = (x < n_x-1) ? up_contra + dn_contra: 0.0f;
        if(i < n_s)
            g_rx[c*n_s+i] = 0.5f * derivative;
    }
    
}


void populate_reg_mean_gradients(const Eigen::GpuDevice& dev, const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c){
    int n_s = n_x*n_y*n_z;
    populate_reg_mean_gradients_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(g, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_c);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_gradient launch failed with error");
}


__global__ void populate_reg_mean_gradients_and_add_kernel(const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    int n_s = n_x*n_y*n_z;
    for(int c = 0; c < n_c; c++){
        //for z
        float up_contra = u[c*n_s+i+1] * g[c*n_s+i];
        float dn_contra = u[c*n_s+i] * g[c*n_s+i+1];
        float derivative = g_rz[c*n_s+i] + 0.5f*( (z < n_z-1) ? up_contra + dn_contra : 0.0f);
        if(i < n_s)
            g_rz[c*n_s+i] = derivative;
        
        //for y
        up_contra = u[c*n_s+i+n_z] * g[c*n_s+i];
        dn_contra = u[c*n_s+i] * g[c*n_s+i+n_z];
        derivative = g_ry[c*n_s+i] + 0.5f*( (y < n_y-1) ? up_contra + dn_contra : 0.0f);
        if(i < n_s)
            g_ry[c*n_s+i] = derivative;
        
        //for x
        up_contra = u[c*n_s+i+n_z*n_y] * g[c*n_s+i];
        dn_contra = u[c*n_s+i] * g[c*n_s+i+n_z*n_y];
        derivative = g_rx[c*n_s+i] + 0.5f*( (x < n_x-1) ? up_contra + dn_contra: 0.0f);
        if(i < n_s)
            g_rx[c*n_s+i] = derivative;
    }
    
}


void populate_reg_mean_gradients_and_add(const Eigen::GpuDevice& dev, const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c){
    int n_s = n_x*n_y*n_z;
    populate_reg_mean_gradients_and_add_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(g, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_c);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_mean_gradients_and_add launch failed with error");
}

__global__ void change_to_diff_kernel(float* t, float* d, const int n_s, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float old_val = t[i];
    float new_val = d[i];
    float diff = tau*(new_val-old_val);
    new_val = old_val + diff;
    if(i < n_s){
        t[i] = new_val;
        d[i] = diff;
    }
}

void change_to_diff(const Eigen::GpuDevice& dev, float* transfer, float* diff, const int n_s, const float tau){
    change_to_diff_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(transfer, diff, n_s, tau);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_gradient launch failed with error");
}

__device__ float get_effective_reg_kernel_up(const float* u, const float* r, const int c, const int n_s, const int i, const int a, const int d, const int n_d){
        float ut = u[c*n_s+i+a];
        float rt = r[c*n_s+i];
        return (d < n_d-1) ? ut*rt : 0.0f;
}
__device__ float get_effective_reg_kernel_dn(const float* u, const float* r, const int c, const int n_s, const int i, const int a, const int d){
        float ut = u[c*n_s+i-a];
        float rt = r[c*n_s+i-a];
        return (d > 0) ? ut*rt : 0.0f;
}

__global__ void get_effective_reg_kernel(float* r_eff, const float* u, const float* rx, const float* ry, const float* rz, const int n_x, const int n_y, const int n_z, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x;
    int n_s = n_x*n_y*n_z;
    
    for(int c = 0; c < n_c; c++){
        float reg_tot = 0.0f;

        //z direction
        reg_tot += get_effective_reg_kernel_up(u, rz, c, n_s, i, 1, z, n_z);
        reg_tot += get_effective_reg_kernel_dn(u, rz, c, n_s, i, 1, z);

        //y direction
        reg_tot += get_effective_reg_kernel_up(u, ry, c, n_s, i, n_z, y, n_y);
        reg_tot += get_effective_reg_kernel_dn(u, ry, c, n_s, i, n_z, y);

        //x direction
        reg_tot += get_effective_reg_kernel_up(u, rx, c, n_s, i, n_z*n_y, x, n_x);
        reg_tot += get_effective_reg_kernel_dn(u, rx, c, n_s, i, n_z*n_y, x);
        
        reg_tot *= 0.5f;
        if(i < n_s)
            r_eff[c*n_s+i] = reg_tot;
    }
}

__global__ void get_effective_reg_kernel(float* r_eff, const float* u, const float* rx, const float* ry, const int n_x, const int n_y, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x;
    int n_s = n_x*n_y;
    
    for(int c = 0; c < n_c; c++){
        float reg_tot = 0.0f;

        //y direction
        reg_tot += get_effective_reg_kernel_up(u, ry, c, n_s, i, 1, y, n_y);
        reg_tot += get_effective_reg_kernel_dn(u, ry, c, n_s, i, 1, y);

        //x direction
        reg_tot += get_effective_reg_kernel_up(u, rx, c, n_s, i, n_y, x, n_x);
        reg_tot += get_effective_reg_kernel_dn(u, rx, c, n_s, i, n_y, x);
        
        reg_tot *= 0.5f;
        if(i < n_s)
            r_eff[c*n_s+i] = reg_tot;
    }
}

__global__ void get_effective_reg_kernel(float* r_eff, const float* u, const float* rx, const int n_x, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    for(int c = 0; c < n_c; c++){
        float reg_tot = 0.0f;
        
        //x direction
        reg_tot += get_effective_reg_kernel_up(u, rx, c, n_x, i, 1, i, n_x);
        reg_tot += get_effective_reg_kernel_dn(u, rx, c, n_x, i, 1, i);
        
        reg_tot *= 0.5f;
        if(i < n_x)
            r_eff[c*n_x+i] = reg_tot;
    }
}

void get_effective_reg(const Eigen::GpuDevice& dev, float* r_eff, const float* u, const float* rx, const float* ry, const float* rz, const int n_x, const int n_y, const int n_z, const int n_c){
    int n_s = n_x*n_y*n_z;
    get_effective_reg_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(r_eff, u, rx, ry, rz, n_x, n_y, n_z, n_c);
    if(CHECK_ERRORS) check_error(dev, "get_effective_reg (3D) launch failed with error");
}
void get_effective_reg(const Eigen::GpuDevice& dev, float* r_eff, const float* u, const float* rx, const float* ry, const int n_x, const int n_y, const int n_c){
    int n_s = n_x*n_y;
    get_effective_reg_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(r_eff, u, rx, ry, n_x, n_y, n_c);
    if(CHECK_ERRORS) check_error(dev, "get_effective_reg (2D) launch failed with error");
}
void get_effective_reg(const Eigen::GpuDevice& dev, float* r_eff, const float* u, const float* rx, const int n_x, const int n_c){
    get_effective_reg_kernel<<<((n_x+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(r_eff, u, rx, n_x, n_c);
    if(CHECK_ERRORS) check_error(dev, "get_effective_reg (1D) launch failed with error");
}


__global__ void add_then_store_kernel(const float* addend1, const float* addend2, float* sum, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float a1 = addend1[i];
    float a2 = addend2[i];
    float res = a1+a2;
    if(i < n_s)
        sum[i] = res;   
}

void add_then_store(const Eigen::GpuDevice& dev, const float* addend1, const float* addend2, float* sum, const int n_s){
    add_then_store_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(addend1, addend2, sum, n_s);
    if(CHECK_ERRORS) check_error(dev, "add_then_store launch failed with error");
}

__global__ void add_then_store_kernel_2(const float* addend1, const float* addend2, float* sum1, float* sum2, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float a1 = addend1[i];
    float a2 = addend2[i];
    float res = a1+a2;
    if(i < n_s){
        sum1[i] = res;   
        sum2[i] = res;
    }
}

void add_then_store_2(const Eigen::GpuDevice& dev, const float* addend1, const float* addend2, float* sum1, float* sum2, const int n_s){
    add_then_store_kernel_2<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(addend1, addend2, sum1, sum2, n_s);
    if(CHECK_ERRORS) check_error(dev, "add_then_store_2 launch failed with error");
}

__global__ void process_grad_potts_kernel(const float* du_i, const float* u, float* loc, const int n_s, const int n_c, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    for(int c = 0; c < n_c; c++){
        float new_grad = 0.0f;
        float uc = u[c*n_s+i];
        for(int a = 0; a < n_c; a++){
            float da = du_i[a*n_s+i];
            if(c == a)
                new_grad += da*(1.0f-uc);
            else
                new_grad -= da*u[a*n_s+i];
        }
        new_grad *= uc*tau;
        if( i < n_s )
            loc[c*n_s+i] = new_grad;
    }
    
}

void process_grad_potts(const Eigen::GpuDevice& dev, const float* du_i, const float* u, float* loc, const int n_s, const int n_c, const float tau){
    process_grad_potts_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(du_i, u, loc, n_s, n_c, tau);
    if(CHECK_ERRORS) check_error(dev, "process_grad_potts launch failed with error");
}

__device__ float get_gradient_for_u_kernel_dn(const float* dy, const float* r, const int c, const int n_s, const int i, const int a, const int d){
    float multiplier = dy[c*n_s+i-a];
    float inc = multiplier*r[c*n_s+i-a];
    return (d > 0) ? inc: 0.0f;
}
__device__ float get_gradient_for_u_kernel_up(const float* dy, const float* r, const int c, const int n_s, const int i, const int a, const int d, const int n_d){
    float multiplier = dy[c*n_s+i+a];
    float inc = multiplier*r[c*n_s+i+a];
    return (d < n_d-1) ? inc: 0.0f;
}

__global__ void get_gradient_for_u_kernel(const float* dy, float* du, const float* rx, const float* ry, const float* rz, const int n_x, const int n_y, const int n_z, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x;
    int n_s = n_x*n_y*n_z;
    
    for(int c = 0; c < n_c; c++){
        float grad_val = 0.0f;
            
        grad_val += get_gradient_for_u_kernel_dn(dy, rz, c, n_s, i, 1, z);
        grad_val += get_gradient_for_u_kernel_up(dy, rz, c, n_s, i, 1, z, n_z);
        
        grad_val += get_gradient_for_u_kernel_dn(dy, ry, c, n_s, i, n_z, y);
        grad_val += get_gradient_for_u_kernel_up(dy, ry, c, n_s, i, n_z, y, n_y);
        
        grad_val += get_gradient_for_u_kernel_dn(dy, rx, c, n_s, i, n_z*n_y, x);
        grad_val += get_gradient_for_u_kernel_up(dy, rx, c, n_s, i, n_z*n_y, x, n_x);
        
        grad_val *= 0.5f;
        if(i < n_s)
            du[c*n_s+i] = grad_val;
    }
}

__global__ void get_gradient_for_u_kernel(const float* dy, float* du, const float* rx, const float* ry, const int n_x, const int n_y, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x;
    int n_s = n_x*n_y;
    
    for(int c = 0; c < n_c; c++){
        float grad_val = 0.0f;
        
        grad_val += get_gradient_for_u_kernel_dn(dy, ry, c, n_s, i, 1, y);
        grad_val += get_gradient_for_u_kernel_up(dy, ry, c, n_s, i, 1, y, n_y);
        
        grad_val += get_gradient_for_u_kernel_dn(dy, rx, c, n_s, i, n_y, x);
        grad_val += get_gradient_for_u_kernel_up(dy, rx, c, n_s, i, n_y, x, n_x);
        
        grad_val *= 0.5f;
        if(i < n_s)
            du[c*n_s+i] = grad_val;
    }
}

__global__ void get_gradient_for_u_kernel(const float* dy, float* du, const float* rx, const int n_x, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    for(int c = 0; c < n_c; c++){
        float grad_val = 0.0f;
        
        grad_val += get_gradient_for_u_kernel_dn(dy, rx, c, n_x, i, 1, i);
        grad_val += get_gradient_for_u_kernel_up(dy, rx, c, n_x, i, 1, i, n_x);
        
        grad_val *= 0.5f;
        if(i < n_x)
            du[c*n_x+i] = grad_val;
    }
}

void get_gradient_for_u(const Eigen::GpuDevice& dev, const float* dy, float* du, const float* rx, const float* ry, const float* rz, const int n_x, const int n_y, const int n_z, const int n_c){
    int n_s = n_x*n_y*n_z;
    get_gradient_for_u_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(dy, du, rx, ry, rz, n_x, n_y, n_z, n_c);
    if(CHECK_ERRORS) check_error(dev, "get_gradient_for_u (3D) launch failed with error");
}
void get_gradient_for_u(const Eigen::GpuDevice& dev, const float* dy, float* du, const float* rx, const float* ry, const int n_x, const int n_y, const int n_c){
    int n_s = n_x*n_y;
    get_gradient_for_u_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(dy, du, rx, ry, n_x, n_y, n_c);
    if(CHECK_ERRORS) check_error(dev, "get_gradient_for_u (2D) launch failed with error");
}
void get_gradient_for_u(const Eigen::GpuDevice& dev, const float* dy, float* du, const float* rx, const int n_x, const int n_c){
    get_gradient_for_u_kernel<<<((n_x+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream()>>>(dy, du, rx, n_x, n_c);
    if(CHECK_ERRORS) check_error(dev, "get_gradient_for_u (1D) launch failed with error");
}

#endif // GOOGLE_CUDA