#include <cuda.h>
//#include <cublas_v2.h>
#include <stdio.h>

#include "gpu_kernels.h"
#include "gpu_kernels_auglag.h"


// Flags specific to the GPU (i.e. debug, number of threads, etc...)
#define CHECK_ERRORS true
#define NUM_THREADS 256
#define epsilon 0.00001f

__global__ void init_flows_binary_kernel(const float* data, float* ps, float* pt, float* u, const int n_s){
	int i = blockIdx.x * NUM_THREADS + threadIdx.x;
	float d_t = (i < n_s) ? data[i] : 0.0f;
	float ps_t = (d_t > 0.0f) ? d_t : 0.0f ;
	float pt_t = (d_t < 0.0f) ? -d_t : 0.0f ;
    float u_t = (d_t > 0.0f) ? 1.0f : 0.0f ;
	if(i < n_s){
		ps[i] = ps_t;
		pt[i] = pt_t;
        u[i]  = u_t;
	}
	
}

void init_flows_binary(const CUDA_DEVICE& dev, const float* data, float* ps, float* pt, float* u, const int n_s){
	init_flows_binary_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), (n_s > NUM_THREADS) ? NUM_THREADS : n_s, 0, dev.stream>>>(data, ps, pt, u, n_s);
    if(CHECK_ERRORS) check_error(dev, "init_flows_binary_kernel launch failed with error");
}

__global__ void init_flows_potts_kernel(const float* data, float* ps, float* pt, float* u, const int n_s, const int n_c){
	int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float max = ((i < n_s) ? data[i] : 0.0f);
    for(int c = 1; c < n_c; c++){
        float d_t = ((i < n_s) ? data[c*n_s+i] : 0.0f);
        if(d_t > max)
            max = d_t;
    }
    
    if(i < n_s)
        ps[i] = -max;
    
    for(int c = 0; c < n_c; c++){
        float d_t = ((i < n_s) ? data[c*n_s+i] : 0.0f);
        float u_t = (d_t == max) ? 1.0f: 0.0f;
        if(i < n_s){
            pt[c*n_s+i] = -max;
            u[c*n_s+i] = u_t;
        }
    }
	
}

void init_flows_potts(const CUDA_DEVICE& dev, const float* data, float* ps, float* pt, float* u, const int n_s, const int n_c){
	init_flows_potts_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(data, ps, pt, u, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "init_flows_potts_kernel launch failed with error");
}


__global__ void prep_flow_hmf_kernel(float* g, float* const* const ps_ind, const float* pt, const float* div, const float* u, const float icc, const int n_s, const int n_c){
    __shared__ float* ps_ind_t [NUM_THREADS];
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    ps_ind_t[threadIdx.x] = ps_ind[threadIdx.x];
    __syncthreads();
    
    for(int c = 0; c < n_c; c++){
        float* ps_ptr = ps_ind_t[c];
        float pre_val = ps_ptr[i] - div[i+c*n_s] - icc * u[i+c*n_s];
        if(i < n_s)
            g[i+c*n_s] = pre_val;
    }
}

void prep_flow_hmf(const CUDA_DEVICE& dev, float* g, float* const* const ps_ind, const float* pt, const float* div, const float* u, const float icc, const int n_s, const int n_c){
    prep_flow_hmf_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, ps_ind, pt, div, u, icc, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "prep_flow_hmf launch failed with error");
}

__global__ void compute_parents_flow_hmf_kernel(float** g_ind, const float* pt, const float* div, const float* u, const float icc, const int n_s, const int n_c){
    __shared__ float* g_ind_t [NUM_THREADS];
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    g_ind_t[threadIdx.x] = g_ind[threadIdx.x];
    __syncthreads();
    
    for(int c = 0; c < n_c; c++){
        float* pg_ptr = g_ind_t[c];
        float update = pg_ptr[i] + pt[i+c*n_s] + div[i+c*n_s] - icc * u[i+c*n_s];
        if( i < n_s )
            pg_ptr[i] = update;
    }
    
}

void compute_parents_flow_hmf(const CUDA_DEVICE& dev, float** g_ind, const float* pt, const float* div, const float* u, const float icc, const int n_s, const int n_c){
    compute_parents_flow_hmf_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g_ind, pt, div, u, icc, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "compute_parents_flow_hmf launch failed with error");
}

__global__ void update_flow_hmf_kernel(float** g_ind, float* g_s, float* g, float* const* const ps_ind, float* ps, float* pt, const float* div, const float* u, const float icc, const int* p_c, const int s_c, const int n_s, const int n_c){
    __shared__ float* g_ind_t [NUM_THREADS/4];
    __shared__ float* ps_ind_t [NUM_THREADS/4];
    __shared__ int c_t [NUM_THREADS/4];
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    if( threadIdx.x < NUM_THREADS/4 ){
        ps_ind_t[threadIdx.x] = ps_ind[threadIdx.x];
        g_ind_t[threadIdx.x] = g_ind[threadIdx.x];
        c_t[threadIdx.x] = p_c[threadIdx.x];
    }
    __syncthreads();
    
    //set buffer to initial self-capacity
    if(i < n_s)
        g_s[i] = icc;
    
    for(int c = 0; c < n_c; c++){
        float* ps_ptr = ps_ind_t[c];
        float pre_val = ps_ptr[i] - div[i+c*n_s] - icc * u[i+c*n_s];
        if(i < n_s)
            g[i+c*n_s] = pre_val;
    }
    
    //insert component from children into parents tmp buffer
    for(int c = 0; c < n_c; c++){
        float* pg_ptr = g_ind_t[c];
        float update = pg_ptr[i] + pt[i+c*n_s] + div[i+c*n_s] - icc * u[i+c*n_s];
        if( i < n_s )
            pg_ptr[i] = update;
    }
    
    //divide out to normalize by number of children and output into flow buffers
    float g_s_val = g_s[i] / (float) s_c;
    if(i < n_s)
        ps[i] = g_s_val;
    for(int c = 0; c < n_c; c++){
        int num_c = c_t[c];
        float g_val = g[i+c*n_s] / (float) num_c;
        if(i < n_s)
            pt[i+c*n_s] = g_val;
    }
}

void update_flow_hmf(const CUDA_DEVICE& dev, float** g_ind, float* g_s, float* g, float** ps_ind, float* ps, float* pt, const float* div, const float* u, const float icc, const int* p_c, const int s_c, const int n_s, const int n_c){
    update_flow_hmf_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g_ind, g_s, g, ps_ind, ps, pt, div, u, icc, p_c, s_c, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "update_flow_hmf launch failed with error");
}

__global__ void divide_out_and_store_hmf_kernel(const float* g_s, const float* g, float* ps, float* pt, const int* p_c, const int s_c, const int n_s, const int n_c){
    __shared__ int c_t [NUM_THREADS];
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    c_t[threadIdx.x] = p_c[threadIdx.x];
    __syncthreads();
    
    //divide out to normalize by number of children and output into flow buffers
    float g_s_val = g_s[i] / (float) s_c;
    if(i < n_s)
        ps[i] = g_s_val;
    for(int c = 0; c < n_c; c++){
        int num_c = c_t[c];
        float g_val = g[i+c*n_s] / (float) num_c;
        if(i < n_s)
            pt[i+c*n_s] = g_val;
    }
}

void divide_out_and_store_hmf(const CUDA_DEVICE& dev, const float* g_s, const float* g, float* ps, float* pt, const int* p_c, const int s_c, const int n_s, const int n_c){
    divide_out_and_store_hmf_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g_s, g, ps, pt, p_c, s_c, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "divide_out_and_store_hmf launch failed with error");
}

// update the source flow
__global__ void update_source_sink_multiplier_potts_kernel(float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_c, const int n_s){
    
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    //update source flow
    float ps_t = icc;
    for(int c = 0; c < n_c; c++){
        ps_t += (c*n_s+i < n_c*n_s) ? pt[c*n_s+i] : 0.0f;
        ps_t += (c*n_s+i < n_c*n_s) ? div[c*n_s+i] : 0.0f;
        ps_t -= (c*n_s+i < n_c*n_s) ? (u[c*n_s+i]*icc) : 0.0f;
    }
    ps_t /= n_c;
    if( i < n_s )
        ps[i] = ps_t;
    
    //update sink flow
    for(int c = 0; c < n_c; c++){
        float div_t = (c*n_s+i < n_c*n_s) ? div[c*n_s+i] : 0.0f;
        float u_t   = (c*n_s+i < n_c*n_s) ? u[c*n_s+i] : 0.0f;
        float d_t   = (c*n_s+i < n_c*n_s) ? -d[c*n_s+i] : 0.0f;
        float pt_t = ps_t - div_t + u_t*icc;
        pt_t = (pt_t > d_t) ? d_t : pt_t;
    
        
        //update multiplier
        float erru_t = cc * (ps_t - div_t - pt_t);
        u_t = u_t + erru_t;
        erru_t = (erru_t < 0.0f) ? -erru_t : erru_t;
        
        //output
        if( i < n_s ){
            pt[c*n_s+i] = pt_t;
            u[c*n_s+i] = u_t;
            erru[c*n_s+i] = erru_t;
        }
    }
    
}

void update_source_sink_multiplier_potts(const CUDA_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_c, const int n_s){
    update_source_sink_multiplier_potts_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(ps, pt, div, u, erru, d, cc, icc, n_c, n_s);
    if(CHECK_ERRORS) check_error(dev, "update_source_sink_multiplier_potts launch failed with error");
}

// update the source flow
__global__ void update_source_sink_multiplier_binary_kernel(float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_s){
    
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float div_t = (i < n_s) ? div[i] : 0.0f;
	float u_t = (i < n_s) ? u[i] : 0.0f;
	float d_t = (i < n_s) ? d[i] : 0.0f;
    float pt_t = (i < n_s) ? pt[i] : 0.0f;
    
    //update source flow and constrain
	float ps_t = icc+pt_t+div_t-u_t*icc;
	ps_t = (d_t <= 0.0f) ? 0.0f : ps_t;
	ps_t = (ps_t > d_t && d_t > 0.0f) ? d_t : ps_t;
    
    //update sink flow and constrain
    pt_t = ps_t-div_t+u_t*icc;
	pt_t = (d_t >= 0.0f) ? 0.0f : pt_t;
	pt_t = (pt_t > -d_t && d_t < 0.0f) ? -d_t : pt_t;
        
	//update multiplier
	float erru_t = cc * (ps_t - div_t - pt_t);
	u_t += erru_t;
	erru_t = (erru_t < 0.0f) ? -erru_t : erru_t;
        
	//output
	if( i < n_s ){
        ps[i] = ps_t;
        pt[i] = pt_t;
		u[i] = u_t;
		erru[i] = erru_t;
	}
    
}

void update_source_sink_multiplier_binary(const CUDA_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_s){
    update_source_sink_multiplier_binary_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(ps, pt, div, u, erru, d, cc, icc, n_s);
    if(CHECK_ERRORS) check_error(dev, "update_source_sink_multiplier_binary launch failed with error");
}

// update the source flow
__global__ void update_multiplier_hmf_kernel(float* const* const ps_ind, const float* pt, const float* div, float* u, float* erru, const float cc, const int n_s, const int n_c){
    __shared__ const float* ps_ind_t [NUM_THREADS];
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    ps_ind_t[threadIdx.x] = ps_ind[threadIdx.x];
    __syncthreads();
    
    for(int c = 0; c < n_c; c++){
        const float* ps_ptr = ps_ind_t[c];
        float diff = cc*(pt[i+c*n_s] + div[i+c*n_s] - ps_ptr[i]);
        float new_u = u[i+c*n_s] - diff;
        if(i < n_s){
            erru[i+c*n_s] = diff;
            u[i+c*n_s] = new_u;
        }
    }
}

void update_multiplier_hmf(const CUDA_DEVICE& dev, float* const* const ps_ind, const float* div, const float* pt, float* u, float* erru, const int n_s, const int n_c, const float cc){
    update_multiplier_hmf_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(ps_ind, pt,div,u,erru,cc, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "update_multiplier_hmf launch failed with error");
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



void update_source_flows(const CUDA_DEVICE& dev, float* ps, const float* pt, const float* div, const float* u, float icc, const int n_c, const int n_s){
    update_source_flows_kernel_channel_first<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(ps, pt, div, u, icc, n_c, n_s);
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


void update_sink_flows(const CUDA_DEVICE& dev, const float* ps, float* pt, const float* div, const float* u, const float* d, float icc, const int n_c, const int n_s){
    update_sink_flows_kernel<<<((n_s*n_c+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(ps, pt, div, u, d, icc, n_c*n_s, n_s );
    if(CHECK_ERRORS) check_error(dev, "update_sink_flows launch failed with error");
}



__global__ void update_multiplier_kernel(const float* ps, const float* pt, const float* div, float* u, float* erru, float cc, const int n_t, const int n_s) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int ps_i = i % n_s;
    
    float erru_t = ps[ps_i];
    erru_t -= div[i];
    erru_t -= pt[i];
    erru_t *= cc;
    float u_t = u[i] + erru_t;
    erru_t = (erru_t < 0.0f) ? -erru_t: erru_t;
    if( i < n_t ){
        u[i] = u_t;
        erru[i] = erru_t;
    }
    
}


void update_multiplier(const CUDA_DEVICE& dev, const float* ps, const float* pt, const float* div, float* u, float* erru, float cc, const int n_c, const int n_s){
    update_multiplier_kernel<<<((n_s*n_c+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(ps, pt, div, u, erru, cc, n_c*n_s, n_s );
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


//find the maximum in the neg_constraint buffer and then multiply it by -1
void find_min_constraint(const CUDA_DEVICE& dev, float* output, const float* neg_constraint, const int n_c, const int n_s){
    find_min_constraint_channel_first<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(output, neg_constraint, n_c, n_s );
    if(CHECK_ERRORS) check_error(dev, "find_min_constraint launch failed with error");
}


__global__ void calc_divergence_kernel(float* div, const float* px, const float* py, const float* pz, const int n_x, const int n_y, const int n_z, const int n_s) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    
    //do z dimension (add just one)
    float div_t = -((i < n_s) ? pz[i] : 0.0f);
    float more_flow = (i-1 < n_s && i-1 >= 0) ? pz[i-1] : 0.0f;
    div_t += (z > 0) ? more_flow : 0.0f;
    
    //do y dimension (add n_z)
    div_t -= (i < n_s) ? py[i] : 0.0f;
    more_flow = (i-n_z < n_s && i-n_z >= 0) ? py[i-n_z] : 0.0f;
    div_t += (y > 0) ? more_flow : 0.0f;
    
    //do x dimension (add n_z*n_y)
    div_t -= (i < n_s) ? px[i] : 0.0f;
    more_flow = (i-n_z*n_y < n_s && i-n_z*n_y >= 0) ? px[i-n_z*n_y] : 0.0f;
    div_t += (x > 0) ? more_flow : 0.0f;
    
    if(i < n_s)
        div[i] = div_t;
}

__global__ void calc_divergence_kernel(float* div, const float* px, const float* py, const int n_x, const int n_y, const int n_s) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;

    //do y dimension (add n_z)
    float div_t = -((i < n_s) ? py[i] : 0.0f);
    float more_flow = (i-1 < n_s && i-1 >= 0) ? py[i-1] : 0.0f;
    div_t += (y > 0) ? more_flow : 0.0f;
    
    //do x dimension (add n_z*n_y)
    div_t -= (i < n_s) ? px[i] : 0.0f;
    more_flow = (i-n_y < n_s && i-n_y >= 0) ? px[i-n_y] : 0.0f;
    div_t += (x > 0) ? more_flow : 0.0f;
    
    if(i < n_s)
        div[i] = div_t;
}

__global__ void calc_divergence_kernel(float* div, const float* px, const int n_x, const int n_s) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int x = i_temp % n_x; i_temp /= n_x;

    //do x dimension (add 1)
    float div_t = -((i < n_s) ? px[i] : 0.0f);
    float more_flow = (i-1 < n_s && i-1 >= 0) ? px[i-1] : 0.0f;
    div_t += (x > 0) ? more_flow : 0.0f;
    
    if(i < n_s)
        div[i] = div_t;
}


__device__ void update_flows_kernel(const int i, const int x, const int y, const int z, const float* g, float* px, float* py, float* pz, const int n_x, const int n_y, const int n_z, const int n_s){
    //for z
    float g_t = ((i < n_s) ? g[i] : 0.0f);
    float capacity = g_t - ((i+1 < n_s) ? g[i+1] : 0.0f);
    float newflow = (i < n_s && z < n_z-1) ? pz[i] + capacity : 0.0f;
    if(i < n_s)
        pz[i] = newflow;
    
    //for y
    capacity = g_t - ((i+n_z < n_s) ? g[i+n_z] : 0.0f);
    newflow = (i < n_s && y < n_y-1) ? py[i] + capacity : 0.0f;
    if(i < n_s)
        py[i] = newflow;
    
    //for x
    capacity = g_t - ((i+n_y*n_z < n_s) ? g[i+n_y*n_z] : 0.0f);
    newflow = (i < n_s && x < n_x-1) ? px[i] + capacity : 0.0f;
    if(i < n_s)
        px[i] = newflow;
}

__device__ void update_flows_kernel(const int i, const int x, const int y, const float* g, float* px, float* py, const int n_x, const int n_y, const int n_s){
    //for y
    float g_t = ((i < n_s) ? g[i] : 0.0f);
    float capacity = g_t - ((i+1 < n_s) ? g[i+1] : 0.0f);
    float newflow = (i < n_s && y < n_y-1) ? py[i] + capacity : 0.0f;
    if(i < n_s)
        py[i] = newflow;
    
    //for x
    capacity = g_t - ((i+n_y < n_s) ? g[i+n_y] : 0.0f);
    newflow = (i < n_s && x < n_x-1) ? px[i] + capacity : 0.0f;
    if(i < n_s)
        px[i] = newflow;
}

__device__ void update_flows_kernel(const int i, const int x, const float* g, float* px, const int n_x, const int n_s){
    //for x
    float capacity = ((i < n_s) ? g[i] : 0.0f) - ((i+1 < n_s) ? g[i+1] : 0.0f);
    float newflow = (i < n_s && x < n_x-1) ? px[i] + capacity : 0.0f;
    if(i < n_s)
        px[i] = newflow;
}

__device__ void abs_constrain_device_auglag(const int i, float* b, const float* r, const float* l, const int n_s){
    float value = (i < n_s) ? b[i] : 0.0f;
    float constraint = (i < n_s) ? r[i] : 0.0f;
    float exception =  (l && i < n_s) ? l[i] : 0.0f;
    value = (value < constraint || value*exception > 0.0) ? value : constraint;
    value = (value > -constraint || value*exception > 0.0) ? value: -constraint;
    
    if(i < n_s)
        b[i] = value;
    
}

__global__ void update_spatial_flows_3d_kernel(const float* const g, float* const div, float* const px, float* const py, float* const pz, const float* const rx, const float* const ry, const float* const rz, const float* const lx, const float* const ly, const float* const lz, const int n_x, const int n_y, const int n_z, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    
    update_flows_kernel(i,x,y,z, g, px, py, pz, n_x, n_y, n_z, n_s);
    abs_constrain_device_auglag(i, px, rx, lx, n_s);
    abs_constrain_device_auglag(i, py, ry, ly, n_s);
    abs_constrain_device_auglag(i, pz, rz, lz, n_s);
}

__global__ void update_spatial_flows_2d_kernel(const float* const g, float* const div, float* const px, float* const py, const float* const rx, const float* const ry, const float* const lx, const float* const ly, const int n_x, const int n_y, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    
    update_flows_kernel(i,x,y, g, px, py, n_x, n_y, n_s);
    abs_constrain_device_auglag(i, px, rx, lx, n_s);
    abs_constrain_device_auglag(i, py, ry, ly, n_s);
}

__global__ void update_spatial_flows_1d_kernel(const float* const g, float* const div, float* const px, const float* const rx, const float* const lx, const int n_x, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int x = i_temp % n_x; i_temp /= n_x;
    
    update_flows_kernel(i, x, g, px, n_x, n_s);
    abs_constrain_device_auglag(i, px, rx, lx, n_s);
}

void update_spatial_flows(const CUDA_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const int dim, const int* const n, const int n_c){
    int n_s = n_c;
    for(int i = 0; i < dim; i++)
        n_s *= n[i];
    
    switch(dim){
        case 1:
            update_spatial_flows_1d_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, div, p[0], r[0], 0, n[0], n_s);
            calc_divergence_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(div, p[0], n[0], n_s);
            break;
        case 2:
            update_spatial_flows_2d_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, div, p[0], p[1], r[0], r[1], 0, 0, n[0], n[1], n_s);
            calc_divergence_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(div, p[0], p[1], n[0], n[1], n_s);
            break;
        case 3:
            update_spatial_flows_3d_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, div, p[0], p[1], p[2], r[0], r[1], r[2], 0, 0, 0, n[0], n[1], n[2], n_s);
            calc_divergence_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(div, p[0], p[1], p[2], n[0], n[1], n[2], n_s);
            break;
    }
    if(CHECK_ERRORS) check_error(dev, "update_spatial_flows_kernel launch failed with error");
}

void update_spatial_star_flows(const CUDA_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const float *const *const l, const int dim, const int* const n, const int n_c){
    int n_s = n_c;
    for(int i = 0; i < dim; i++)
        n_s *= n[i];
    
    switch(dim){
        case 1:
            update_spatial_flows_1d_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, div, p[0], r[0], l[0], n[0], n_s);
            calc_divergence_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(div, p[0], n[0], n_s);
            break;
        case 2:
            update_spatial_flows_2d_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, div, p[0], p[1], r[0], r[1], l[0], l[1], n[0], n[1], n_s);
            calc_divergence_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(div, p[0], p[1], n[0], n[1], n_s);
            break;
        case 3:
            update_spatial_flows_3d_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, div, p[0], p[1], p[2], r[0], r[1], r[2], l[0], l[1], l[2], n[0], n[1], n[2], n_s);
            calc_divergence_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(div, p[0], p[1], p[2], n[0], n[1], n[2], n_s);
            break;
    }
    if(CHECK_ERRORS) check_error(dev, "update_spatial_flows_kernel launch failed with error");
}

__global__ void calc_capacity_potts_kernel(float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float ps_t = (i < n_s) ? ps[i] : 0.0f;
    for(int c = 0; c < n_c; c++){
        float div_t = (i+c*n_s < n_c*n_s) ? div[i+c*n_s] : 0.0f;
        float pt_t  = (i+c*n_s < n_c*n_s) ? pt [i+c*n_s] : 0.0f;
        float u_t   = (i+c*n_s < n_c*n_s) ? u  [i+c*n_s] : 0.0f;
        float g_t = div_t + pt_t - ps_t - icc * u_t;
        g_t *= tau;
        if(i < n_s)
            g[i+c*n_s] = g_t;
    }
}

void calc_capacity_potts(const CUDA_DEVICE& dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau){
    calc_capacity_potts_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, div, ps, pt, u, n_s, n_c, icc, tau);
    if(CHECK_ERRORS) check_error(dev, "calc_capacity_potts launch failed with error");
}

__global__ void calc_capacity_binary_kernel(float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const float icc, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float g_t = (i < n_s) ? div[i] + pt[i] - ps[i] - icc * u[i] : 0.0f;
    g_t *= tau;
	if(i < n_s)
		g[i] = g_t;
}

void calc_capacity_binary(const CUDA_DEVICE& dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const float icc, const float tau){
	calc_capacity_binary_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, div, ps, pt, u, n_s, icc, tau);
    if(CHECK_ERRORS) check_error(dev, "calc_capacity_binary_kernel launch failed with error");
}

__global__ void calc_capacity_potts_source_separate_kernel(float* g, const float* div, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    for(int c = 0; c < n_c; c++){
        float g_t = (i+c*n_s < n_c*n_s) ? div[i+c*n_s] + pt[i+c*n_s] - icc * u[i+c*n_s] : 0.0f;
        g_t *= tau;
        if(i < n_s)
            g[i+c*n_s] = g_t;
    }
}

void calc_capacity_potts_source_separate(const CUDA_DEVICE& dev, float* g, const float* div, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau){
    calc_capacity_potts_source_separate_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, div, pt, u, n_s, n_c, icc, tau);
    if(CHECK_ERRORS) check_error(dev, "calc_capacity_potts_source_separate launch failed with error");
}

__global__ void calc_capacity_hmf_kernel(float* g, float* const* const ps_ind, const float* div, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau){
    __shared__ const float* ps_ind_t [NUM_THREADS];
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    ps_ind_t[threadIdx.x] = ps_ind[threadIdx.x];
    __syncthreads();
    
    for(int c = 0; c < n_c; c++){
        const float* ps_ptr = ps_ind_t[c];
        float g_t = div[i+c*n_s] + pt[i+c*n_s] - ps_ptr[0] - icc * u[i+c*n_s];
        g_t *= tau;
        if(i < n_s)
            g[i+c*n_s] = g_t;
    }
}

void calc_capacity_hmf(const CUDA_DEVICE& dev, float* g, float* const* const ps_ind, const float* div, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau){
    calc_capacity_hmf_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, ps_ind, div, pt, u, n_s, n_c, icc, tau);
    if(CHECK_ERRORS) check_error(dev, "calc_capacity_hmf launch failed with error");
}