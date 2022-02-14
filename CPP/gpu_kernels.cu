#include <cuda.h>
//#include <cublas_v2.h>
#include <stdio.h>

#include "gpu_kernels.h"

#define NUM_THREADS 256
#define epsilon 0.00001f

#define CHECK_ERRORS true

void let_device_catchup(const cudaStream_t& dev){
    cudaError_t cudaerr = cudaStreamSynchronize(dev);
}

void check_error(const cudaStream_t& dev, const char* string){
    cudaError_t cudaerr = cudaStreamSynchronize(dev);
    if (cudaerr != cudaSuccess){
        printf(string);
        printf(" \"%s\".\n", cudaGetErrorString(cudaerr));
        exit(-1);
    }
}

void* allocate_on_gpu(const cudaStream_t& dev, size_t amount){
    void* ptr;
    cudaMalloc(&ptr,amount);
    if(CHECK_ERRORS) check_error(dev, "cudaMalloc failed");
    return ptr;
}

void deallocate_on_gpu(const cudaStream_t& dev, void* ptr){
    cudaFree(ptr);
}

void get_from_gpu(const cudaStream_t& dev, const void* source, void* dest, size_t amount){
    cudaMemcpyAsync(dest,source,amount,cudaMemcpyDeviceToHost,dev);
    cudaStreamSynchronize(dev);
}

void print_buffer(const cudaStream_t& dev, const float* buffer, const int n_s){
	float* c_buffer = (float*) malloc(n_s*sizeof(float));
	get_from_gpu(dev, buffer, c_buffer, n_s*sizeof(float));
	for(int i = 0; i < n_s; i++)
		printf("%f ",c_buffer[i]);
	printf("\n");
	free(c_buffer);
}

void send_to_gpu(const cudaStream_t& dev, const void* source, void* dest, size_t amount){
    cudaStreamSynchronize(dev);
    cudaMemcpyAsync(dest,source,amount,cudaMemcpyHostToDevice,dev);
    cudaStreamSynchronize(dev);
}

// Sets variables to 0.0f
void clear_buffer(const cudaStream_t& dev, float* buffer, const int size){
    set_buffer(dev, buffer, 0.0f, size);
}

void clear_buffer(const cudaStream_t& dev, int* buffer, const int size){
    cudaMemsetAsync(buffer, 0, size*sizeof(int),dev);
}

__global__ void set_kernel(float* buffer, const float number, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    if( i < n_s )
        buffer[i] = number;
}
    

void set_buffer(const cudaStream_t& dev, float* buffer, const float number, const int n_s){
    set_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(buffer, number, n_s);
    if(CHECK_ERRORS) check_error(dev, "set_buffer launch failed with error");
}

__global__ void mark_neg_equal_kernel(const float* buffer_s, const float* buffer_l, float* u, const int n_s, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float num1 = buffer_s[i];
    for(int c = 0; c < n_c; c++){
        float num2 = buffer_l[i+c*n_s];
        float out = (num1 == -num2) ? 1.0f : 0.0f;
        if(i < n_s)
            u[i+c*n_s] = out;
    }
}

void mark_neg_equal(const cudaStream_t& dev, const float* buffer_s, const float* buffer_l, float* u, const int n_s, const int n_c){
    mark_neg_equal_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(buffer_s, buffer_l, u, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "mark_neg_equal launch failed with error");
}

__global__ void aggregate_bottom_up_kernel(float** p_ind, float* buffer, const float* org, const int n_s, const int n_c, const int n_r){
    __shared__ float* p_ind_t [NUM_THREADS];
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    p_ind_t[threadIdx.x] = p_ind[threadIdx.x];
    __syncthreads();
    
    //clear buffers
    for(int c = 0; c < n_r; c++){
        if( c < n_r-n_c){
            if( i < n_s )
                buffer[i+c*n_s] = 0.0f;
        }else{
            float outputval = org[i+c*n_s];
            if( i < n_s )
                buffer[i+c*n_s] = outputval;
        }
    }
    
    for(int c = 0; c < n_r; c++){
        //get value of own buffer (and write it if leaf)
        float buffer_val = buffer[i+c*n_s];
        
        //accumulate into parents buffer
        float* p_ptr = p_ind_t[c];
        if( p_ptr ){
            buffer_val += p_ptr[i];
            if( i < n_s )
                p_ptr[i] = buffer_val;
        }
    }
}

void aggregate_bottom_up(const cudaStream_t& dev, float** p_ind, float* buffer, const float* org, const int n_s, const int n_c, const int n_r){
    aggregate_bottom_up_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(p_ind, buffer, org, n_s, n_c, n_r);
    if(CHECK_ERRORS) check_error(dev, "aggregate_bottom_up launch failed with error");
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

void prep_flow_hmf(const cudaStream_t& dev, float* g, float* const* const ps_ind, const float* pt, const float* div, const float* u, const float icc, const int n_s, const int n_c){
    prep_flow_hmf_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, ps_ind, pt, div, u, icc, n_s, n_c);
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

void compute_parents_flow_hmf(const cudaStream_t& dev, float** g_ind, const float* pt, const float* div, const float* u, const float icc, const int n_s, const int n_c){
    compute_parents_flow_hmf_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g_ind, pt, div, u, icc, n_s, n_c);
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

void update_flow_hmf(const cudaStream_t& dev, float** g_ind, float* g_s, float* g, float** ps_ind, float* ps, float* pt, const float* div, const float* u, const float icc, const int* p_c, const int s_c, const int n_s, const int n_c){
    update_flow_hmf_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g_ind, g_s, g, ps_ind, ps, pt, div, u, icc, p_c, s_c, n_s, n_c);
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

void divide_out_and_store_hmf(const cudaStream_t& dev, const float* g_s, const float* g, float* ps, float* pt, const int* p_c, const int s_c, const int n_s, const int n_c){
    divide_out_and_store_hmf_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g_s, g, ps, pt, p_c, s_c, n_s, n_c);
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

void update_source_sink_multiplier_potts(const cudaStream_t& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_c, const int n_s){
    update_source_sink_multiplier_potts_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(ps, pt, div, u, erru, d, cc, icc, n_c, n_s);
    if(CHECK_ERRORS) check_error(dev, "update_source_sink_multiplier_potts launch failed with error");
}

// update the source flow
__global__ void update_source_sink_multiplier_binary_kernel(float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_s){
    
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    //update source flow and constrain
	float div_t = (i < n_s) ? div[i] : 0.0f;
	float u_t = (i < n_s) ? u[i] : 0.0f;
	float d_t = (i < n_s) ? d[i] : 0.0f;
    float pt_t = (i < n_s) ? pt[i] : 0.0f;
    float ps_t = icc+pt_t+div_t-u_t*icc;
	ps_t = (d_t <= 0.0f) ? 0.0f : ps_t;
	ps_t = (ps_t > d_t && d_t > 0.0f) ? d_t : ps_t;
    if( i < n_s )
        ps[i] = ps_t;
    
    //update sink flow and constrain
    pt_t = ps_t-div_t+u_t*icc;
	pt_t = (d_t >= 0.0f) ? 0.0f : pt_t;
	pt_t = (pt_t > -d_t && d_t < 0.0f) ? -d_t : pt_t;
    if( i < n_s )
        pt[i] = pt_t;
        
	//update multiplier
	float erru_t = cc * (ps_t - div_t - pt_t);
	u_t += erru_t;
	erru_t = (erru_t < 0.0f) ? -erru_t : erru_t;
        
	//output
	if( i < n_s ){
		u[i] = u_t;
		erru[i] = erru_t;
	}
    
}

void update_source_sink_multiplier_binary(const cudaStream_t& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_s){
    update_source_sink_multiplier_binary_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(ps, pt, div, u, erru, d, cc, icc, n_s);
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

void update_multiplier_hmf(const cudaStream_t& dev, float* const* const ps_ind, const float* div, const float* pt, float* u, float* erru, const int n_s, const int n_c, const float cc){
    update_multiplier_hmf_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(ps_ind, pt,div,u,erru,cc, n_s, n_c);
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



void update_source_flows(const cudaStream_t& dev, float* ps, const float* pt, const float* div, const float* u, float icc, const int n_c, const int n_s){
    update_source_flows_kernel_channel_first<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(ps, pt, div, u, icc, n_c, n_s);
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


void update_sink_flows(const cudaStream_t& dev, const float* ps, float* pt, const float* div, const float* u, const float* d, float icc, const int n_c, const int n_s){
    update_sink_flows_kernel<<<((n_s*n_c+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(ps, pt, div, u, d, icc, n_c*n_s, n_s );
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


void update_multiplier(const cudaStream_t& dev, const float* ps, const float* pt, const float* div, float* u, float* erru, float cc, const int n_c, const int n_s){
    update_multiplier_kernel<<<((n_s*n_c+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(ps, pt, div, u, erru, cc, n_c*n_s, n_s );
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
void find_min_constraint(const cudaStream_t& dev, float* output, const float* neg_constraint, const int n_c, const int n_s){
    find_min_constraint_channel_first<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(output, neg_constraint, n_c, n_s );
    if(CHECK_ERRORS) check_error(dev, "find_min_constraint launch failed with error");
}


__global__ void rep_buffer_channel_last(const float* in, float* out, const int n_c, const int n_s) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float out_t = in[i];
    for(int c = 1; c < n_c; c++)
        if( i < n_s )
            out[i*n_c+c] = out_t;
}

//channel first is a series of memcpy's

void rep_buffer(const cudaStream_t& dev, const float* input, float* output, const int n_c, const int n_s){
    for(int c = 0; c < n_c; c++)
        cudaMemcpyAsync((void*)(output+c*n_s),input,n_s*sizeof(float),cudaMemcpyDeviceToDevice,dev);
    if(CHECK_ERRORS) check_error(dev, "rep_buffer launch failed with error");
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


__device__ void abs_constrain_device(const int i, float* b, const float* r, const int n_s){
    float value = (i < n_s) ? b[i] : 0.0f;
    float constraint = (i < n_s) ? r[i] : 0.0f;
    value = (value > constraint) ? constraint: value;
    value = (value < -constraint) ? -constraint: value;
    
    if(i < n_s)
        b[i] = value;
    
}

__global__ void abs_constrain_kernel(float* b, const float* r, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    abs_constrain_device(i, b, r, n_s);
}


void abs_constrain(const cudaStream_t& dev, float* buffer, const float* constrain, const int n_s){
    abs_constrain_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(buffer, constrain, n_s);
    if(CHECK_ERRORS) check_error(dev, "abs_constrain launch failed with error");
}

__global__ void max_neg_constrain_kernel(float* b, const float* r, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float value = (i < n_s) ? b[i] : 0.0f;
    float constraint = (i < n_s) ? r[i] : 0.0f;
    value = (value > -constraint) ? -constraint: value;
    
    if(i < n_s)
        b[i] = value;
    
}


void max_neg_constrain(const cudaStream_t& dev, float* buffer, const float* constrain, const int n_s){
    max_neg_constrain_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(buffer, constrain, n_s);
    if(CHECK_ERRORS) check_error(dev, "max_constrain launch failed with error");
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

__global__ void update_spatial_flows_3d_kernel(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float* rz, const int n_x, const int n_y, const int n_z, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    
    update_flows_kernel(i,x,y,z, g, px, py, pz, n_x, n_y, n_z, n_s);
    abs_constrain_device(i, px, rx, n_s);
    abs_constrain_device(i, py, ry, n_s);
    abs_constrain_device(i, pz, rz, n_s);
}

__global__ void update_spatial_flows_2d_kernel(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const int n_x, const int n_y, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    
    update_flows_kernel(i,x,y, g, px, py, n_x, n_y, n_s);
    abs_constrain_device(i, px, rx, n_s);
    abs_constrain_device(i, py, ry, n_s);
}

__global__ void update_spatial_flows_1d_kernel(const float* g, float* div, float* px, const float* rx, const int n_x, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int x = i_temp % n_x; i_temp /= n_x;
    
    update_flows_kernel(i, x, g, px, n_x, n_s);
    abs_constrain_device(i, px, rx, n_s);
}

void update_spatial_flows(const cudaStream_t& dev, const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float* rz, const int n_x, const int n_y, const int n_z, const int n_s){
    update_spatial_flows_3d_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, div, px, py, pz, rx, ry, rz, n_x, n_y, n_z, n_s);
    calc_divergence_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(div, px, py, pz, n_x, n_y, n_z, n_s);
    if(CHECK_ERRORS) check_error(dev, "update_spatial_flows_3d_kernel launch failed with error");
}
void update_spatial_flows(const cudaStream_t& dev, const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const int n_x, const int n_y, const int n_s){
    update_spatial_flows_2d_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, div, px, py, rx, ry, n_x, n_y, n_s);
    calc_divergence_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(div, px, py, n_x, n_y, n_s);
    if(CHECK_ERRORS) check_error(dev, "update_spatial_flows_2d_kernel launch failed with error");
}
void update_spatial_flows(const cudaStream_t& dev, const float* g, float* div, float* px, const float* rx, const int n_x, const int n_s){
    update_spatial_flows_1d_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, div, px, rx, n_x, n_s);
    calc_divergence_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(div, px, n_x, n_s);
    if(CHECK_ERRORS) check_error(dev, "update_spatial_flows_1d_kernel launch failed with error");
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

void calc_capacity_potts(const cudaStream_t& dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau){
    calc_capacity_potts_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, div, ps, pt, u, n_s, n_c, icc, tau);
    if(CHECK_ERRORS) check_error(dev, "calc_capacity_potts launch failed with error");
}

__global__ void calc_capacity_binary_kernel(float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const float icc, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float g_t = (i < n_s) ? div[i] + pt[i] - ps[i] - icc * u[i] : 0.0f;
    g_t *= tau;
	if(i < n_s)
		g[i] = g_t;
}

void calc_capacity_binary(const cudaStream_t& dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const float icc, const float tau){
	calc_capacity_binary_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, div, ps, pt, u, n_s, icc, tau);
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

void calc_capacity_potts_source_separate(const cudaStream_t& dev, float* g, const float* div, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau){
    calc_capacity_potts_source_separate_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, div, pt, u, n_s, n_c, icc, tau);
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

void calc_capacity_hmf(const cudaStream_t& dev, float* g, float* const* const ps_ind, const float* div, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau){
    calc_capacity_hmf_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, ps_ind, div, pt, u, n_s, n_c, icc, tau);
    if(CHECK_ERRORS) check_error(dev, "calc_capacity_hmf launch failed with error");
}

__global__ void log_buffer_kernel(const float* in, float* out, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float value = in[i];
	value = (value < epsilon) ? epsilon : value;
    value = log(value);
    
    if(i < n_s)
        out[i] = value;
    
}

void log_buffer(const cudaStream_t& dev, const float* in, float* out, const int n_s){
    log_buffer_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(in, out, n_s);
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


void binary_constrain(const cudaStream_t& dev, float* buffer, const int n_s){
    binary_constrain_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(buffer, n_s);
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



float mean_of_buffer(const cudaStream_t& dev, const float* buffer, const int n_s){
    float* buffer_c = (float*) malloc(n_s*sizeof(float));
    get_from_gpu(dev,buffer,buffer_c,n_s*sizeof(float));
	cudaStreamSynchronize(dev);
    double mean_value = 0.0;
    for(int s = 0; s < n_s; s++)
        mean_value += (buffer_c[s] < 0) ? -buffer_c[s] : buffer_c[s];
    free(buffer_c);
    return (float)(mean_value / (double) n_s);
}

float max_of_buffer(const cudaStream_t& dev, const float* buffer, const int n_s){
    float* buffer_c = (float*) malloc(n_s*sizeof(float));
    get_from_gpu(dev,buffer,buffer_c,n_s*sizeof(float));
	cudaStreamSynchronize(dev);
    float max_value = 0.0f;
    for(int s = 0; s < n_s; s++){
        if( max_value < buffer_c[s] )
            max_value = buffer_c[s];
        else if( max_value < -buffer_c[s] )
            max_value = -buffer_c[s];
    }
    free(buffer_c);
    return max_value;
}

float spat_max_of_buffer(const cudaStream_t& dev, const float* buffer, const int n_s, const int n_c){
    float* buffer_c = (float*) malloc(n_c*n_s*sizeof(float));
    get_from_gpu(dev,buffer,buffer_c,n_c*n_s*sizeof(float));
	cudaStreamSynchronize(dev);
    float max_value = 0.0f;
    float sum_value = 0.0f;
    for(int s = 0; s < n_s; s++){
        sum_value = 0.0f;
        for(int c = 0; c < n_c; c++)
            sum_value += (buffer_c[c*n_s+s] < 0) ? -buffer_c[c*n_s+s] : buffer_c[c*n_s+s];
        
        max_value = (max_value > sum_value) ? max_value : sum_value;
    }
    free(buffer_c);
    return max_value;
}

__global__ void copy_kernel(const float* source, float* dest, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = (i < n_s) ? source[i] : 0.0f;
    if(i < n_s)
        dest[i] = val;
}

void copy_buffer(const cudaStream_t& dev, const float* source, float* dest, const int n_s){
    copy_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(source, dest, n_s);
    if(CHECK_ERRORS) check_error(dev, "copy_buffer launch failed with error");
}


__global__ void copy_clean_kernel(const float* source, float* dest, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = (i < n_s) ? source[i] : 0.0f;
    val = isfinite(val) ? val: 0.0f;
    if(i < n_s)
        dest[i] = val;
}

void copy_buffer_clean(const cudaStream_t& dev, const float* source, float* dest, const int n_s){
    copy_clean_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(source, dest, n_s);
    if(CHECK_ERRORS) check_error(dev, "copy_buffer_clean launch failed with error");
}

__global__ void copy_clip_kernel(const float* source, float* dest, const int n_s, const float clip){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = (i < n_s) ? source[i] : 0.0f;
    val = isfinite(val) ? val: 0.0f;
    val = val < -clip ? -clip: val;
    val = val >  clip ?  clip: val;
    if(i < n_s)
        dest[i] = val;
}

void copy_buffer_clip(const cudaStream_t& dev, const float* source, float* dest, const int n_s, float clip){
    copy_clip_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(source, dest, n_s, clip);
    if(CHECK_ERRORS) check_error(dev, "copy_buffer_clip launch failed with error");
}

__global__ void inc_kernel(const float* inc, float* acc, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = (i < n_s) ? acc[i] : 0.0f;
    float increment = (i < n_s) ? inc[i] : 0.0f;
    val += increment;
    if(i < n_s)
        acc[i] = val;
}

void inc_buffer(const cudaStream_t& dev, const float* inc, float* acc, const int n_s){
    inc_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(inc, acc, n_s);
    if(CHECK_ERRORS) check_error(dev, "inc_buffer launch failed with error");
}

__global__ void inc_inc_minc_kernel(const float* inc1, const float* inc2, const float* minc, const float multi, float* acc, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = (i < n_s) ? acc[i] : 0.0f;
    float increment1 = (i < n_s) ? inc1[i] : 0.0f;
    float increment2 = (i < n_s) ? inc2[i] : 0.0f;
    float minincrement = (i < n_s) ? minc[i] : 0.0f;
    val += increment1 + increment2 + multi*minincrement;
    if(i < n_s)
        acc[i] = val;
}

void inc_inc_minc_buffer(const cudaStream_t& dev, const float* inc1, const float* inc2, const float* minc, const float multi, float* acc, const int n_s){
    inc_inc_minc_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(inc1, inc2, minc, multi, acc, n_s);
    if(CHECK_ERRORS) check_error(dev, "inc_inc_minc_buffer launch failed with error");
}

__global__ void ninc_kernel(const float* inc, float* acc, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = (i < n_s) ? acc[i] : 0.0f;
    float increment = (i < n_s) ? inc[i] : 0.0f;
    val -= increment;
    if(i < n_s)
        acc[i] = val;
}

void ninc_buffer(const cudaStream_t& dev, const float* inc, float* acc, const int n_s){
    ninc_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(inc, acc, n_s);
    if(CHECK_ERRORS) check_error(dev, "ninc_buffer launch failed with error");
}

__global__ void inc_mult_kernel(const float* inc, float* acc, const int n_s, const float multi){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = (i < n_s) ? acc[i] : 0.0f;
    float increment = multi * ((i < n_s) ? inc[i] : 0.0f);
    val += increment;
    if(i < n_s)
        acc[i] = val;
}

void inc_mult_buffer(const cudaStream_t& dev, const float* inc, float* acc, const int n_s, const float multi){
    inc_mult_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(inc, acc, n_s, multi);
    if(CHECK_ERRORS) check_error(dev, "inc_mult_buffer launch failed with error");
}

__global__ void inc2_mult_kernel(const float* inc_m1, const float* inc_m2, float* acc, const int n_s, const float multi){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = (i < n_s) ? acc[i] : 0.0f;
    float increment = multi*((i < n_s) ? inc_m1[i]*inc_m2[i] : 0.0f);
    val += increment;
    if(i < n_s)
        acc[i] = val;
}

void inc2_mult_buffer(const cudaStream_t& dev, const float* inc_m1, const float* inc_m2, float* acc, const int n_s, const float multi){
    inc2_mult_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(inc_m1, inc_m2, acc, n_s, multi);
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

void div_buffer(const cudaStream_t& dev, const float* div, float* res, const int n_s){
    div_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(div, res, n_s);
    if(CHECK_ERRORS) check_error(dev, "div_buffer launch failed with error");
}

__global__ void div_kernel(const float divisor, float* res, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = res[i];
    val /= divisor;
    if(i < n_s)
        res[i] = val;
}

void div_buffer(const cudaStream_t& dev, const float divisor, float* res, const int n_s){
    div_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(divisor, res, n_s);
    if(CHECK_ERRORS) check_error(dev, "div_buffer launch failed with error");
}

__global__ void mult_kernel(const float mult, float* res, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = res[i]*mult;
    if(i < n_s)
        res[i] = val;
}

void mult_buffer(const cudaStream_t& dev, const float mult, float* res, const int n_s){
    mult_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(mult, res, n_s);
    if(CHECK_ERRORS) check_error(dev, "mult_buffer launch failed with error");
}

__global__ void mult_kernel2(const float mult, const float* input, float* res, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = input[i]*mult;
    if(i < n_s)
        res[i] = val;
}

void mult_buffer(const cudaStream_t& dev, const float mult, const float* input, float* res, const int n_s){
    mult_kernel2<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(mult, input, res, n_s);
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

void exp_and_inc_buffer(const cudaStream_t& dev, const float* max, float* cost, float* acc, const int n_s){
    exp_and_inc_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(max, cost, acc, n_s);
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

void add_store_then_max_buffer(const cudaStream_t& dev, const float* comp1, const float* comp2, float* store, float* res, const int n_s){
    add_store_then_max_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(comp1, comp2, store, res, n_s);
    if(CHECK_ERRORS) check_error(dev, "add_store_then_max_buffer launch failed with error");
}

__global__ void softmax_kernel(const float* e1, float* u, const int n_s, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    //find maxmum cost
    float max_cost = (i < n_c*n_s) ? e1[i] : -1.0/0.0f;
    for(int c = 1; c < n_c; c++){
        float cost = (c*n_s+i < n_c*n_s) ? e1[c*n_s+i] : 0.0f;
        if(cost > max_cost)
            max_cost = cost;
    }
    
    //find accumulator
    float accum = 0.0f;
    for(int c = 0; c < n_c; c++)
        accum += exp( ((c*n_s+i < n_c*n_s) ? e1[c*n_s+i] : 0.0f) -max_cost);
    
    //apply and normalize
    for(int c = 0; c < n_c; c++){
        float value = exp( ((c*n_s+i < n_c*n_s) ? e1[c*n_s+i] : 0.0f) -max_cost) / accum;
        if(i < n_s)
            u[c*n_s+i] = value;
    }
}

__global__ void softmax_kernel(const float* e1, const float* e2, float* u, const int n_s, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    //find maxmum cost
    float max_cost = (i < n_c*n_s) ?  (e1[i]+e2[i]) : -1.0/0.0f;
    for(int c = 1; c < n_c; c++){
        float cost = (c*n_s+i < n_c*n_s) ? (e1[c*n_s+i]+e2[c*n_s+i]) : 0.0f;
        if(cost > max_cost)
            max_cost = cost;
    }
    
    //find accumulator
    float accum = 0.0f;
    for(int c = 0; c < n_c; c++)
        accum += exp(((c*n_s+i < n_c*n_s) ? (e1[c*n_s+i]+e2[c*n_s+i]) : 0.0f)-max_cost);
    
    //apply and normalize
    for(int c = 0; c < n_c; c++){
        float value = exp(((c*n_s+i < n_c*n_s) ? (e1[c*n_s+i]+e2[c*n_s+i]) : 0.0f)-max_cost) / accum;
        //value = (value < epsilon) ? epsilon : value;
        //value = (value > 1.0f-epsilon) ? 1.0f-epsilon : value;
        if(i < n_s)
            u[c*n_s+i] = value;
    }
}

void softmax(const cudaStream_t& dev, const float* e1, const float* e2, float* u, const int n_s, const int n_c){
    if(e2 != NULL)
        softmax_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(e1, e2, u, n_s, n_c);
    else
        softmax_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(e1, u, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "softmax launch failed with error");
}



__global__ void neg_softmax_kernel(const float* e, float* u, const int n_s, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    //find maxmum cost
    float max_cost = e[i];
    for(int c = 1; c < n_c; c++){
        float cost = e[c*n_s+i];
        if(cost > max_cost)
            max_cost = cost;
    }
    
    //find accumulator
    float accum = 0.0f;
    for(int c = 0; c < n_c; c++)
        accum += exp(e[c*n_s+i]-max_cost);
    
    //apply and normalize
    for(int c = 0; c < n_c; c++){
        float value = exp(e[c*n_s+i]-max_cost) / accum;
        //value = (value < epsilon) ? epsilon : value;
        //value = (value > 1.0f-epsilon) ? 1.0f-epsilon : value;
        if(i < n_s)
            u[c*n_s+i] = value;
    }
}

void neg_softmax(const cudaStream_t& dev, const float* e, float* u, const int n_s, const int n_c){
    neg_softmax_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(e, u, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "neg_softmax launch failed with error");
}


__global__ void sigmoid_kernel(const float* e1, const float* e2, float* u, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float cost = ((i < n_s) ? e1[i] : 0.0f) + ((i < n_s) ? e2[i] : 0.0f);
	float value = 1.0f / (1.0f + exp(-cost));
	if(i < n_s)
		u[i] = value;
}

__global__ void sigmoid_kernel(const float* e1, float* u, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float cost = (i < n_s) ? e1[i] : 0.0f;
	float value = 1.0f / (1.0f + exp(-cost));
	if(i < n_s)
		u[i] = value;
}

void sigmoid(const cudaStream_t& dev, const float* e1, const float* e2, float* u, const int n_s){
    if(e2 != NULL)
        sigmoid_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(e1, e2, u, n_s);
    else
        sigmoid_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(e1, u, n_s);
    if(CHECK_ERRORS) check_error(dev, "sigmoid launch failed with error");
}


__global__ void exp_kernel(const float* e1, float* u, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float cost = e1[i];
	float value = exp(cost);
	if(i < n_s)
		u[i] = value;
}

void exp(const cudaStream_t& dev, const float* e1, float* u, const int n_s){
    exp_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(e1, u, n_s);
    if(CHECK_ERRORS) check_error(dev, "sigmoid launch failed with error");
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
        float up_contra = (c*n_s+i+1 < n_c*n_s) ? (2.0f*u[c*n_s+i+1]-1.0f) * g[c*n_s+i] : 0.0f;
        float dn_contra = (c*n_s+i+1 < n_c*n_s) ? (2.0f*u[c*n_s+i]-1.0f) * g[c*n_s+i+1] : 0.0f;
        float derivative = (z < n_z-1) ? up_contra + dn_contra : 0.0f;
        if(i < n_s)
            g_rz[c*n_s+i] = derivative;
        
        //for y
        up_contra = (c*n_s+i+n_z < n_c*n_s) ? (2.0f*u[c*n_s+i+n_z]-1.0f) * g[c*n_s+i] : 0.0f;
        dn_contra = (c*n_s+i+n_z < n_c*n_s) ? (2.0f*u[c*n_s+i] -1.0f)* g[c*n_s+i+n_z] : 0.0f;
        derivative = (y < n_y-1) ? up_contra + dn_contra : 0.0f;
        if(i < n_s)
            g_ry[c*n_s+i] = derivative;
        
        //for x
        up_contra = (c*n_s+i+n_y*n_z < n_c*n_s) ? (2.0f*u[c*n_s+i+n_z*n_y]-1.0f) * g[c*n_s+i] : 0.0f;
        dn_contra = (c*n_s+i+n_y*n_z < n_c*n_s) ? (2.0f*u[c*n_s+i]-1.0f) * g[c*n_s+i+n_z*n_y] : 0.0f;
        derivative = (x < n_x-1) ? up_contra + dn_contra: 0.0f;
        if(i < n_s)
            g_rx[c*n_s+i] = derivative;
    }
}

__global__ void populate_reg_mean_gradients_kernel(const float* g, const float* u, float* g_rx, float* g_ry, const int n_x, const int n_y, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    int n_s = n_x*n_y;
    for(int c = 0; c < n_c; c++){
        //for y
        float up_contra = (c*n_s+i+1 < n_c*n_s) ? (2.0f*u[c*n_s+i+1]-1.0f) * g[c*n_s+i] : 0.0f;
        float dn_contra = (c*n_s+i+1 < n_c*n_s) ? (2.0f*u[c*n_s+i]-1.0f) * g[c*n_s+i+1] : 0.0f;
        float derivative = (y < n_y-1) ? up_contra + dn_contra : 0.0f;
        if(i < n_s)
            g_ry[c*n_s+i] = derivative;
        
        //for x
        up_contra = (c*n_s+i+n_y < n_c*n_s) ? (2.0f*u[c*n_s+i+n_y]-1.0f) * g[c*n_s+i] : 0.0f;
        dn_contra = (c*n_s+i+n_y < n_c*n_s) ? (2.0f*u[c*n_s+i]-1.0f) * g[c*n_s+i+n_y] : 0.0f;
        derivative = (x < n_x-1) ? up_contra + dn_contra: 0.0f;
        if(i < n_s)
            g_rx[c*n_s+i] = derivative;
    }
}

__global__ void populate_reg_mean_gradients_kernel(const float* g, const float* u, float* g_rx, const int n_x, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int x = i_temp % n_x; i_temp /= n_x;
    int n_s = n_x;
    for(int c = 0; c < n_c; c++){
        //for x
        float up_contra = (c*n_s+i+1 < n_c*n_s) ? (2.0f*u[c*n_s+i+1]-1.0f) * g[c*n_s+i] : 0.0f;
        float dn_contra = (c*n_s+i+1 < n_c*n_s) ? (2.0f*u[c*n_s+i]-1.0f) * g[c*n_s+i+1] : 0.0f;
        float derivative = (x < n_x-1) ? up_contra + dn_contra: 0.0f;
        if(i < n_s)
            g_rx[c*n_s+i] = derivative;
    }
}


void populate_reg_mean_gradients(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c){
    int n_s = n_x*n_y*n_z;
    populate_reg_mean_gradients_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_c);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_gradient launch failed with error");
}
void populate_reg_mean_gradients(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, float* g_ry, const int n_x, const int n_y, const int n_c){
    int n_s = n_x*n_y;
    populate_reg_mean_gradients_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, u, g_rx, g_ry, n_x, n_y, n_c);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_gradient launch failed with error");
}
void populate_reg_mean_gradients(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, const int n_x, const int n_c){
    int n_s = n_x;
    populate_reg_mean_gradients_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, u, g_rx, n_x, n_c);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_gradient launch failed with error");
}


__global__ void populate_reg_mean_gradients_and_add_kernel(const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    int n_s = n_x*n_y*n_z;
    for(int c = 0; c < n_c; c++){
        //for z
        float up_contra = (c*n_s+i+1 < n_c*n_s) ? (2.0f*u[c*n_s+i+1]-1.0f) * g[c*n_s+i] : 0.0f;
        float dn_contra = (c*n_s+i+1 < n_c*n_s) ? (2.0f*u[c*n_s+i]-1.0f) * g[c*n_s+i+1] : 0.0f;
        float derivative = ((c*n_s+i < n_c*n_s) ? g_rz[c*n_s+i] : 0.0f) + tau*(( (z < n_z-1) ? up_contra + dn_contra : 0.0f));
        if(i < n_s)
            g_rz[c*n_s+i] = derivative;
        
        //for y
        up_contra = (c*n_s+i+n_z < n_c*n_s) ? (2.0f*u[c*n_s+i+n_z]-1.0f) * g[c*n_s+i] : 0.0f;
        dn_contra = (c*n_s+i+n_z < n_c*n_s) ? (2.0f*u[c*n_s+i]-1.0f) * g[c*n_s+i+n_z] : 0.0f;
        derivative = ((c*n_s+i < n_c*n_s) ? g_ry[c*n_s+i] : 0.0f) + tau*(( (y < n_y-1) ? up_contra + dn_contra : 0.0f));
        if(i < n_s)
            g_ry[c*n_s+i] = derivative;
        
        //for x
        up_contra = (c*n_s+i+n_y*n_z < n_c*n_s) ? (2.0f*u[c*n_s+i+n_z*n_y]-1.0f) * g[c*n_s+i] : 0.0f;
        dn_contra = (c*n_s+i+n_y*n_z < n_c*n_s) ? (2.0f*u[c*n_s+i]-1.0f) * g[c*n_s+i+n_z*n_y] : 0.0f;
        derivative = ((c*n_s+i < n_c*n_s) ? g_rx[c*n_s+i] : 0.0f) + tau*(( (x < n_x-1) ? up_contra + dn_contra: 0.0f));
        if(i < n_s)
            g_rx[c*n_s+i] = derivative;
    }
    
}
__global__ void populate_reg_mean_gradients_and_add_kernel(const float* g, const float* u, float* g_rx, float* g_ry, const int n_x, const int n_y, const int n_c, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x; i_temp /= n_x;
    int n_s = n_x*n_y;
    for(int c = 0; c < n_c; c++){
        
        //for y
        float up_contra = (c*n_s+i+1 < n_c*n_s) ? (2.0f*u[c*n_s+i+1]-1.0f) * g[c*n_s+i] : 0.0f;
        float dn_contra = (c*n_s+i+1 < n_c*n_s) ? (2.0f*u[c*n_s+i]-1.0f) * g[c*n_s+i+1] : 0.0f;
        float derivative = ((c*n_s+i < n_c*n_s) ? g_ry[c*n_s+i] : 0.0f) + tau*(( (y < n_y-1) ? up_contra + dn_contra : 0.0f));
        if(i < n_s)
            g_ry[c*n_s+i] = derivative;
        
        //for x
        up_contra = (c*n_s+i+n_y < n_c*n_s) ? (2.0f*u[c*n_s+i+n_y]-1.0f) * g[c*n_s+i] : 0.0f;
        dn_contra = (c*n_s+i+n_y < n_c*n_s) ? (2.0f*u[c*n_s+i]-1.0f) * g[c*n_s+i+n_y] : 0.0f;
        derivative = ((c*n_s+i < n_c*n_s) ? g_rx[c*n_s+i] : 0.0f) + tau*(( (x < n_x-1) ? up_contra + dn_contra: 0.0f));
        if(i < n_s)
            g_rx[c*n_s+i] = derivative;
    }
    
}
__global__ void populate_reg_mean_gradients_and_add_kernel(const float* g, const float* u, float* g_rx, const int n_x, const int n_c, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int x = i_temp % n_x; i_temp /= n_x;
    int n_s = n_x;
    for(int c = 0; c < n_c; c++){
        //for x
        float up_contra = (c*n_s+i+1 < n_c*n_s) ? (2.0f*u[c*n_s+i+1]-1.0f) * g[c*n_s+i] : 0.0f;
        float dn_contra = (c*n_s+i+1 < n_c*n_s) ? (2.0f*u[c*n_s+i]-1.0f) * g[c*n_s+i+1] : 0.0f;
        float derivative = ((c*n_s+i < n_c*n_s) ? g_rx[c*n_s+i] : 0.0f) + tau*(( (x < n_x-1) ? up_contra + dn_contra: 0.0f));
        if(i < n_s)
            g_rx[c*n_s+i] = derivative;
    }
    
}


void populate_reg_mean_gradients_and_add(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
    int n_s = n_x*n_y*n_z;
    populate_reg_mean_gradients_and_add_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_c, tau);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_mean_gradients_and_add launch failed with error");
}
void populate_reg_mean_gradients_and_add(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, float* g_ry, const int n_x, const int n_y, const int n_c, const float tau){
    int n_s = n_x*n_y;
    populate_reg_mean_gradients_and_add_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, u, g_rx, g_ry, n_x, n_y, n_c, tau);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_mean_gradients_and_add launch failed with error");
}
void populate_reg_mean_gradients_and_add(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, const int n_x, const int n_c, const float tau){
    int n_s = n_x;
    populate_reg_mean_gradients_and_add_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(g, u, g_rx, n_x, n_c, tau);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_mean_gradients_and_add launch failed with error");
}

__global__ void change_to_diff_kernel(float* t, float* d, const int n_s, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float old_val = (i < n_s) ? t[i] : 0.0f;
    float new_val = (i < n_s) ? d[i] : 0.0f;
    float diff = tau*(new_val-old_val);
    new_val = old_val + diff;
    if(i < n_s){
        t[i] = new_val;
        d[i] = diff;
    }
}

void change_to_diff(const cudaStream_t& dev, float* transfer, float* diff, const int n_s, const float tau){
    change_to_diff_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(transfer, diff, n_s, tau);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_gradient launch failed with error");
}

__global__ void parity_mask_kernel(float* buffer, const int n_x, const int n_y, const int n_z, const int n_c, const int parity){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x;
    int n_s = n_x*n_y*n_z*n_c;
    
    float value = (i < n_s) ? buffer[i] : 0.0f;
    value *= (parity ^ x ^ y ^ z) & 1;
    if(i < n_s)
        buffer[i] = value;
}

__global__ void parity_mask_kernel(float* buffer, const int n_x, const int n_y, const int n_c, const int parity){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x;
    int n_s = n_x*n_y*n_c;
    
    float value = (i < n_s) ? buffer[i] : 0.0f;
    value *= (parity ^ x ^ y) & 1;
    if(i < n_s)
        buffer[i] = value;
}

__global__ void parity_mask_kernel(float* buffer, const int n_x, const int n_c, const int parity){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int x = i_temp % n_x;
    int n_s = n_x*n_c;
    
    float value = (i < n_s) ? buffer[i] : 0.0f;
    value *= (parity ^ x) & 1;
    if(i < n_s)
        buffer[i] = value;
}

__global__ void parity_mask_kernel(float* buffer, const float* other, const int n_x, const int n_y, const int n_z, const int n_c, const int parity){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x;
    int n_s = n_x*n_y*n_z*n_c;
    
    float value = (i < n_s) ? buffer[i] : 0.0f;
    float other_value =  (i < n_s) ? other[i] : 0.0f;
    value *= (parity ^ x ^ y ^ z) & 1;
    value += other_value * ((parity ^ x ^ y ^ z ^ 1) & 1);
    if(i < n_s)
        buffer[i] = value;
}

__global__ void parity_mask_kernel(float* buffer, const float* other, const int n_x, const int n_y, const int n_c, const int parity){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x;
    int n_s = n_x*n_y*n_c;
    
    float value = (i < n_s) ? buffer[i] : 0.0f;
    float other_value =  (i < n_s) ? other[i] : 0.0f;
    value *= (parity ^ x ^ y) & 1;
    value += other_value * ((parity ^ x ^ y ^1) & 1);
    if(i < n_s)
        buffer[i] = value;
}

__global__ void parity_mask_kernel(float* buffer, const float* other, const int n_x, const int n_c, const int parity){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int x = i_temp % n_x;
    int n_s = n_x*n_c;
    
    float value = (i < n_s) ? buffer[i] : 0.0f;
    float other_value =  (i < n_s) ? other[i] : 0.0f;
    value *= (parity ^ x) & 1;
    value += other_value * ((parity ^ x ^ 1) & 1);
    if(i < n_s)
        buffer[i] = value;
}

void parity_mask(const cudaStream_t& dev, float* buffer, const int n_x, const int n_y, const int n_z, const int n_c, const int parity){
    int n_s = n_x*n_y*n_z*n_c;
    parity_mask_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(buffer, n_x, n_y, n_z, n_c, parity);
    if(CHECK_ERRORS) check_error(dev, "parity_mask (3D) launch failed with error");
}
void parity_mask(const cudaStream_t& dev, float* buffer, const int n_x, const int n_y, const int n_c, const int parity){
    int n_s = n_x*n_y*n_c;
    parity_mask_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(buffer, n_x, n_y, n_c, parity);
    if(CHECK_ERRORS) check_error(dev, "parity_mask (2D) launch failed with error");
}
void parity_mask(const cudaStream_t& dev, float* buffer, const int n_x, const int n_c, const int parity){
    int n_s = n_x*n_c;
    parity_mask_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(buffer, n_x, n_c, parity);
    if(CHECK_ERRORS) check_error(dev, "parity_mask (1D) launch failed with error");
}
void parity_mask(const cudaStream_t& dev, float* buffer, const float* other, const int n_x, const int n_y, const int n_z, const int n_c, const int parity){
    int n_s = n_x*n_y*n_z*n_c;
    parity_mask_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(buffer, other, n_x, n_y, n_z, n_c, parity);
    if(CHECK_ERRORS) check_error(dev, "parity_mask (3D -merge) launch failed with error");
}
void parity_mask(const cudaStream_t& dev, float* buffer, const float* other, const int n_x, const int n_y, const int n_c, const int parity){
    int n_s = n_x*n_y*n_c;
    parity_mask_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(buffer, other, n_x, n_y, n_c, parity);
    if(CHECK_ERRORS) check_error(dev, "parity_mask (2D -merge) launch failed with error");
}
void parity_mask(const cudaStream_t& dev, float* buffer, const float* other, const int n_x, const int n_c, const int parity){
    int n_s = n_x*n_c;
    parity_mask_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(buffer, other, n_x, n_c, parity);
    if(CHECK_ERRORS) check_error(dev, "parity_mask (1D -merge) launch failed with error");
}

__device__ float get_effective_reg_kernel_up(const float* u, const float* r, const int c, const int n_c, const int n_s, const int i, const int a, const int d, const int n_d){
        float ut = (c*n_s+i+a < n_c*n_s) ? u[c*n_s+i+a] : 0.0f;
        ut = 2.0f*ut -1.0f;
        float rt = (c*n_s+i < n_c*n_s) ? r[c*n_s+i] : 0.0f;
        return (d < n_d-1) ? ut*rt : 0.0f;
}
__device__ float get_effective_reg_kernel_dn(const float* u, const float* r, const int c, const int n_c, const int n_s, const int i, const int a, const int d){
        float ut = (c*n_s+i-a >= 0 && c*n_s+i-a < n_c*n_s) ? u[c*n_s+i-a] : 0.0f;
        ut = 2.0f*ut -1.0f;
        float rt = (c*n_s+i-a >= 0 && c*n_s+i-a < n_c*n_s) ? r[c*n_s+i-a] : 0.0f;
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
        reg_tot += get_effective_reg_kernel_up(u, rz, c, n_c, n_s, i, 1, z, n_z);
        reg_tot += get_effective_reg_kernel_dn(u, rz, c, n_c, n_s, i, 1, z);

        //y direction
        reg_tot += get_effective_reg_kernel_up(u, ry, c, n_c, n_s, i, n_z, y, n_y);
        reg_tot += get_effective_reg_kernel_dn(u, ry, c, n_c, n_s, i, n_z, y);

        //x direction
        reg_tot += get_effective_reg_kernel_up(u, rx, c, n_c, n_s, i, n_z*n_y, x, n_x);
        reg_tot += get_effective_reg_kernel_dn(u, rx, c, n_c, n_s, i, n_z*n_y, x);
        
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
        reg_tot += get_effective_reg_kernel_up(u, ry, c, n_c, n_s, i, 1, y, n_y);
        reg_tot += get_effective_reg_kernel_dn(u, ry, c, n_c, n_s, i, 1, y);

        //x direction
        reg_tot += get_effective_reg_kernel_up(u, rx, c, n_c, n_s, i, n_y, x, n_x);
        reg_tot += get_effective_reg_kernel_dn(u, rx, c, n_c, n_s, i, n_y, x);
        
        if(i < n_s)
            r_eff[c*n_s+i] = reg_tot;
    }
}

__global__ void get_effective_reg_kernel(float* r_eff, const float* u, const float* rx, const int n_x, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    for(int c = 0; c < n_c; c++){
        float reg_tot = 0.0f;
        
        //x direction
        reg_tot += get_effective_reg_kernel_up(u, rx, c, n_c, n_x, i, 1, i, n_x);
        reg_tot += get_effective_reg_kernel_dn(u, rx, c, n_c, n_x, i, 1, i);
        
        if(i < n_x)
            r_eff[c*n_x+i] = reg_tot;
    }
}

void get_effective_reg(const cudaStream_t& dev, float* r_eff, const float* u, const float* rx, const float* ry, const float* rz, const int n_x, const int n_y, const int n_z, const int n_c){
    int n_s = n_x*n_y*n_z;
    get_effective_reg_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(r_eff, u, rx, ry, rz, n_x, n_y, n_z, n_c);
    if(CHECK_ERRORS) check_error(dev, "get_effective_reg (3D) launch failed with error");
}
void get_effective_reg(const cudaStream_t& dev, float* r_eff, const float* u, const float* rx, const float* ry, const int n_x, const int n_y, const int n_c){
    int n_s = n_x*n_y;
    get_effective_reg_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(r_eff, u, rx, ry, n_x, n_y, n_c);
    if(CHECK_ERRORS) check_error(dev, "get_effective_reg (2D) launch failed with error");
}
void get_effective_reg(const cudaStream_t& dev, float* r_eff, const float* u, const float* rx, const int n_x, const int n_c){
    get_effective_reg_kernel<<<((n_x+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(r_eff, u, rx, n_x, n_c);
    if(CHECK_ERRORS) check_error(dev, "get_effective_reg (1D) launch failed with error");
}


__global__ void add_then_store_kernel(const float* addend1, const float* addend2, float* sum, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float a1 = (i < n_s) ? addend1[i] : 0.0f;
    float a2 = (i < n_s) ? addend2[i] : 0.0f;
    float res = a1+a2;
    if(i < n_s)
        sum[i] = res;   
}

void add_then_store(const cudaStream_t& dev, const float* addend1, const float* addend2, float* sum, const int n_s){
    add_then_store_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(addend1, addend2, sum, n_s);
    if(CHECK_ERRORS) check_error(dev, "add_then_store launch failed with error");
}

__global__ void add_then_store_kernel_2(const float* addend1, const float* addend2, float* sum1, float* sum2, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float a1 = (i < n_s) ? addend1[i] : 0.0f;
    float a2 = (i < n_s) ? addend2[i] : 0.0f;
    float res = a1+a2;
    if(i < n_s){
        sum1[i] = res;   
        sum2[i] = res;
    }
}

void add_then_store_2(const cudaStream_t& dev, const float* addend1, const float* addend2, float* sum1, float* sum2, const int n_s){
    add_then_store_kernel_2<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(addend1, addend2, sum1, sum2, n_s);
    if(CHECK_ERRORS) check_error(dev, "add_then_store_2 launch failed with error");
}

__global__ void untangle_softmax_kernel(const float* du_i, const float* u, float* loc, const int n_s, const int n_c){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    for(int c = 0; c < n_c; c++){
        float new_grad = 0.0f;
        float uc = (c*n_s+i < n_s*n_c) ? u[c*n_s+i] : 0.0f;
        for(int a = 0; a < n_c; a++){
            float da = (a*n_s+i < n_s*n_c) ? du_i[a*n_s+i] : 0.0f;
            float dua = (a*n_s+i < n_s*n_c) ? u[a*n_s+i] : 0.0f;
            if(c == a)
                new_grad += da*(1.0f-uc);
            else
                new_grad -= da*dua;
        }
        new_grad *= uc;
        if( i < n_s )
            loc[c*n_s+i] = new_grad;
    }
    
}

void untangle_softmax(const cudaStream_t& dev, const float* du_i, const float* u, float* loc, const int n_s, const int n_c){
    untangle_softmax_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(du_i, u, loc, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "untangle_softmax launch failed with error");
}



__global__ void untangle_sigmoid_kernel(const float* du_i, const float* u, float* loc, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float du_t = (i < n_s) ? du_i[i] : 0.0f;
    float u_t = (i < n_s) ? u[i] : 0.0f;
	float new_grad = du_t*u_t*(1.0f-u_t);
	if( i < n_s )
        loc[i] = new_grad;
    
}

void untangle_sigmoid(const cudaStream_t& dev, const float* du_i, const float* u, float* loc, const int n_s){
    untangle_sigmoid_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(du_i, u, loc, n_s);
    if(CHECK_ERRORS) check_error(dev, "process_grad_binary launch failed with error");
}

__device__ float get_gradient_for_u_kernel_dn(const float* dy, const float* r, const int c, const int n_c, const int n_s, const int i, const int a, const int d){
    float multiplier = (c*n_s+i-a >= 0 && c*n_s+i-a < n_c*n_s) ? dy[c*n_s+i-a] : 0.0f;
    float inc = (c*n_s+i-a >= 0 && c*n_s+i-a < n_c*n_s) ? r[c*n_s+i-a] : 0.0f;
    inc = 2.0f * multiplier * inc;
    return (d > 0) ? inc: 0.0f;
}
__device__ float get_gradient_for_u_kernel_up(const float* dy, const float* r, const int c, const int n_c, const int n_s, const int i, const int a, const int d, const int n_d){
    float multiplier = (c*n_s+i+a < n_c*n_s) ? dy[c*n_s+i+a] : 0.0f;
    float inc = (c*n_s+i < n_c*n_s) ? r[c*n_s+i] : 0.0f;
    inc = 2.0f * multiplier * inc;
    return (d < n_d-1) ? inc: 0.0f;
}

__global__ void get_gradient_for_u_kernel(const float* dy, float* du, const float* rx, const float* ry, const float* rz, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int z = i_temp % n_z; i_temp /= n_z;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x;
    int n_s = n_x*n_y*n_z;
    
    for(int c = 0; c < n_c; c++){
        float grad_val = 0.0f;
            
        grad_val += get_gradient_for_u_kernel_dn(dy, rz, c, n_c, n_s, i, 1, z);
        grad_val += get_gradient_for_u_kernel_up(dy, rz, c, n_c, n_s, i, 1, z, n_z);
        
        grad_val += get_gradient_for_u_kernel_dn(dy, ry, c, n_c, n_s, i, n_z, y);
        grad_val += get_gradient_for_u_kernel_up(dy, ry, c, n_c, n_s, i, n_z, y, n_y);
        
        grad_val += get_gradient_for_u_kernel_dn(dy, rx, c, n_c, n_s, i, n_z*n_y, x);
        grad_val += get_gradient_for_u_kernel_up(dy, rx, c, n_c, n_s, i, n_z*n_y, x, n_x);
        
		float old_grad = (c*n_s+i < n_c*n_s) ? du[c*n_s+i] : 0.0f;
		float new_grad = tau*grad_val + (1-tau)*old_grad;
        if(i < n_s)
            du[c*n_s+i] = new_grad;
    }
}

__global__ void get_gradient_for_u_kernel(const float* dy, float* du, const float* rx, const float* ry, const int n_x, const int n_y, const int n_c, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    int i_temp = i;
    int y = i_temp % n_y; i_temp /= n_y;
    int x = i_temp % n_x;
    int n_s = n_x*n_y;
    
    for(int c = 0; c < n_c; c++){
        float grad_val = 0.0f;
        
        grad_val += get_gradient_for_u_kernel_dn(dy, ry, c, n_c, n_s, i, 1, y);
        grad_val += get_gradient_for_u_kernel_up(dy, ry, c, n_c, n_s, i, 1, y, n_y);
        
        grad_val += get_gradient_for_u_kernel_dn(dy, rx, c, n_c, n_s, i, n_y, x);
        grad_val += get_gradient_for_u_kernel_up(dy, rx, c, n_c, n_s, i, n_y, x, n_x);
        
		float old_grad = (c*n_s+i < n_c*n_s) ? du[c*n_s+i] : 0.0f;
		float new_grad = tau*grad_val + (1-tau)*old_grad;
        if(i < n_s)
            du[c*n_s+i] = new_grad;
    }
}

__global__ void get_gradient_for_u_kernel(const float* dy, float* du, const float* rx, const int n_x, const int n_c, const float tau){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    for(int c = 0; c < n_c; c++){
        float grad_val = 0.0f;
        
        grad_val += get_gradient_for_u_kernel_dn(dy, rx, c, n_c, n_x, i, 1, i);
        grad_val += get_gradient_for_u_kernel_up(dy, rx, c, n_c, n_x, i, 1, i, n_x);
        
		float old_grad = (c*n_x+i < n_c*n_x) ? du[c*n_x+i] : 0.0f;
		float new_grad = tau*grad_val + (1-tau)*old_grad;
        if(i < n_x)
            du[c*n_x+i] = new_grad;
    }
}

void get_gradient_for_u(const cudaStream_t& dev, const float* dy, float* du, const float* rx, const float* ry, const float* rz, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
    int n_s = n_x*n_y*n_z;
    get_gradient_for_u_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(dy, du, rx, ry, rz, n_x, n_y, n_z, n_c, tau);
    if(CHECK_ERRORS) check_error(dev, "get_gradient_for_u (3D) launch failed with error");
}
void get_gradient_for_u(const cudaStream_t& dev, const float* dy, float* du, const float* rx, const float* ry, const int n_x, const int n_y, const int n_c, const float tau){
    int n_s = n_x*n_y;
    get_gradient_for_u_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(dy, du, rx, ry, n_x, n_y, n_c, tau);
    if(CHECK_ERRORS) check_error(dev, "get_gradient_for_u (2D) launch failed with error");
}
void get_gradient_for_u(const cudaStream_t& dev, const float* dy, float* du, const float* rx, const int n_x, const int n_c, const float tau){
    get_gradient_for_u_kernel<<<((n_x+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(dy, du, rx, n_x, n_c, tau);
    if(CHECK_ERRORS) check_error(dev, "get_gradient_for_u (1D) launch failed with error");
}

__global__ void taylor_series_channels_first_kernel(const float* input, const float* coeffs, float* output, int n_b, int n_s, int n_c, int n_i){
    __shared__ float coeffs_shr [NUM_THREADS];
	int p = blockIdx.x * NUM_THREADS + threadIdx.x;
    
	for(int c = 0; c < n_c; c++){
		
		//load coeffs in as shared data
		coeffs_shr[threadIdx.x] = coeffs[c*n_i+threadIdx.x];
		__syncthreads();
		
		for(int b = 0; b < n_b; b++){
			float output_t = coeffs_shr[0];
			float x = input[(b*n_c+c)*n_s+p];
			float poly = 1.0f;
			for(int i = 1; i < n_i; i++){
				poly *= x / (float) i;
				output_t += poly * coeffs_shr[i];
			}
			if(p < n_s)
				output[(b*n_c+c)*n_s+p] = output_t;
		}
	}
}

__global__ void taylor_series_channels_last_kernel(const float* input, const float* coeffs, float* output, int n_b, int n_s, int n_c, int n_i){
    __shared__ float coeffs_shr [NUM_THREADS];
	int p = blockIdx.x * NUM_THREADS + threadIdx.x;
	int c = p % n_c;
    coeffs_shr[threadIdx.x] = coeffs[threadIdx.x];
	
	for(int b = 0; b < n_b; b++){
		float output_t = coeffs_shr[c*n_i];
		float x = input[b*n_s*n_c+p];
		float poly = 1.0f;
		for(int i = 1; i < n_i; i++){
			poly *= x / (float) i;
			output_t += poly * coeffs_shr[c*n_i+i];
		}
		if(p < n_s*n_c)
			output[b*n_s*n_c+p] = output_t;
	}
}
void taylor_series_channels_first(const cudaStream_t& dev, const float* input, const float* coeffs, float* output, int n_b, int n_s, int n_c, int n_i){
	taylor_series_channels_first_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(input, coeffs, output, n_b, n_s, n_c, n_i);
    if(CHECK_ERRORS) check_error(dev, "taylor_series_channels_first launch failed with error");
}

void taylor_series_channels_last(const cudaStream_t& dev, const float* input, const float* coeffs, float* output, int n_b, int n_s, int n_c, int n_i){
	if( n_c * n_i > NUM_THREADS ){
		printf("Too many coeffs to run on GPU with channels last. Either decrease the polynomial degree / number of channels, or switch to channel first.");
		return;
	}
	taylor_series_channels_last_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(input, coeffs, output, n_b, n_s, n_c, n_i);
    if(CHECK_ERRORS) check_error(dev, "taylor_series_channels_last launch failed with error");
}

__global__ void taylor_series_grad_channels_first_kernel(const float* input, const float* coeffs, const float* grad, float* g_input, float* g_coeffs, const int n_b, const int n_s, const int n_c, const int n_i){
	__shared__ float coeffs_shr [NUM_THREADS];
	__shared__ float g_coeffs_shr [NUM_THREADS];
	__shared__ float temp_space [NUM_THREADS];
	int p = blockIdx.x * NUM_THREADS + threadIdx.x;
    
	for(int c = 0; c < n_c; c++){
		
		//load coeffs in as shared data
		coeffs_shr[threadIdx.x] = coeffs[c*n_i+threadIdx.x];
		__syncthreads();
		g_coeffs_shr[threadIdx.x] = 0.0f;
		
		for(int b = 0; b < n_b; b++){
			const float x = input[(b*n_c+c)*n_s+p];
			float g_input_t = 0.0f;
			float poly = 1.0f;
			const float grad_t = grad[(b*n_c+c)*n_s+p];
			
			for(int i = 0; i < n_i; i++){
				if( i > 0 )
					g_input_t += poly * coeffs_shr[i];
				temp_space[threadIdx.x] = 0.0f;
				temp_space[threadIdx.x] = p < n_s ? poly * grad_t : 0.0f;
				__syncthreads();
				
				//reduce add temp space into g_coeffs
				for(int t = NUM_THREADS/2; t > 0; t/=2){
					if(threadIdx.x < t)
						temp_space[threadIdx.x] += temp_space[threadIdx.x+t];
					__syncthreads();
				}
				if(threadIdx.x == 0)
					g_coeffs_shr[i] += temp_space[0];
				temp_space[threadIdx.x] = 0.0f;
				__syncthreads();
				
				poly *= x / (float) (i+1);
				
			}
			g_input_t *= grad_t;
			if(p < n_s)
				g_input[(b*n_c+c)*n_s+p] = g_input_t;
		}
		
		if(threadIdx.x < n_i)
			atomicAdd(g_coeffs+(c*n_i+threadIdx.x), g_coeffs_shr[threadIdx.x]);
			//g_coeffs[c*n_i+threadIdx.x] += g_coeffs_shr[threadIdx.x];
	}
}

__global__ void taylor_series_grad_channels_last_kernel(const float* input, const float* coeffs, const float* grad, float* g_input, float* g_coeffs, int n_s, int n_c, int n_i){
	__shared__ float coeffs_shr [NUM_THREADS];
	__shared__ float g_coeffs_shr [NUM_THREADS];
	__shared__ float temp_space [NUM_THREADS];
	int p = blockIdx.x * NUM_THREADS + threadIdx.x;
	int c = p % n_c;
    coeffs_shr[threadIdx.x] = coeffs[threadIdx.x];
	g_coeffs_shr[threadIdx.x] = 0.0f;
	
	float g_input_t = 0.0f;
	float x = input[p];
	float poly = 1.0f;
	float grad_t = grad[p];
	
	for(int i = 0; i < n_i; i++){
		if( i > 0 )
			g_input_t += poly * coeffs_shr[c*n_i+i];
		
		temp_space[threadIdx.x] = p < n_s*n_c ? poly * grad_t : 0.0f;
		__syncthreads();
		
		//reduce add temp space into g_coeffs
		if(threadIdx.x < n_c)
			for(int t = threadIdx.x; t < NUM_THREADS; t+=n_c)
				g_coeffs_shr[c*n_i+i] += temp_space[t];
		__syncthreads();
		
		poly *= x / (float) (i+1);

	}
	g_input_t *= grad_t;
	if(p < n_s*n_c)
		g_input[p] = g_input_t;
	
	if(threadIdx.x < n_i*n_c)
		atomicAdd(g_coeffs+threadIdx.x, g_coeffs_shr[threadIdx.x]);

}
void taylor_series_grad_channels_first(const cudaStream_t& dev, const float* input, const float* coeffs, const float* grad, float* g_input, float* g_coeffs, int n_b, int n_s, int n_c, int n_i){
	clear_buffer(dev, g_coeffs, n_c*n_i);
	clear_buffer(dev, g_input, n_b*n_c*n_s);
	taylor_series_grad_channels_first_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(input, coeffs, grad, g_input, g_coeffs, n_b, n_s, n_c, n_i);
    if(CHECK_ERRORS) check_error(dev, "taylor_series_channels_first launch failed with error");
}

void taylor_series_grad_channels_last(const cudaStream_t& dev, const float* input, const float* coeffs, const float* grad, float* g_input, float* g_coeffs, int n_b, int n_s, int n_c, int n_i){
	clear_buffer(dev, g_coeffs, n_c*n_i);
	clear_buffer(dev, g_input, n_b*n_c*n_s);
	if( n_c * n_i > NUM_THREADS ){
		printf("Too many coeffs to run on GPU with channels last. Either decrease the polynomial degree / number of channels, or switch to channel first.");
		return;
	}
	taylor_series_grad_channels_last_kernel<<<((n_b*n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(input, coeffs, grad, g_input, g_coeffs, n_b*n_s, n_c, n_i);
    if(CHECK_ERRORS) check_error(dev, "taylor_series_channels_last launch failed with error");
}


__global__ void init_flows_binary_kernel(const float* data, float* ps, float* pt, const int n_s){
	int i = blockIdx.x * NUM_THREADS + threadIdx.x;
	float d_t = (i < n_s) ? data[i] : 0.0f;
	float ps_t = (d_t > 0.0f) ? d_t : 0.0f ;
	float pt_t = (d_t < 0.0f) ? -d_t : 0.0f ;
	if(i < n_s){
		ps[i] = ps_t;
		pt[i] = pt_t;
	}
	
}

void init_flows_binary(const cudaStream_t& dev, const float* data, float* ps, float* pt, const int n_s){
	init_flows_binary_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), (n_s > NUM_THREADS) ? NUM_THREADS : n_s, 0, dev>>>(data, ps, pt, n_s);
    if(CHECK_ERRORS) check_error(dev, "init_flows_binary_kernel launch failed with error");
}

__global__ void init_flows_potts_kernel(const float* data, float* ps, float* pt, float* u, const int n_s, const int n_c){
	int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float min = -((i < n_s) ? data[i] : -1.0f/0.0f);
    for(int c = 1; c < n_c; c++){
        float d_t = - ((i < n_s) ? data[c*n_s+i] : 0.0f);
        if(d_t < min)
            min = d_t;
    }
    
    if(i < n_s)
        ps[i] = min;
    
    for(int c = 0; c < n_c; c++){
        float d_t = - ((i < n_s) ? data[c*n_s+i] : 0.0f);
        float u_t = (d_t == min) ? 1.0f: 0.0f;
        if(i < n_s){
            pt[c*n_s+i] = min;
            u[c*n_s+i] = u_t;
        }
    }
	
}

void init_flows_potts(const cudaStream_t& dev, const float* data, float* ps, float* pt, float* u, const int n_s, const int n_c){
	init_flows_potts_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev>>>(data, ps, pt, u, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "init_flows_potts_kernel launch failed with error");
}
