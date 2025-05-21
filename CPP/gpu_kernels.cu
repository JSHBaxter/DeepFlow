#include <cuda.h>
//#include <cublas_v2.h>
#include <stdio.h>

#include "gpu_kernels.h"

// Flags specific to the GPU (i.e. debug, number of threads, etc...)
#define CHECK_ERRORS true
#define NUM_THREADS 256
#define epsilon 0.00001f

void let_device_catchup(const CUDA_DEVICE& dev){
    cudaError_t cudaerr = cudaStreamSynchronize(dev.stream);
}

void check_error(const CUDA_DEVICE& dev, const char* string){
    cudaError_t cudaerr = cudaStreamSynchronize(dev.stream);
    if (cudaerr != cudaSuccess){
        printf(string);
        printf(" \"%s\".\n", cudaGetErrorString(cudaerr));
        exit(-1);
    }
}

void* allocate_on_gpu(const CUDA_DEVICE& dev, size_t amount){
    void* ptr;
    cudaMalloc(&ptr,amount);
    if(CHECK_ERRORS) check_error(dev, "cudaMalloc failed");
    return ptr;
}

void deallocate_on_gpu(const CUDA_DEVICE& dev, void* ptr){
    cudaFree(ptr);
}

void get_from_gpu(const CUDA_DEVICE& dev, const void* source, void* dest, size_t amount){
    cudaMemcpyAsync(dest,source,amount,cudaMemcpyDeviceToHost,dev.stream);
    cudaStreamSynchronize(dev.stream);
}

void print_buffer(const CUDA_DEVICE& dev, const float* buffer, const int n_s){
	float* c_buffer = (float*) malloc(n_s*sizeof(float));
	get_from_gpu(dev, buffer, c_buffer, n_s*sizeof(float));
	for(int i = 0; i < n_s; i++)
		printf("%f ",c_buffer[i]);
	printf("\n");
	free(c_buffer);
}

void send_to_gpu(const CUDA_DEVICE& dev, const void* source, void* dest, size_t amount){
    cudaStreamSynchronize(dev.stream);
    cudaMemcpyAsync(dest,source,amount,cudaMemcpyHostToDevice,dev.stream);
    cudaStreamSynchronize(dev.stream);
}

// Sets variables to 0.0f
void clear_buffer(const CUDA_DEVICE& dev, float* buffer, const int size){
    set_buffer(dev, buffer, 0.0f, size);
}

void clear_buffer(const CUDA_DEVICE& dev, int* buffer, const int size){
    cudaMemsetAsync(buffer, 0, size*sizeof(int),dev.stream);
}

__global__ void set_kernel(float* buffer, const float number, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    if( i < n_s )
        buffer[i] = number;
}
    

void set_buffer(const CUDA_DEVICE& dev, float* buffer, const float number, const int n_s){
    if( n_s <= 0 )
        return;
    set_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(buffer, number, n_s);
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

void mark_neg_equal(const CUDA_DEVICE& dev, const float* buffer_s, const float* buffer_l, float* u, const int n_s, const int n_c){
    mark_neg_equal_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(buffer_s, buffer_l, u, n_s, n_c);
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

void aggregate_bottom_up(const CUDA_DEVICE& dev, float** p_ind, float* buffer, const float* org, const int n_s, const int n_c, const int n_r){
    aggregate_bottom_up_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(p_ind, buffer, org, n_s, n_c, n_r);
    if(CHECK_ERRORS) check_error(dev, "aggregate_bottom_up launch failed with error");
}


__global__ void rep_buffer_channel_last(const float* in, float* out, const int n_c, const int n_s) {
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float out_t = in[i];
    for(int c = 1; c < n_c; c++)
        if( i < n_s )
            out[i*n_c+c] = out_t;
}

//channel first is a series of memcpy's

void rep_buffer(const CUDA_DEVICE& dev, const float* input, float* output, const int n_c, const int n_s){
    for(int c = 0; c < n_c; c++)
        cudaMemcpyAsync((void*)(output+c*n_s),input,n_s*sizeof(float),cudaMemcpyDeviceToDevice,dev.stream);
    if(CHECK_ERRORS) check_error(dev, "rep_buffer launch failed with error");
}


__global__ void log_buffer_kernel(const float* in, float* out, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float value = in[i];
	value = (value < epsilon) ? epsilon : value;
    value = log(value);
    
    if(i < n_s)
        out[i] = value;
    
}

void log_buffer(const CUDA_DEVICE& dev, const float* in, float* out, const int n_s){
    log_buffer_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(in, out, n_s);
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


void binary_constrain(const CUDA_DEVICE& dev, float* buffer, const int n_s){
    binary_constrain_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(buffer, n_s);
    if(CHECK_ERRORS) check_error(dev, "binary_constrain launch failed with error");   
}




__device__ void abs_constrain_device(const int i, float* b, const float* r, const float* l, const int n_s){
    float value = (i < n_s) ? b[i] : 0.0f;
    float constraint = (i < n_s) ? r[i] : 0.0f;
    float exception =  (l && i < n_s) ? l[i] : 0.0f;
    value = (value < constraint || value*exception >= 0.0) ? value : constraint;
    value = (value > -constraint || value*exception >= 0.0) ? value: -constraint;
    
    if(i < n_s)
        b[i] = value;
    
}

__global__ void abs_constrain_kernel(float* b, const float* r, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    abs_constrain_device(i, b, r, 0, n_s);
}


void abs_constrain(const CUDA_DEVICE& dev, float* buffer, const float* constrain, const int n_s){
    abs_constrain_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(buffer, constrain, n_s);
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


void max_neg_constrain(const CUDA_DEVICE& dev, float* buffer, const float* constrain, const int n_s){
    max_neg_constrain_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(buffer, constrain, n_s);
    if(CHECK_ERRORS) check_error(dev, "max_constrain launch failed with error");
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



float mean_of_buffer(const CUDA_DEVICE& dev, const float* buffer, const int n_s){
    float* buffer_c = (float*) malloc(n_s*sizeof(float));
    get_from_gpu(dev,buffer,buffer_c,n_s*sizeof(float));
	cudaStreamSynchronize(dev.stream);
    double mean_value = 0.0;
    for(int s = 0; s < n_s; s++)
        mean_value += (buffer_c[s] < 0) ? -buffer_c[s] : buffer_c[s];
    free(buffer_c);
    return (float)(mean_value / (double) n_s);
}

float max_of_buffer(const CUDA_DEVICE& dev, const float* buffer, const int n_s){
    float* buffer_c = (float*) malloc(n_s*sizeof(float));
    get_from_gpu(dev,buffer,buffer_c,n_s*sizeof(float));
	cudaStreamSynchronize(dev.stream);
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

float spat_max_of_buffer(const CUDA_DEVICE& dev, const float* buffer, const int n_s, const int n_c){
    float* buffer_c = (float*) malloc(n_c*n_s*sizeof(float));
    get_from_gpu(dev,buffer,buffer_c,n_c*n_s*sizeof(float));
	cudaStreamSynchronize(dev.stream);
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

void copy_buffer(const CUDA_DEVICE& dev, const float* source, float* dest, const int n_s){
    copy_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(source, dest, n_s);
    if(CHECK_ERRORS) check_error(dev, "copy_buffer launch failed with error");
}


__global__ void copy_clean_kernel(const float* source, float* dest, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = (i < n_s) ? source[i] : 0.0f;
    val = isfinite(val) ? val: 0.0f;
    if(i < n_s)
        dest[i] = val;
}

void copy_buffer_clean(const CUDA_DEVICE& dev, const float* source, float* dest, const int n_s){
    copy_clean_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(source, dest, n_s);
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

void copy_buffer_clip(const CUDA_DEVICE& dev, const float* source, float* dest, const int n_s, float clip){
    copy_clip_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(source, dest, n_s, clip);
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

void inc_buffer(const CUDA_DEVICE& dev, const float* inc, float* acc, const int n_s){
    inc_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(inc, acc, n_s);
    if(CHECK_ERRORS) check_error(dev, "inc_buffer launch failed with error");
}

__global__ void inc_kernel(const float inc, float* acc, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = (i < n_s) ? acc[i] : 0.0f;
    val += inc;
    if(i < n_s)
        acc[i] = val;
}

void inc_buffer(const CUDA_DEVICE& dev, const float inc, float* acc, const int n_s){
    inc_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(inc, acc, n_s);
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

void inc_inc_minc_buffer(const CUDA_DEVICE& dev, const float* inc1, const float* inc2, const float* minc, const float multi, float* acc, const int n_s){
    inc_inc_minc_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(inc1, inc2, minc, multi, acc, n_s);
    if(CHECK_ERRORS) check_error(dev, "inc_inc_minc_buffer launch failed with error");
}


__global__ void m_inc_inc_ninc_minc_kernel(const float* inc1, const float* inc2, const float* ninc, const float* minc, const float multi_end, const float multi_all, float* acc, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = (i < n_s) ? acc[i] : 0.0f;
    float increment1 = (i < n_s) ? inc1[i] : 0.0f;
    float increment2 = (i < n_s) ? inc2[i] : 0.0f;
    float nincrement = (i < n_s) ? ninc[i] : 0.0f;
    float minincrement = (i < n_s) ? minc[i] : 0.0f;
    val += multi_all*(increment1 + increment2 - nincrement + multi_end*minincrement);
    if(i < n_s)
        acc[i] = val;
}

void m_inc_inc_ninc_minc_buffer(const CUDA_DEVICE& dev, const float* inc1, const float* inc2, const float* ninc, const float* minc, const float multi_end, const float multi_all, float* acc, const int n_s){
    m_inc_inc_ninc_minc_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(inc1, inc2, ninc, minc, multi_end, multi_all, acc, n_s);
    if(CHECK_ERRORS) check_error(dev, "m_inc_inc_ninc_minc_buffer launch failed with error");
}


__global__ void ninc_kernel(const float* inc, float* acc, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = (i < n_s) ? acc[i] : 0.0f;
    float increment = (i < n_s) ? inc[i] : 0.0f;
    val -= increment;
    if(i < n_s)
        acc[i] = val;
}

void ninc_buffer(const CUDA_DEVICE& dev, const float* inc, float* acc, const int n_s){
    ninc_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(inc, acc, n_s);
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

void inc_mult_buffer(const CUDA_DEVICE& dev, const float* inc, float* acc, const int n_s, const float multi){
    inc_mult_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(inc, acc, n_s, multi);
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

void inc2_mult_buffer(const CUDA_DEVICE& dev, const float* inc_m1, const float* inc_m2, float* acc, const int n_s, const float multi){
    inc2_mult_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(inc_m1, inc_m2, acc, n_s, multi);
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

void div_buffer(const CUDA_DEVICE& dev, const float* div, float* res, const int n_s){
    div_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(div, res, n_s);
    if(CHECK_ERRORS) check_error(dev, "div_buffer launch failed with error");
}

__global__ void div_kernel(const float divisor, float* res, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = res[i];
    val /= divisor;
    if(i < n_s)
        res[i] = val;
}

void div_buffer(const CUDA_DEVICE& dev, const float divisor, float* res, const int n_s){
    div_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(divisor, res, n_s);
    if(CHECK_ERRORS) check_error(dev, "div_buffer launch failed with error");
}

__global__ void mult_kernel(const float mult, float* res, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = res[i]*mult;
    if(i < n_s)
        res[i] = val;
}

void mult_buffer(const CUDA_DEVICE& dev, const float mult, float* res, const int n_s){
    mult_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(mult, res, n_s);
    if(CHECK_ERRORS) check_error(dev, "mult_buffer launch failed with error");
}

__global__ void mult_kernel2(const float mult, const float* input, float* res, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = input[i]*mult;
    if(i < n_s)
        res[i] = val;
}

void mult_buffer(const CUDA_DEVICE& dev, const float mult, const float* input, float* res, const int n_s){
    mult_kernel2<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(mult, input, res, n_s);
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

void exp_and_inc_buffer(const CUDA_DEVICE& dev, const float* max, float* cost, float* acc, const int n_s){
    exp_and_inc_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(max, cost, acc, n_s);
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

void add_store_then_max_buffer(const CUDA_DEVICE& dev, const float* comp1, const float* comp2, float* store, float* res, const int n_s){
    add_store_then_max_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(comp1, comp2, store, res, n_s);
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

void softmax(const CUDA_DEVICE& dev, const float* e1, const float* e2, float* u, const int n_s, const int n_c){
    if(e2 != NULL)
        softmax_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(e1, e2, u, n_s, n_c);
    else
        softmax_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(e1, u, n_s, n_c);
    if(CHECK_ERRORS) check_error(dev, "softmax launch failed with error");
}

void softmax(const CUDA_DEVICE& dev, const float* bufferin, float* bufferout, const int n_s, const int n_c){
    softmax_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(bufferin, bufferout, n_s, n_c);
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

void neg_softmax(const CUDA_DEVICE& dev, const float* e, float* u, const int n_s, const int n_c){
    neg_softmax_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(e, u, n_s, n_c);
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

void sigmoid(const CUDA_DEVICE& dev, const float* e1, const float* e2, float* u, const int n_s){
    if(e2 != NULL)
        sigmoid_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(e1, e2, u, n_s);
    else
        sigmoid_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(e1, u, n_s);
    if(CHECK_ERRORS) check_error(dev, "sigmoid launch failed with error");
}


__global__ void exp_kernel(const float* e1, float* u, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    
    float cost = e1[i];
	float value = exp(cost);
	if(i < n_s)
		u[i] = value;
}

void exp(const CUDA_DEVICE& dev, const float* e1, float* u, const int n_s){
    exp_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(e1, u, n_s);
    if(CHECK_ERRORS) check_error(dev, "sigmoid launch failed with error");
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


void parity_mask(const CUDA_DEVICE& dev, float* buffer, const int dim, const int* const n, const int n_c, const int parity){
    int n_s = n_c;
    for(int i = 0; i < dim; i++)
        n_s *= n[i];
    
    switch(dim){
        case 1:
            parity_mask_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(buffer, n[0], n_c, parity);
            break;
        case 2:
            parity_mask_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(buffer, n[0], n[1], n_c, parity);
            break;
        case 3:
            parity_mask_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(buffer, n[0], n[1], n[2], n_c, parity);
            break;
    }
    if(CHECK_ERRORS) check_error(dev, "parity_mask launch failed with error");
}

void parity_mask(const CUDA_DEVICE& dev, float* buffer, const float* other, const int dim, const int* const n, const int n_c, const int parity){
    int n_s = n_c;
    for(int i = 0; i < dim; i++)
        n_s *= n[i];
    
    switch(dim){
        case 1:
            parity_mask_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(buffer, other, n[0], n_c, parity);
            break;
        case 2:
            parity_mask_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(buffer, other, n[0], n[1], n_c, parity);
            break;
        case 3:
            parity_mask_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(buffer, other, n[0], n[1], n[2], n_c, parity);
            break;
    }
    if(CHECK_ERRORS) check_error(dev, "parity_mask -merge launch failed with error");
}



__global__ void add_then_store_kernel(const float* addend1, const float* addend2, float* sum, const int n_s){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
    float a1 = (i < n_s) ? addend1[i] : 0.0f;
    float a2 = (i < n_s) ? addend2[i] : 0.0f;
    float res = a1+a2;
    if(i < n_s)
        sum[i] = res;   
}

void add_then_store(const CUDA_DEVICE& dev, const float* addend1, const float* addend2, float* sum, const int n_s){
    add_then_store_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(addend1, addend2, sum, n_s);
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

void add_then_store(const CUDA_DEVICE& dev, const float* addend1, const float* addend2, float* sum1, float* sum2, const int n_s){
    add_then_store_kernel_2<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(addend1, addend2, sum1, sum2, n_s);
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

void untangle_softmax(const CUDA_DEVICE& dev, const float* du_i, const float* u, float* loc, const int n_s, const int n_c){
    untangle_softmax_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(du_i, u, loc, n_s, n_c);
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

void untangle_sigmoid(const CUDA_DEVICE& dev, const float* du_i, const float* u, float* loc, const int n_s){
    untangle_sigmoid_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(du_i, u, loc, n_s);
    if(CHECK_ERRORS) check_error(dev, "process_grad_binary launch failed with error");
}
