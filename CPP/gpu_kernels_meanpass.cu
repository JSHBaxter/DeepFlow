#include <cuda.h>
//#include <cublas_v2.h>
#include <stdio.h>

#include "gpu_kernels.h"
#include "gpu_kernels_meanpass.h"

// Flags specific to the GPU (i.e. debug, number of threads, etc...)
#define CHECK_ERRORS true
#define NUM_THREADS 256
#define epsilon 0.00001f

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

void populate_reg_mean_gradients_and_add(const CUDA_DEVICE& dev, const float* g, const float* u, float *const *const g_r, const int dim, const int* const n, const int n_c, const float tau){
    int n_s = 1;
    for(int i = 0; i < dim; i++)
        n_s *= n[i];
    
    switch(dim){
        case 1:
            populate_reg_mean_gradients_and_add_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, u, g_r[0], n[0], n_c, tau);
            break;
        case 2:
            populate_reg_mean_gradients_and_add_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, u, g_r[0], g_r[1], n[0], n[1], n_c, tau);
            break;
        case 3:
            populate_reg_mean_gradients_and_add_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(g, u, g_r[0], g_r[1], g_r[2], n[0], n[1], n[2], n_c, tau);
            break;
    }
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

void change_to_diff(const CUDA_DEVICE& dev, float* transfer, float* diff, const int n_s, const float tau){
    change_to_diff_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(transfer, diff, n_s, tau);
    if(CHECK_ERRORS) check_error(dev, "populate_reg_gradient launch failed with error");
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

void get_effective_reg(const CUDA_DEVICE& dev, float* const r_eff, const float* const u, const float *const *const r, const int dim, const int* const n, const int n_c){
    int n_s = 1;
    for(int i = 0; i < dim; i++)
        n_s *= n[i];
    
    switch(dim){
        case 1:
            get_effective_reg_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(r_eff, u, r[0], n[0], n_c);
            break;
        case 2:
            get_effective_reg_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(r_eff, u, r[0], r[1], n[0], n[1], n_c);
            break;
        case 3:
            get_effective_reg_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(r_eff, u, r[0], r[1], r[2], n[0], n[1], n[2], n_c);
            break;
    }
    if(CHECK_ERRORS) check_error(dev, "get_effective_reg launch failed with error");   
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

void get_gradient_for_u(const CUDA_DEVICE & dev, const float* dy, const float *const *const r, float* const du, const int dim, const int* const n, const int n_c, const float tau){
    int n_s = 1;
    for(int i = 0; i < dim; i++)
        n_s *= n[i];
    
    switch(dim){
        case 1:
            get_gradient_for_u_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(dy, du, r[0], n[0], n_c, tau);
            break;
        case 2:
            get_gradient_for_u_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(dy, du, r[0], r[1], n[0], n[1], n_c, tau);
            break;
        case 3:
            get_gradient_for_u_kernel<<<((n_s+NUM_THREADS-1)/NUM_THREADS), NUM_THREADS, 0, dev.stream>>>(dy, du, r[0], r[1], r[2], n[0], n[1], n[2], n_c, tau);
            break;
        }
    if(CHECK_ERRORS) check_error(dev, "get_gradient_for_u launch failed with error");
}
