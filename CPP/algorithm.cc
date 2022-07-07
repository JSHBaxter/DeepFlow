#include "algorithm.h"

#include <stdexcept>
#include <iostream>
#include <string.h>

#ifdef USE_CUDA
#include "gpu_kernels.h"
#endif

template <class DEV>
MAXFLOW_ALGORITHM<DEV>::MAXFLOW_ALGORITHM(const DEV & dev) :
dev(dev),
buffer(0)
{}

template <class DEV>
void MAXFLOW_ALGORITHM<DEV>::run_solver(){
    if(DEBUG_ITER) std::cout << "\tRunning solver" << std::endl;
    run();
}

template <class DEV>
MAXFLOW_ALGORITHM<DEV>::~MAXFLOW_ALGORITHM(){
    if(DEBUG_ITER) std::cout << "\tDestructor " << buffer << std::endl;
    deallocate_memory(dev, (void**) &buffer);
}

template <class DEV>
void MAXFLOW_ALGORITHM<DEV>::construct(){
    int buffer_size = get_buffer_size();
    
    if(DEBUG_ITER) std::cout << "\tTrying to allocate " << buffer_size << std::endl;
    buffer = (float*) allocate_memory(dev, buffer_size*sizeof(float));
    allocate_buffers(buffer,0,0);
    if(DEBUG_ITER) std::cout << "\tAllocate Finished" << std::endl;
}

void* allocate_memory(const CPU_DEVICE & dev, size_t amount){
    void* buffer = 0;
    buffer = malloc(amount);
    if(buffer)
        return buffer;
    throw std::length_error( "Out of memory" );

}
void deallocate_memory(const CPU_DEVICE & dev, void** ptr){
    if(!ptr || !*ptr)
        return;
    free(*ptr);
    *ptr = 0;
}

void move_memory_to_device(const CPU_DEVICE & dev, const void* const src_ptr, void* const dest_ptr, size_t amount){
    memcpy(dest_ptr, src_ptr, amount);
}

void get_memory_from_device(const CPU_DEVICE & dev, const void* const src_ptr, void* const dest_ptr, size_t amount){
    memcpy(dest_ptr, src_ptr, amount);
}

//create device managing class for CUDA devices
#ifdef USE_CUDA
void* allocate_memory(const CUDA_DEVICE & dev, size_t amount){
    cudaSetDevice(dev.dev_number);
    float* buffer = 0;
    cudaMalloc( &buffer, amount);
    if(buffer)
        return buffer;
    throw std::length_error( "Out of memory" );

}

void deallocate_memory(const CUDA_DEVICE & dev, void** ptr){
    cudaSetDevice(dev.dev_number);
    if(!ptr || !*ptr)
        return;
    cudaFree(*ptr);
    *ptr = 0;
}

void move_memory_to_device(const CUDA_DEVICE & dev, const void* const src_ptr, void* const dest_ptr, size_t amount){
    send_to_gpu(dev, src_ptr, dest_ptr, amount);
}

void get_memory_from_device(const CUDA_DEVICE & dev, const void* const src_ptr, void* const dest_ptr, size_t amount){
    get_from_gpu(dev, src_ptr, dest_ptr, amount);
}

#endif
