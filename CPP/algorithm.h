#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <stdexcept>
#include <stdlib.h>


struct CPU_DEVICE {
    const bool channels_first;
};

void* allocate_memory(const CPU_DEVICE & dev, size_t amount);
void deallocate_memory(const CPU_DEVICE & dev, void** ptr);
void move_memory_to_device(const CPU_DEVICE & dev, const void* const src_ptr, void* const dest_ptr, size_t amount);
void get_memory_from_device(const CPU_DEVICE & dev, const void* const src_ptr, void* const dest_ptr, size_t amount);

//create device managing class for CUDA devices
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
struct CUDA_DEVICE{
    const int dev_number;
    const cudaStream_t & stream;
};
}

void* allocate_memory(const CUDA_DEVICE & dev, size_t amount);
void deallocate_memory(const CUDA_DEVICE & dev, void** ptr);
void move_memory_to_device(const CUDA_DEVICE & dev, const void* const src_ptr, void* const dest_ptr, size_t amount);
void get_memory_from_device(const CUDA_DEVICE & dev, const void* const src_ptr, void* const dest_ptr, size_t amount);

#endif

//Abstract base class for algorithm parts (designed for them to contain sub-parts)
template <typename DEV>
class MAXFLOW_ALGORITHM {
protected:
    const DEV & dev;
    float* buffer;
    
    void construct();
    
    MAXFLOW_ALGORITHM(const DEV & dev);
    ~MAXFLOW_ALGORITHM();
    
public:
    void run_solver();
    
    virtual void allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over) = 0;
    virtual int get_buffer_size() = 0;
    virtual void run() = 0;
};


template class MAXFLOW_ALGORITHM<CPU_DEVICE>;
#ifdef USE_CUDA
template class MAXFLOW_ALGORITHM<CUDA_DEVICE>;
#endif

#endif