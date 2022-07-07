#ifndef ALGORITHM_FACTORY_GPU_H
#define ALGORITHM_FACTORY_GPU_H

#include <cuda_runtime.h>
#include <cuda.h>

class ALGORITHM_GPU {
public:
    virtual void run() = 0;
};

//flags: bit 0 - channels_first (0) or channels_last(1)
//       bit 1 - star-convex or no

ALGORITHM_GPU* get_algorithm(const cudaStream_t & dev, unsigned int type, unsigned int dim, const int* dims, unsigned int flags, const float* const* const inputs, float* const* const outputs);

#endif