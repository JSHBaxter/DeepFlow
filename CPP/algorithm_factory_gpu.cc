#include "algorithm_factory_gpu.h"

ALGORITHM_GPU* get_algorithm(const cudaStream_t & dev, unsigned int type, unsigned int dim, unsigned int flags, const float* const* const inputs, float* const* const outputs){
    if( dim > 3 ){
        return NULL
    }
    if( type > 4){
        return NULL
    }
}