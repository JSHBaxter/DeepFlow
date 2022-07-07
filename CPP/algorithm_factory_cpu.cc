#include "algorithm_factory_cpu.h"

#include "

ALGORITHM_CPU* get_algorithm(unsigned int type, unsigned int dim, const int* dims, unsigned int flags, const float* const* const inputs, float* const* const outputs){
    if( dim > 3 ){
        return NULL
    }
    if( type > 4){
        return NULL
    }
}