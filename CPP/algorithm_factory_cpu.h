#ifndef ALGORITHM_FACTORY_CPU_H
#define ALGORITHM_FACTORY_CPU_H

class ALGORITHM_CPU {
public:
    virtual void run() = 0;
};

ALGORITHM_CPU* get_algorithm(unsigned int type, unsigned int dim, const int* dims, unsigned int flags, const float* const* const inputs, float* const* const outputs);

#endif