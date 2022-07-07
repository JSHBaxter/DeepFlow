    

#ifndef POTTS_AUGLAG_SOLVER_H
#define POTTS_AUGLAG_SOLVER_H

#include "algorithm.h"
#include "spatial_auglag.h"

template<typename DEV>
class POTTS_AUGLAG_SOLVER : public MAXFLOW_ALGORITHM<DEV>
{
private:
    
protected:
    SPATIAL_AUGLAG_SOLVER<DEV>* spatial_flow;
    const int n_c;
    const int n_s;
    const float* const data;
    float* u;
    float* ps;
    float* pt;
    float* div;
    float* g;
    
    // optimization constants
    const float tau = 0.1f;
    const float beta = 0.001f;
    const float epsilon = 10e-5f;
    const float cc = 0.25f;
    const float icc = 1.0f/cc;
    
    void block_iter();
    
public:
    POTTS_AUGLAG_SOLVER(
        const DEV & dev,
        const bool star,
        const int dim,
        const int* dims,
        const int n_c,
        const float * const * const inputs,
        float* u);
    
    ~POTTS_AUGLAG_SOLVER();
    
    void allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over);
    int get_buffer_size();

    void run();
};

template class POTTS_AUGLAG_SOLVER<CPU_DEVICE>;
#ifdef USE_CUDA
template class POTTS_AUGLAG_SOLVER<CUDA_DEVICE>;
#endif

#endif //POTTS_AUGLAG_SOLVER_H
