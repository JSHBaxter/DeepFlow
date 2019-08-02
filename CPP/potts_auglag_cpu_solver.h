    

#ifndef POTTS_AUGLAG_CPU_SOLVER_H
#define POTTS_AUGLAG_CPU_SOLVER_H

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

class POTTS_AUGLAG_CPU_SOLVER_BASE
{
private:

protected:
    const int b;
    const int n_c;
    const int n_s;
    const float* const data;
    float* const u;
    float* ps;
    float* pt;
    float* div;
    float* g;
    
    // optimization constants
    const float tau = 0.1f;
    const float beta = 0.005f;
    const float epsilon = 10e-5f;
    const float cc = 0.1f;
    const float icc = 1.0f/cc;
    
    virtual int min_iter_calc() = 0;
    virtual void clear_spatial_flows() = 0;
    virtual void update_spatial_flow_calc() = 0;
    virtual void clean_up() = 0;
    void block_iter();
    
public:
    POTTS_AUGLAG_CPU_SOLVER_BASE(
        const int batch,
        const int n_s,
        const int n_c,
        const float* data_cost,
        float* u) ;
        
    ~POTTS_AUGLAG_CPU_SOLVER_BASE();
    
    void operator()();
};

#endif //POTTS_AUGLAG_CPU_SOLVER_H
