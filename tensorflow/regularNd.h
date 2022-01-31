/// \file regularNd.h
/// \author John S.H. Baxter
/// \brief Tensorflow-facing code for "regular" solvers, i.e. inputs are all same-sized

#ifndef REGULARND_H
#define REGULARND_H

#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

template <typename Device>
class RegularNdOp : public OpKernel {
protected:
    const int N;
    const int I;
	const int O;
    
    int n_s;
    int n_i;
    int* size_array;
    
    Tensor** outputs;
    
    void CheckInputs(OpKernelContext* context);
    void GetOutputTensors(OpKernelContext* context);
    
    //leave the calling to the child classes
    virtual void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) = 0;
    virtual int Get_Num_Intermediates_Full() = 0;
    virtual int Get_Num_Intermediates_Images() = 0;
    
public:
    explicit RegularNdOp(OpKernelConstruction* context, int N, int num_inputs, int num_outputs);
    ~RegularNdOp();
    void Compute(OpKernelContext* context) override;
};

template class RegularNdOp<Eigen::ThreadPoolDevice>;
#if GOOGLE_CUDA
template class RegularNdOp<Eigen::GpuDevice>;
#endif

#endif //REGULARND_H