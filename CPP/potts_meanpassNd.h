/// \file potts_meanpassNd.h
/// \author John S.H. Baxter
/// \brief Implementation of the mean-field message passing approximate solver for a Potts 
/// segmentation model operation in Tensorflow.

#ifndef POTTS_MEANPASSND_H
#define POTTS_MEANPASSND_H

#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

// Load the CPU kernels
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename Device>
class PottsMeanpassNdOp : public OpKernel {
protected:
    const int N;
    const int I;
    
    int n_s;
    int n_i;
    int* size_array;
    
    Tensor* u;
    
    void CheckInputs(OpKernelContext* context);
    void GetOutputTensors(OpKernelContext* context);
    
    //leave the calling to the child classes
    virtual void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) = 0;
    virtual int Get_Num_Intermediates_Full() = 0;
    virtual int Get_Num_Intermediates_Images() = 0;
    
public:
    explicit PottsMeanpassNdOp(OpKernelConstruction* context, int N, int with_init);
    ~PottsMeanpassNdOp();
    void Compute(OpKernelContext* context) override;
};

template <typename Device>
class PottsMeanpassNdGradOp : public OpKernel {
protected:
    const int N;
    const int I;
    
    int n_s;
    int n_i;
    int* size_array;
    
    Tensor** grads;
    
    void CheckInputs(OpKernelContext* context);
    void GetOutputTensors(OpKernelContext* context);
    
    //leave the calling to the child classes
    virtual void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) = 0;
    virtual int Get_Num_Intermediates_Full() = 0;
    virtual int Get_Num_Intermediates_Images() = 0;
    
public:
    explicit PottsMeanpassNdGradOp(OpKernelConstruction* context, int N, int with_init);
    ~PottsMeanpassNdGradOp();
    void Compute(OpKernelContext* context) override;
};

template class PottsMeanpassNdOp<CPUDevice>;
template class PottsMeanpassNdGradOp<CPUDevice>;
#if GOOGLE_CUDA
template class PottsMeanpassNdOp<GPUDevice>;
template class PottsMeanpassNdGradOp<GPUDevice>;
#endif

#endif //POTTS_MEANPASSND_H