/// \file hmfNd.h
/// \author John S.H. Baxter
/// \brief Tensorflow-facing code for "regular" solvers, i.e. inputs are all same-sized

#ifndef HMFND_H
#define HMFND_H

#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

// Load the CPU kernels
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename Device>
class HmfNdOp : public OpKernel {
protected:
    const int N;
    const int I;
	const int O;
    
	bool channels_first;
	bool is_grad;
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
    explicit HmfNdOp(OpKernelConstruction* context, int N, int num_inputs, int num_outputs, bool channels_first, bool is_grad);
    ~HmfNdOp();
    void Compute(OpKernelContext* context) override;
};

template class HmfNdOp<CPUDevice>;
#if GOOGLE_CUDA
template class HmfNdOp<GPUDevice>;
#endif

#endif //HMFND_H