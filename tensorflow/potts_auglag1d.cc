/// \file potts_auglag1d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the augmented Lagrangian solver for a Potts 
/// segmentation model operation in Tensorflow.

#include "regularNd.h"
#include "potts_auglag1d.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <math.h>
#include <iostream>
using namespace tensorflow;

// Load the CPU kernels
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
#include "potts_auglag1d_cpu_functor.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "potts_auglag1d_gpu_functor.cc"
#endif

// Define the OpKernel class
template <typename Device>
class PottsAuglag1dOp : public RegularNdOp<Device> {
public:
    explicit PottsAuglag1dOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 1, 2, 1) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return PottsAuglag1dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return PottsAuglag1dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        Tensor* u = this->outputs[0];
        
        // call function
        PottsAuglag1dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

REGISTER_OP("PottsAuglag1d")
  .Input("data: float")
  .Input("rx: float")
  .Output("u: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs(); i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 3, &input));
    c->set_output(0, c->input(0));
    return Status::OK();
  });
// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("PottsAuglag1d").Device(DEVICE_CPU), PottsAuglag1dOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("PottsAuglag1d").Device(DEVICE_GPU), PottsAuglag1dOp<GPUDevice>);
#endif  // GOOGLE_CUDA


