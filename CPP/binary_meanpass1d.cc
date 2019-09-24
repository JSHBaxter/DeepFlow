/// \file binary_meanpass1d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the mean-field message passing approximate solver for a Binary 
/// segmentation model operation in Tensorflow.

#include "regularNd.h"
#include "binary_meanpass1d.h"
#include "tf_memory_utils.h"
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
#include "binary_meanpass1d_cpu_solver.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "binary_meanpass1d_gpu_solver.cc"
#endif

// Define the OpKernel class
template <typename Device>
class BinaryMeanpass1dOp : public RegularNdOp<Device> {
public:
    explicit BinaryMeanpass1dOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 1, 2, 1) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return BinaryMeanpass1dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return BinaryMeanpass1dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        Tensor* u = this->outputs[0];
        
        // call function
        BinaryMeanpass1dFunctor<Device>()(
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

// Define the OpKernel class
template <typename Device>
class BinaryMeanpass1dGradOp : public RegularNdOp<Device> {
public:
    explicit BinaryMeanpass1dGradOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 1, 4, 2) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return BinaryMeanpass1dGradFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return BinaryMeanpass1dGradFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* grad = &(context->input(0));
        const Tensor* data_cost = &(context->input(1));
        const Tensor* rx_cost = &(context->input(2));
        const Tensor* u = &(context->input(3));
        Tensor* grad_data = this->outputs[0];
        Tensor* grad_rx = this->outputs[1];
        
        // call function
        BinaryMeanpass1dGradFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            u->flat<float>().data(),
            grad->flat<float>().data(),
            grad_data->flat<float>().data(),
            grad_rx->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

REGISTER_OP("BinaryMeanpass1d")
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
REGISTER_OP("BinaryMeanpass1dGrad")
  .Input("grad: float")
  .Input("data: float")
  .Input("rx: float")
  .Input("u: float")
  .Output("grad_data: float")
  .Output("grad_rx: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs(); i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 3, &input));
    c->set_output(0, c->input(0));
    return Status::OK();
  });


// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("BinaryMeanpass1d").Device(DEVICE_CPU), BinaryMeanpass1dOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("BinaryMeanpass1dGrad").Device(DEVICE_CPU), BinaryMeanpass1dGradOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("BinaryMeanpass1d").Device(DEVICE_GPU), BinaryMeanpass1dOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("BinaryMeanpass1dGrad").Device(DEVICE_GPU), BinaryMeanpass1dGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA


