/// \file potts_meanpass1d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the mean-field message passing approximate solver for a Potts 
/// segmentation model operation in Tensorflow.

#include "regularNd.h"
#include "potts_meanpass1d.h"
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
#include "potts_meanpass1d_cpu_functor.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "potts_meanpass1d_gpu_functor.cc"
#endif

// Define the OpKernel class
template <typename Device>
class PottsMeanpass1dOp : public RegularNdOp<Device> {
public:
    explicit PottsMeanpass1dOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 1, 2, 1) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return PottsMeanpass1dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return PottsMeanpass1dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        Tensor* new_u = this->outputs[0];
        
        // call function
        PottsMeanpass1dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            0,
            new_u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

template <typename Device>
class PottsMeanpass1dWithInitOp : public RegularNdOp<Device> {
public:
    explicit PottsMeanpass1dWithInitOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 1, 3, 1) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return PottsMeanpass1dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return PottsMeanpass1dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* init_u = &(context->input(2));
        Tensor* new_u = this->outputs[0];
        
        // call function
        PottsMeanpass1dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            init_u->flat<float>().data(),
            new_u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

template <typename Device>
class PottsMeanpass1dGradOp : public RegularNdOp<Device> {
public:
    explicit PottsMeanpass1dGradOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 1, 4, 2) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return PottsMeanpass1dGradFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return PottsMeanpass1dGradFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* grad = &(context->input(1));
        const Tensor* data_cost = &(context->input(1));
        const Tensor* rx_cost = &(context->input(2));
        const Tensor* u = &(context->input(3));
        Tensor* grad_data = this->outputs[0];
        Tensor* grad_rx = this->outputs[1];
        
        // call function for gradient
        PottsMeanpass1dGradFunctor<Device>()(
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

template <typename Device>
class PottsMeanpass1dWithInitGradOp : public RegularNdOp<Device> {
public:
    explicit PottsMeanpass1dWithInitGradOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 1, 4, 3) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return PottsMeanpass1dGradFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return PottsMeanpass1dGradFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* grad = &(context->input(1));
        const Tensor* data_cost = &(context->input(1));
        const Tensor* rx_cost = &(context->input(2));
        const Tensor* u = &(context->input(3));
        Tensor* grad_data = this->outputs[0];
        Tensor* grad_rx = this->outputs[1];
        
        // call function for gradient
        PottsMeanpass1dGradFunctor<Device>()(
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

REGISTER_OP("PottsMeanpass1d")
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
REGISTER_OP("PottsMeanpass1dWithInit")
  .Input("data: float")
  .Input("rx: float")
  .Input("init_u: float")
  .Output("u: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs(); i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 3, &input));
    c->set_output(0, c->input(0));
    return Status::OK();
  });
REGISTER_OP("PottsMeanpass1dGrad")
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
    for (size_t i = 0; i < c->num_inputs()-2; i++)
        c->set_output(i, c->input(i+1));
    return Status::OK();
  });
REGISTER_OP("PottsMeanpass1dWithInitGrad")
  .Input("grad: float")
  .Input("data: float")
  .Input("rx: float")
  .Input("u: float")
  .Output("grad_data: float")
  .Output("grad_rx: float")
  .Output("grad_init: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs(); i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 3, &input));
    for (size_t i = 0; i < c->num_inputs()-1; i++)
        c->set_output(i, c->input(i+1));
    return Status::OK();
  });


// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass1d").Device(DEVICE_CPU), PottsMeanpass1dOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass1dWithInit").Device(DEVICE_CPU), PottsMeanpass1dWithInitOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass1dGrad").Device(DEVICE_CPU), PottsMeanpass1dGradOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass1dWithInitGrad").Device(DEVICE_CPU), PottsMeanpass1dWithInitGradOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass1d").Device(DEVICE_GPU), PottsMeanpass1dOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass1dWithInit").Device(DEVICE_GPU), PottsMeanpass1dWithInitOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass1dGrad").Device(DEVICE_GPU), PottsMeanpass1dGradOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass1dWithInitGrad").Device(DEVICE_GPU), PottsMeanpass1dWithInitGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA


