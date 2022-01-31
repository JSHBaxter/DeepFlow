/// \file potts_meanpass3d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the mean-field message passing approximate solver for a Potts 
/// segmentation model operation in Tensorflow.

#include "potts_meanpass3d.h"
#include "regularNd.h"
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
#include "potts_meanpass3d_cpu_functor.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "potts_meanpass3d_gpu_functor.cc"
#endif

// Define the OpKernel class
template <typename Device>
class PottsMeanpass3dOp : public RegularNdOp<Device> {
public:
    explicit PottsMeanpass3dOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 3, 4, 1) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return PottsMeanpass3dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return PottsMeanpass3dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* rz_cost = &(context->input(3));
        Tensor* new_u = this->outputs[0];
        
        // call function
        PottsMeanpass3dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            rz_cost->flat<float>().data(),
            0,
            new_u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

// Define the OpKernel class
template <typename Device>
class PottsMeanpass3dWithInitOp : public RegularNdOp<Device> {
public:
    explicit PottsMeanpass3dWithInitOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 3, 5, 1) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return PottsMeanpass3dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return PottsMeanpass3dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* rz_cost = &(context->input(3));
        const Tensor* init_u = &(context->input(4));
        Tensor* new_u = this->outputs[0];
        
        // call function
        PottsMeanpass3dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            rz_cost->flat<float>().data(),
            init_u->flat<float>().data(),
            new_u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

template <typename Device>
class PottsMeanpass3dGradOp : public RegularNdOp<Device> {
public:
    explicit PottsMeanpass3dGradOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 3, 6, 4) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return PottsMeanpass3dGradFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return PottsMeanpass3dGradFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* grad = &(context->input(0));
        const Tensor* data_cost = &(context->input(1));
        const Tensor* rx_cost = &(context->input(2));
        const Tensor* ry_cost = &(context->input(3));
        const Tensor* rz_cost = &(context->input(4));
        const Tensor* u = &(context->input(5));
        Tensor* grad_data = this->outputs[0];
        Tensor* grad_rx = this->outputs[1];
        Tensor* grad_ry = this->outputs[2];
        Tensor* grad_rz = this->outputs[3];
        
        // call function for gradient
        PottsMeanpass3dGradFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            rz_cost->flat<float>().data(),
            u->flat<float>().data(),
            grad->flat<float>().data(),
            grad_data->flat<float>().data(),
            grad_rx->flat<float>().data(),
            grad_ry->flat<float>().data(),
            grad_rz->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

template <typename Device>
class PottsMeanpass3dWithInitGradOp : public RegularNdOp<Device> {
public:
    explicit PottsMeanpass3dWithInitGradOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 3, 6, 5) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return PottsMeanpass3dGradFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return PottsMeanpass3dGradFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* grad = &(context->input(0));
        const Tensor* data_cost = &(context->input(1));
        const Tensor* rx_cost = &(context->input(2));
        const Tensor* ry_cost = &(context->input(3));
        const Tensor* rz_cost = &(context->input(4));
        const Tensor* u = &(context->input(5));
        Tensor* grad_data = this->outputs[0];
        Tensor* grad_rx = this->outputs[1];
        Tensor* grad_ry = this->outputs[2];
        Tensor* grad_rz = this->outputs[3];
        Tensor* grad_init = this->outputs[4];
        
        // call function for gradient
        PottsMeanpass3dGradFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            rz_cost->flat<float>().data(),
            u->flat<float>().data(),
            grad->flat<float>().data(),
            grad_data->flat<float>().data(),
            grad_rx->flat<float>().data(),
            grad_ry->flat<float>().data(),
            grad_rz->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

REGISTER_OP("PottsMeanpass3d")
  .Input("data: float")
  .Input("rx: float")
  .Input("ry: float")
  .Input("rz: float")
  .Output("u: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs(); i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 5, &input));
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("PottsMeanpass3dWithInit")
  .Input("data: float")
  .Input("rx: float")
  .Input("ry: float")
  .Input("rz: float")
  .Input("init_u: float")
  .Output("u: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs(); i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 5, &input));
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("PottsMeanpass3dGrad")
  .Input("grad: float")
  .Input("data: float")
  .Input("rx: float")
  .Input("ry: float")
  .Input("rz: float")
  .Input("u: float")
  .Output("grad_data: float")
  .Output("grad_rx: float")
  .Output("grad_ry: float")
  .Output("grad_rz: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs(); i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 5, &input));
    for (size_t i = 0; i < c->num_inputs()-2; i++)
        c->set_output(i, c->input(i+1));
    return Status::OK();
  });

REGISTER_OP("PottsMeanpass3dWithInitGrad")
  .Input("grad: float")
  .Input("data: float")
  .Input("rx: float")
  .Input("ry: float")
  .Input("rz: float")
  .Input("u: float")
  .Output("grad_data: float")
  .Output("grad_rx: float")
  .Output("grad_ry: float")
  .Output("grad_rz: float")
  .Output("grad_init: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs(); i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 5, &input));
    for (size_t i = 0; i < c->num_inputs()-1; i++)
        c->set_output(i, c->input(i+1));
    return Status::OK();
  });


// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass3d").Device(DEVICE_CPU), PottsMeanpass3dOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass3dWithInit").Device(DEVICE_CPU), PottsMeanpass3dWithInitOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass3dGrad").Device(DEVICE_CPU), PottsMeanpass3dGradOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass3dWithInitGrad").Device(DEVICE_CPU), PottsMeanpass3dWithInitGradOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass3d").Device(DEVICE_GPU), PottsMeanpass3dOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass3dWithInit").Device(DEVICE_GPU), PottsMeanpass3dWithInitOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass3dGrad").Device(DEVICE_GPU), PottsMeanpass3dGradOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass3dWithInitGrad").Device(DEVICE_GPU), PottsMeanpass3dWithInitGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA


