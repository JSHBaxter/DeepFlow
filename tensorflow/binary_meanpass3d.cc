/// \file binary_meanpass3d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the mean-field message passing approximate solver for a Binary 
/// segmentation model operation in Tensorflow.

#include "regularNd.h"
#include "binary_meanpass3d.h"

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
#include "binary_meanpass3d_cpu_functor.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "binary_meanpass3d_gpu_functor.cc"
#endif

// Define the OpKernel class
template <typename Device>
class BinaryMeanpass3dOp : public RegularNdOp<Device> {
public:
    explicit BinaryMeanpass3dOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 3, 4, 1) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return BinaryMeanpass3dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return BinaryMeanpass3dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* rz_cost = &(context->input(3));
        Tensor* u = this->outputs[0];
        
        // call function
        BinaryMeanpass3dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            rz_cost->flat<float>().data(),
			0,
            u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

// Define the OpKernel class
template <typename Device>
class BinaryMeanpass3dWithInitOp : public RegularNdOp<Device> {
public:
    explicit BinaryMeanpass3dWithInitOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 3, 5, 1) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return BinaryMeanpass3dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return BinaryMeanpass3dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* rz_cost = &(context->input(3));
        const Tensor* init_u = &(context->input(4));
        Tensor* u = this->outputs[0];
        
        // call function
        BinaryMeanpass3dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            rz_cost->flat<float>().data(),
            init_u->flat<float>().data(),
            u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

// Define the OpKernel class
template <typename Device>
class BinaryMeanpass3dGradOp : public RegularNdOp<Device> {
public:
    explicit BinaryMeanpass3dGradOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 3, 6, 4) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return BinaryMeanpass3dGradFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return BinaryMeanpass3dGradFunctor<Device>().num_buffers_images();
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
        
        // call function
        BinaryMeanpass3dGradFunctor<Device>()(
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

// Define the OpKernel class
template <typename Device>
class BinaryMeanpass3dWithInitGradOp : public RegularNdOp<Device> {
public:
    explicit BinaryMeanpass3dWithInitGradOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 3, 6, 5) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return BinaryMeanpass3dGradFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return BinaryMeanpass3dGradFunctor<Device>().num_buffers_images();
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
        
        // call function
        BinaryMeanpass3dGradFunctor<Device>()(
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

REGISTER_OP("BinaryMeanpass3d")
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
REGISTER_OP("BinaryMeanpass3dWithInit")
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
REGISTER_OP("BinaryMeanpass3dGrad")
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
    for (size_t i = 0; i < 4; i++)
		c->set_output(i, c->input(0));
    return Status::OK();
  });
REGISTER_OP("BinaryMeanpass3dWithInitGrad")
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
    for (size_t i = 0; i < 5; i++)
		c->set_output(i, c->input(0));
    return Status::OK();
  });


// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("BinaryMeanpass3d").Device(DEVICE_CPU), BinaryMeanpass3dOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("BinaryMeanpass3dWithInit").Device(DEVICE_CPU), BinaryMeanpass3dWithInitOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("BinaryMeanpass3dGrad").Device(DEVICE_CPU), BinaryMeanpass3dGradOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("BinaryMeanpass3dWithInitGrad").Device(DEVICE_CPU), BinaryMeanpass3dWithInitGradOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("BinaryMeanpass3d").Device(DEVICE_GPU), BinaryMeanpass3dOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("BinaryMeanpass3dWithInit").Device(DEVICE_GPU), BinaryMeanpass3dWithInitOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("BinaryMeanpass3dGrad").Device(DEVICE_GPU), BinaryMeanpass3dGradOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("BinaryMeanpass3dWithInitGrad").Device(DEVICE_GPU), BinaryMeanpass3dWithInitGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA


