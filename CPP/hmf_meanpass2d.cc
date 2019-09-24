/// \file hmf_meanpass2d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the augmented Lagrangian solver for an HMF 
/// segmentation model operation in Tensorflow.

#include "hmfNd.h"
#include "hmf_meanpass2d.h"

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
#include "hmf_meanpass2d_cpu_solver.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "hmf_meanpass2d_gpu_solver.cc"
#endif

// Define the OpKernel class
template <typename Device>
class HmfMeanpass2dOp : public HmfNdOp<Device> {
public:
    HmfMeanpass2dOp(OpKernelConstruction* context) : 
		HmfNdOp<Device>(context, 2, 3, 1, std::is_same<Device, GPUDevice>::value, false)	{}

protected:

    int Get_Num_Intermediates_Full() override {
        return HmfMeanpass2dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return HmfMeanpass2dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* parentage = &(context->input(3));
        const Tensor* data_index = &(context->input(4));
        Tensor* u = this->outputs[0];
		
        // call function
        HmfMeanpass2dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
			0,
            u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

// Define the OpKernel class
template <typename Device>
class HmfMeanpass2dWithInitOp : public HmfNdOp<Device> {
public:
    HmfMeanpass2dWithInitOp(OpKernelConstruction* context) : 
		HmfNdOp<Device>(context, 2, 4, 1, std::is_same<Device, GPUDevice>::value, false)	{}

protected:

    int Get_Num_Intermediates_Full() override {
        return HmfMeanpass2dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return HmfMeanpass2dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* init_u = &(context->input(3));
        const Tensor* parentage = &(context->input(4));
        const Tensor* data_index = &(context->input(5));
        Tensor* u = this->outputs[0];
		
        // call function
        HmfMeanpass2dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            init_u->flat<float>().data(),
            u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

// Define the OpKernel class
template <typename Device>
class HmfMeanpass2dGradOp : public HmfNdOp<Device> {
public:
    HmfMeanpass2dGradOp(OpKernelConstruction* context) : 
		HmfNdOp<Device>(context, 2, 5, 3, std::is_same<Device, GPUDevice>::value, true)	{
		}

protected:

    int Get_Num_Intermediates_Full() override {
        return HmfMeanpass2dGradFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return HmfMeanpass2dGradFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* grad = &(context->input(3));
        const Tensor* u = &(context->input(4));
        const Tensor* parentage = &(context->input(5));
        const Tensor* data_index = &(context->input(6));
        Tensor* grad_data = this->outputs[0];
        Tensor* grad_rx = this->outputs[1];
        Tensor* grad_ry = this->outputs[2];
        Tensor* grad_par = this->outputs[3];
        Tensor* grad_didx = this->outputs[4];
			
        // call function
        HmfMeanpass2dGradFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            u->flat<float>().data(),
            grad->flat<float>().data(),
            grad_data->flat<float>().data(),
            grad_rx->flat<float>().data(),
            grad_ry->flat<float>().data(),
            grad_par->flat<int>().data(),
            grad_didx->flat<int>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

// Define the OpKernel class
template <typename Device>
class HmfMeanpass2dWithInitGradOp : public HmfNdOp<Device> {
public:
    HmfMeanpass2dWithInitGradOp(OpKernelConstruction* context) : 
		HmfNdOp<Device>(context, 2, 5, 4, std::is_same<Device, GPUDevice>::value, true)	{}

protected:

    int Get_Num_Intermediates_Full() override {
        return HmfMeanpass2dGradFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return HmfMeanpass2dGradFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* grad = &(context->input(3));
        const Tensor* u = &(context->input(4));
        const Tensor* parentage = &(context->input(5));
        const Tensor* data_index = &(context->input(6));
        Tensor* grad_data = this->outputs[0];
        Tensor* grad_rx = this->outputs[1];
        Tensor* grad_ry = this->outputs[2];
        Tensor* grad_init = this->outputs[3];
        Tensor* grad_par = this->outputs[4];
        Tensor* grad_didx = this->outputs[5];
		
        // call function
        HmfMeanpass2dGradFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            u->flat<float>().data(),
            grad->flat<float>().data(),
            grad_data->flat<float>().data(),
            grad_rx->flat<float>().data(),
            grad_ry->flat<float>().data(),
            grad_par->flat<int>().data(),
            grad_didx->flat<int>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

REGISTER_OP("HmfMeanpass2d")
  .Input("data: float")
  .Input("rx: float")
  .Input("ry: float")
  .Input("parentage: int32")
  .Input("data_index: int32")
  .Output("u: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs()-2; i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 4, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-2), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-1), 1, &input));
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("HmfMeanpass2dWithInit")
  .Input("data: float")
  .Input("rx: float")
  .Input("ry: float")
  .Input("init_u: float")
  .Input("parentage: int32")
  .Input("data_index: int32")
  .Output("u: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs()-2; i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 4, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-2), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-1), 1, &input));
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("HmfMeanpass2dGrad")
  .Input("data: float")
  .Input("rx: float")
  .Input("ry: float")
  .Input("grad: float")
  .Input("u: float")
  .Input("parentage: int32")
  .Input("data_index: int32")
  .Output("grad_data: float")
  .Output("grad_rx: float")
  .Output("grad_ry: float")
  .Output("grad_par: int32")
  .Output("grad_didx: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs()-2; i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 4, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-2), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-1), 1, &input));
    for (size_t i = 0; i < c->num_inputs()-4; i++)
        c->set_output(i, c->input(i));
	c->set_output(c->num_inputs()-4, c->input(c->num_inputs()-2));
	c->set_output(c->num_inputs()-3, c->input(c->num_inputs()-1));
    return Status::OK();
  });
  
REGISTER_OP("HmfMeanpass2dWithInitGrad")
  .Input("data: float")
  .Input("rx: float")
  .Input("ry: float")
  .Input("grad: float")
  .Input("u: float")
  .Input("parentage: int32")
  .Input("data_index: int32")
  .Output("grad_data: float")
  .Output("grad_rx: float")
  .Output("grad_ry: float")
  .Output("grad_init: float")
  .Output("grad_par: int32")
  .Output("grad_didx: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs()-2; i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 4, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-2), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-1), 1, &input));
    for (size_t i = 0; i < c->num_inputs()-3; i++)
        c->set_output(i, c->input(i));
	c->set_output(c->num_inputs()-3, c->input(c->num_inputs()-2));
	c->set_output(c->num_inputs()-2, c->input(c->num_inputs()-1));
    return Status::OK();
  });

// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass2d").Device(DEVICE_CPU), HmfMeanpass2dOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass2dWithInit").Device(DEVICE_CPU), HmfMeanpass2dWithInitOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass2dGrad").Device(DEVICE_CPU), HmfMeanpass2dGradOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass2dWithInitGrad").Device(DEVICE_CPU), HmfMeanpass2dWithInitGradOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass2d").Device(DEVICE_GPU), HmfMeanpass2dOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass2dWithInit").Device(DEVICE_GPU), HmfMeanpass2dWithInitOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass2dGrad").Device(DEVICE_GPU), HmfMeanpass2dGradOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass2dWithInitGrad").Device(DEVICE_GPU), HmfMeanpass2dWithInitGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA
