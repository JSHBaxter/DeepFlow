/// \file hmf_meanpass1d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the augmented Lagrangian solver for an HMF 
/// segmentation model operation in Tensorflow.

#include "hmfNd.h"
#include "hmf_meanpass1d.h"

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
#include "hmf_meanpass1d_cpu_solver.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "hmf_meanpass1d_gpu_solver.cc"
#endif

// Define the OpKernel class
template <typename Device>
class HmfMeanpass1dOp : public HmfNdOp<Device> {
public:
    HmfMeanpass1dOp(OpKernelConstruction* context) : 
		HmfNdOp<Device>(context, 1, 2, 1, std::is_same<Device, GPUDevice>::value, false)	{}

protected:

    int Get_Num_Intermediates_Full() override {
        return HmfMeanpass1dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return HmfMeanpass1dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* parentage = &(context->input(2));
        const Tensor* data_index = &(context->input(3));
        Tensor* u = this->outputs[0];
		
        // call function
        HmfMeanpass1dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
			0,
            u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};
// Define the OpKernel class
template <typename Device>
class HmfMeanpass1dWithInitOp : public HmfNdOp<Device> {
public:
    HmfMeanpass1dWithInitOp(OpKernelConstruction* context) : 
		HmfNdOp<Device>(context, 1, 3, 1, std::is_same<Device, GPUDevice>::value, false)	{}

protected:

    int Get_Num_Intermediates_Full() override {
        return HmfMeanpass1dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return HmfMeanpass1dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* init_u = &(context->input(2));
        const Tensor* parentage = &(context->input(3));
        const Tensor* data_index = &(context->input(4));
        Tensor* u = this->outputs[0];
		
        // call function
        HmfMeanpass1dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            init_u->flat<float>().data(),
            u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

// Define the OpKernel class
template <typename Device>
class HmfMeanpass1dGradOp : public HmfNdOp<Device> {
public:
    HmfMeanpass1dGradOp(OpKernelConstruction* context) : 
		HmfNdOp<Device>(context, 1, 4, 2, std::is_same<Device, GPUDevice>::value, true)	{}

protected:

    int Get_Num_Intermediates_Full() override {
        return HmfMeanpass1dGradFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return HmfMeanpass1dGradFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* grad = &(context->input(2));
        const Tensor* u = &(context->input(3));
        const Tensor* parentage = &(context->input(4));
        const Tensor* data_index = &(context->input(5));
        Tensor* grad_data = this->outputs[0];
        Tensor* grad_rx = this->outputs[1];
        Tensor* grad_par = this->outputs[2];
        Tensor* grad_didx = this->outputs[3];
		
        // call function
        HmfMeanpass1dGradFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            u->flat<float>().data(),
            grad->flat<float>().data(),
            grad_data->flat<float>().data(),
            grad_rx->flat<float>().data(),
            grad_par->flat<int>().data(),
            grad_didx->flat<int>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

// Define the OpKernel class
template <typename Device>
class HmfMeanpass1dWithInitGradOp : public HmfNdOp<Device> {
public:
    HmfMeanpass1dWithInitGradOp(OpKernelConstruction* context) : 
		HmfNdOp<Device>(context, 1, 4, 3, std::is_same<Device, GPUDevice>::value, true)	{}

protected:

    int Get_Num_Intermediates_Full() override {
        return HmfMeanpass1dGradFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return HmfMeanpass1dGradFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* grad = &(context->input(2));
        const Tensor* u = &(context->input(3));
        const Tensor* parentage = &(context->input(4));
        const Tensor* data_index = &(context->input(5));
        Tensor* grad_data = this->outputs[0];
        Tensor* grad_rx = this->outputs[1];
        Tensor* grad_par = this->outputs[3];
        Tensor* grad_didx = this->outputs[4];
		
        // call function
        HmfMeanpass1dGradFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            u->flat<float>().data(),
            grad->flat<float>().data(),
            grad_data->flat<float>().data(),
            grad_rx->flat<float>().data(),
            grad_par->flat<int>().data(),
            grad_didx->flat<int>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

REGISTER_OP("HmfMeanpass1d")
  .Input("data: float")
  .Input("rx: float")
  .Input("parentage: int32")
  .Input("data_index: int32")
  .Output("u: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs()-2; i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 3, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-2), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-1), 1, &input));
    c->set_output(0, c->input(0));
    return Status::OK();
  });
  
REGISTER_OP("HmfMeanpass1dWithInit")
  .Input("data: float")
  .Input("rx: float")
  .Input("init_u: float")
  .Input("parentage: int32")
  .Input("data_index: int32")
  .Output("u: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs()-2; i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 3, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-2), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-1), 1, &input));
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("HmfMeanpass1dGrad")
  .Input("data: float")
  .Input("rx: float")
  .Input("grad: float")
  .Input("u: float")
  .Input("parentage: int32")
  .Input("data_index: int32")
  .Output("grad_data: float")
  .Output("grad_rx: float")
  .Output("grad_par: int32")
  .Output("grad_didx: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs()-2; i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 3, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-2), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-1), 1, &input));
    for (size_t i = 0; i < c->num_inputs()-4; i++)
        c->set_output(i, c->input(i));
	c->set_output(c->num_inputs()-4, c->input(c->num_inputs()-2));
	c->set_output(c->num_inputs()-3, c->input(c->num_inputs()-1));
    return Status::OK();
  });

REGISTER_OP("HmfMeanpass1dWithInitGrad")
  .Input("data: float")
  .Input("rx: float")
  .Input("grad: float")
  .Input("u: float")
  .Input("parentage: int32")
  .Input("data_index: int32")
  .Output("grad_data: float")
  .Output("grad_rx: float")
  .Output("grad_init: float")
  .Output("grad_par: int32")
  .Output("grad_didx: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs()-2; i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 3, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-2), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-1), 1, &input));
    for (size_t i = 0; i < c->num_inputs()-3; i++)
        c->set_output(i, c->input(i));
	c->set_output(c->num_inputs()-3, c->input(c->num_inputs()-2));
	c->set_output(c->num_inputs()-2, c->input(c->num_inputs()-1));
    return Status::OK();
  });


// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass1d").Device(DEVICE_CPU), HmfMeanpass1dOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass1dWithInit").Device(DEVICE_CPU), HmfMeanpass1dWithInitOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass1dGrad").Device(DEVICE_CPU), HmfMeanpass1dGradOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass1dWithInitGrad").Device(DEVICE_CPU), HmfMeanpass1dWithInitGradOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass1d").Device(DEVICE_GPU), HmfMeanpass1dOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass1dWithInit").Device(DEVICE_GPU), HmfMeanpass1dWithInitOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass1dGrad").Device(DEVICE_GPU), HmfMeanpass1dGradOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass1dWithInitGrad").Device(DEVICE_GPU), HmfMeanpass1dWithInitGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA
