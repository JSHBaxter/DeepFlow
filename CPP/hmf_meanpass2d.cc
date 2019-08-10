/// \file hmf_meanpass2d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the augmented Lagrangian solver for an HMF 
/// segmentation model operation in Tensorflow.

#include "hmf_meanpass2d.h"
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
#include "hmf_meanpass2d_cpu_solver.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "hmf_meanpass2d_gpu_solver.cc"
#endif

// Define the OpKernel class
// template parameter <float> is the datatype of the tensors.
template <typename Device>
class HmfMeanpass2dOp : public OpKernel {
public:
    explicit HmfMeanpass2dOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        
        // ensure all inputs are present
        DCHECK_EQ(5, context->num_inputs());

        // get the input tensors
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* parentage = &(context->input(3));
        const Tensor* data_index = &(context->input(4));

        // Ensure tensor is small enough to function
        OP_REQUIRES(context, data_cost->NumElements() <= tensorflow::kint32max / 16,
                    errors::InvalidArgument("Too many elements in tensor"));
        
        // check shapes of input and weights
        const DataType data_type = data_cost->dtype();
        const TensorShape& data_shape = data_cost->shape();
        const TensorShape& rx_shape = rx_cost->shape();
        const TensorShape& ry_shape = ry_cost->shape();
        const TensorShape& parentage_shape = parentage->shape();
        const TensorShape& data_index_shape = data_index->shape();
        int size_array[6] = {(int) data_shape.dim_size(0),
                             (int) data_shape.dim_size(1),
                             (int) data_shape.dim_size(2),
                             (int) data_shape.dim_size(3),
                             (int) rx_shape.dim_size(1),
                             (int) rx_shape.dim_size(3)};

        // check input is of correct rank
        DCHECK_EQ(data_shape.dims(), 4);
        DCHECK_EQ(rx_shape.dims(), 4);
        DCHECK_EQ(ry_shape.dims(), 4);

        // check input is of correct size
        DCHECK_EQ(data_shape.dim_size(0), rx_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(2), rx_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(0), ry_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(2), ry_shape.dim_size(2));
        DCHECK_EQ(rx_shape.dim_size(3), ry_shape.dim_size(3));
        DCHECK_EQ(rx_shape.dim_size(1), ry_shape.dim_size(1));

        // create output tensor
        Tensor* u = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, data_shape, &u));

        // create intermediate buffers as needed
        int n_i = size_array[2]*size_array[3];
        int n_s = n_i*size_array[4];
        int num_intermediates_full = HmfMeanpass2dFunctor<Device>().num_buffers_full();
        int num_intermediates_images = HmfMeanpass2dFunctor<Device>().num_buffers_images();
        float** buffers_full = NULL;
        float** buffers_imgs = NULL;
        get_temporary_buffers(context, buffers_full, n_s, num_intermediates_full, buffers_imgs, n_i, num_intermediates_images, data_cost);
		
        // call function
        HmfMeanpass2dFunctor<Device>()(
            context->eigen_device<Device>(),
            size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
        
        //deallocate buffers
        clear_temporary_buffers(context, buffers_full, n_s, num_intermediates_full);
        clear_temporary_buffers(context, buffers_imgs, n_i, num_intermediates_images);
    }
};

// Define the OpKernel class
// template parameter <float> is the datatype of the tensors.
template <typename Device>
class HmfMeanpass2dGradOp : public OpKernel {
public:
    explicit HmfMeanpass2dGradOp(OpKernelConstruction* context) : OpKernel(context)  {}

    void Compute(OpKernelContext* context) override {
        
		std::cout << "In kernel" << std::endl;
		
        // ensure all inputs are present
        DCHECK_EQ(7, context->num_inputs());

        // get the input tensors
        const Tensor* grad = &(context->input(0));
        const Tensor* data_cost = &(context->input(1));
        const Tensor* rx_cost = &(context->input(2));
        const Tensor* ry_cost = &(context->input(3));
        const Tensor* parentage = &(context->input(4));
        const Tensor* data_index = &(context->input(5));
        const Tensor* u = &(context->input(6));

        // Ensure tensor is small enough to function
        OP_REQUIRES(context, data_cost->NumElements() <= tensorflow::kint32max / 16,
                    errors::InvalidArgument("Too many elements in tensor"));
        
        // check shapes of input and weights
        const DataType data_type = data_cost->dtype();
        const TensorShape& data_shape = data_cost->shape();
        const TensorShape& rx_shape = rx_cost->shape();
        const TensorShape& ry_shape = ry_cost->shape();
        const TensorShape& parentage_shape = parentage->shape();
        const TensorShape& data_index_shape = data_index->shape();
        int size_array[6] = {(int) data_shape.dim_size(0),
                             (int) data_shape.dim_size(1),
                             (int) data_shape.dim_size(2),
                             (int) data_shape.dim_size(3),
                             (int) rx_shape.dim_size(1),
                             (int) rx_shape.dim_size(3)};

        // check input is of rank 5
        DCHECK_EQ(data_shape.dims(), 4);
        DCHECK_EQ(rx_shape.dims(), 4);
        DCHECK_EQ(ry_shape.dims(), 4);

        // check input is of correct size
        DCHECK_EQ(data_shape.dim_size(0), rx_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(2), rx_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(0), ry_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(2), ry_shape.dim_size(2));
        DCHECK_EQ(rx_shape.dim_size(3), ry_shape.dim_size(3));
        DCHECK_EQ(rx_shape.dim_size(1), ry_shape.dim_size(1));
        
        //get output tensors
        Tensor* grad_data = NULL;
        Tensor* grad_rx = NULL;
        Tensor* grad_ry = NULL;
        Tensor* grad_par = NULL;
        Tensor* grad_didx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, data_shape, &grad_data));
        OP_REQUIRES_OK(context, context->allocate_output(1, rx_shape, &grad_rx));
        OP_REQUIRES_OK(context, context->allocate_output(2, ry_shape, &grad_ry));
        OP_REQUIRES_OK(context, context->allocate_output(3, parentage_shape, &grad_par));
        OP_REQUIRES_OK(context, context->allocate_output(4, data_index_shape, &grad_didx));

        // create intermediate buffers as needed
        int n_i = size_array[2]*size_array[3];
        int n_s = n_i*size_array[4];
        int num_intermediates_full = HmfMeanpass2dGradFunctor<Device>().num_buffers_full();
        int num_intermediates_images = HmfMeanpass2dGradFunctor<Device>().num_buffers_images();
        float** buffers_full = NULL;
        float** buffers_imgs = NULL;
        get_temporary_buffers(context, buffers_full, n_s, num_intermediates_full, buffers_imgs, n_i, num_intermediates_images, data_cost);
        
        // call function for gradient
        HmfMeanpass2dGradFunctor<Device>()(
            context->eigen_device<Device>(),
            size_array,
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
        
        //deallocate buffers
        clear_temporary_buffers(context, buffers_full, n_s, num_intermediates_full);
        clear_temporary_buffers(context, buffers_imgs, n_i, num_intermediates_images);
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

REGISTER_OP("HmfMeanpass2dGrad")
  .Input("grad: float")
  .Input("data: float")
  .Input("rx: float")
  .Input("ry: float")
  .Input("parentage: int32")
  .Input("data_index: int32")
  .Input("u: float")
  .Output("grad_data: float")
  .Output("grad_rx: float")
  .Output("grad_ry: float")
  .Output("grad_par: int32")
  .Output("grad_didx: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs()-3; i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 4, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-3), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-2), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-1), 4, &input));
    for (size_t i = 0; i < c->num_inputs()-2; i++)
        c->set_output(i, c->input(i+1));
    return Status::OK();
  });


// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass2d").Device(DEVICE_CPU), HmfMeanpass2dOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass2dGrad").Device(DEVICE_CPU), HmfMeanpass2dGradOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass2d").Device(DEVICE_GPU), HmfMeanpass2dOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass2dGrad").Device(DEVICE_GPU), HmfMeanpass2dGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA


