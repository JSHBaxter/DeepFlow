/// \file hmf_meanpass3d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the augmented Lagrangian solver for an HMF 
/// segmentation model operation in Tensorflow.

#include "hmf_meanpass3d.h"
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
#include "hmf_meanpass3d_cpu_solver.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "hmf_meanpass3d_gpu_solver.cc"
#endif

// Define the OpKernel class
// template parameter <float> is the datatype of the tensors.
template <typename Device>
class HmfMeanpass3dOp : public OpKernel {
public:
    explicit HmfMeanpass3dOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        
        // ensure all inputs are present
        DCHECK_EQ(6, context->num_inputs());

        // get the input tensors
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* rz_cost = &(context->input(3));
        const Tensor* parentage = &(context->input(4));
        const Tensor* data_index = &(context->input(5));

        // Ensure tensor is small enough to function
        OP_REQUIRES(context, data_cost->NumElements() <= tensorflow::kint32max / 16,
                    errors::InvalidArgument("Too many elements in tensor"));
        
        // check shapes of input and weights
        const TensorShape& data_shape = data_cost->shape();
        const TensorShape& rx_shape = rx_cost->shape();
        const TensorShape& ry_shape = ry_cost->shape();
        const TensorShape& rz_shape = rz_cost->shape();
        const TensorShape& parentage_shape = parentage->shape();
        const TensorShape& data_index_shape = data_index->shape();
        int size_array[7] = {(int) data_shape.dim_size(0),
                             (int) data_shape.dim_size(1),
                             (int) data_shape.dim_size(2),
                             (int) data_shape.dim_size(3),
                             (int) data_shape.dim_size(4),
                             (int) rx_shape.dim_size(1),
                             (int) rx_shape.dim_size(4)};

        // check input is of rank 5
        DCHECK_EQ(data_shape.dims(), 5);
        DCHECK_EQ(rx_shape.dims(), 5);
        DCHECK_EQ(ry_shape.dims(), 5);
        DCHECK_EQ(rz_shape.dims(), 5);

        // check input is of correct size
        DCHECK_EQ(data_shape.dim_size(0), rx_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(2), rx_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(3), rx_shape.dim_size(3));
        DCHECK_EQ(data_shape.dim_size(0), ry_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(2), ry_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(3), ry_shape.dim_size(3));
        DCHECK_EQ(data_shape.dim_size(0), rz_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(2), rz_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(3), rz_shape.dim_size(3));
        DCHECK_EQ(rx_shape.dim_size(4), ry_shape.dim_size(4));
        DCHECK_EQ(rx_shape.dim_size(4), rz_shape.dim_size(4));
        DCHECK_EQ(rx_shape.dim_size(1), ry_shape.dim_size(1));
        DCHECK_EQ(rx_shape.dim_size(1), rz_shape.dim_size(1));

        // create output tensor
        Tensor* u = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, data_shape, &u));

        // create intermediate buffers as needed
        int num_intermediates_full = HmfMeanpass3dFunctor<Device>().num_buffers_full();
        int num_intermediates_images = HmfMeanpass3dFunctor<Device>().num_buffers_images();
        float** buffers_full = (num_intermediates_full > 0) ? new float*[num_intermediates_full]: NULL;
        float** buffers_imgs = (num_intermediates_images > 0) ? new float*[num_intermediates_images]: NULL;
        TensorShape full_shape;
        full_shape.AddDim(size_array[1]);
        full_shape.AddDim(size_array[2]);
        full_shape.AddDim(size_array[3]);
        full_shape.AddDim(size_array[4]);
        for(int b = 0; b < num_intermediates_full; b++){
            Tensor buffer;
            OP_REQUIRES_OK(context, context->allocate_temp(data_cost->dtype(), full_shape, &buffer));
            buffers_full[b] = buffer.flat<float>().data();
        }
        TensorShape img_shape;
        img_shape.AddDim(size_array[2]);
        img_shape.AddDim(size_array[3]);
        img_shape.AddDim(size_array[4]);
        for(int b = 0; b < num_intermediates_images; b++){
            Tensor buffer;
            OP_REQUIRES_OK(context, context->allocate_temp(data_cost->dtype(), img_shape, &buffer));
            buffers_imgs[b] = buffer.flat<float>().data();
        }
        
        // call function
        HmfMeanpass3dFunctor<Device>()(
            context->eigen_device<Device>(),
            size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            rz_cost->flat<float>().data(),
            u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
        
        //deallocate buffers - done autoamtically
        if(buffers_full != NULL)
            free(buffers_full);
        if(buffers_imgs != NULL)
            free(buffers_imgs);
    }
};

// Define the OpKernel class
// template parameter <float> is the datatype of the tensors.
template <typename Device>
class HmfMeanpass3dGradOp : public OpKernel {
public:
    explicit HmfMeanpass3dGradOp(OpKernelConstruction* context) : OpKernel(context)  {}

    void Compute(OpKernelContext* context) override {
        
        // ensure all inputs are present
        DCHECK_EQ(8, context->num_inputs());

        // get the input tensors
        const Tensor* grad = &(context->input(0));
        const Tensor* data_cost = &(context->input(1));
        const Tensor* rx_cost = &(context->input(2));
        const Tensor* ry_cost = &(context->input(3));
        const Tensor* rz_cost = &(context->input(4));
        const Tensor* parentage = &(context->input(5));
        const Tensor* data_index = &(context->input(6));
        const Tensor* u = &(context->input(7));

        // Ensure tensor is small enough to function
        OP_REQUIRES(context, data_cost->NumElements() <= tensorflow::kint32max / 16,
                    errors::InvalidArgument("Too many elements in tensor"));
        
        // check shapes of input and weights
        const TensorShape& data_shape = data_cost->shape();
        const TensorShape& rx_shape = rx_cost->shape();
        const TensorShape& ry_shape = ry_cost->shape();
        const TensorShape& rz_shape = rz_cost->shape();
        const TensorShape& parentage_shape = parentage->shape();
        const TensorShape& data_index_shape = data_index->shape();
        int size_array[7] = {(int) data_shape.dim_size(0),
                             (int) data_shape.dim_size(1),
                             (int) data_shape.dim_size(2),
                             (int) data_shape.dim_size(3),
                             (int) data_shape.dim_size(4),
                             (int) rx_shape.dim_size(1),
                             (int) rx_shape.dim_size(4)};

        // check input is of rank 5
        DCHECK_EQ(data_shape.dims(), 5);
        DCHECK_EQ(rx_shape.dims(), 5);
        DCHECK_EQ(ry_shape.dims(), 5);
        DCHECK_EQ(rz_shape.dims(), 5);

        // check input is of correct size
        DCHECK_EQ(data_shape.dim_size(0), rx_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(2), rx_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(3), rx_shape.dim_size(3));
        DCHECK_EQ(data_shape.dim_size(0), ry_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(2), ry_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(3), ry_shape.dim_size(3));
        DCHECK_EQ(data_shape.dim_size(0), rz_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(2), rz_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(3), rz_shape.dim_size(3));
        DCHECK_EQ(rx_shape.dim_size(4), ry_shape.dim_size(4));
        DCHECK_EQ(rx_shape.dim_size(4), rz_shape.dim_size(4));
        DCHECK_EQ(rx_shape.dim_size(1), ry_shape.dim_size(1));
        DCHECK_EQ(rx_shape.dim_size(1), rz_shape.dim_size(1));

        // create intermediate buffers as needed
        int num_intermediates_full = HmfMeanpass3dGradFunctor<Device>().num_buffers_full();
        int num_intermediates_images = HmfMeanpass3dGradFunctor<Device>().num_buffers_images();
        float** buffers_full = (num_intermediates_full > 0) ? new float*[num_intermediates_full]: NULL;
        float** buffers_imgs = (num_intermediates_images > 0) ? new float*[num_intermediates_images]: NULL;
        TensorShape full_shape;
        full_shape.AddDim(size_array[1]);
        full_shape.AddDim(size_array[2]);
        full_shape.AddDim(size_array[3]);
        full_shape.AddDim(size_array[4]);
        for(int b = 0; b < num_intermediates_full; b++){
            Tensor buffer;
            OP_REQUIRES_OK(context, context->allocate_temp(data_cost->dtype(), full_shape, &buffer));
            buffers_full[b] = buffer.flat<float>().data();
        }
        TensorShape img_shape;
        img_shape.AddDim(size_array[2]);
        img_shape.AddDim(size_array[3]);
        img_shape.AddDim(size_array[4]);
        for(int b = 0; b < num_intermediates_images; b++){
            Tensor buffer;
            OP_REQUIRES_OK(context, context->allocate_temp(data_cost->dtype(), img_shape, &buffer));
            buffers_imgs[b] = buffer.flat<float>().data();
        }
        
        //get output tensors
        Tensor* grad_data = NULL;
        Tensor* grad_rx = NULL;
        Tensor* grad_ry = NULL;
        Tensor* grad_rz = NULL;
        Tensor* grad_par = NULL;
        Tensor* grad_didx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, data_shape, &grad_data));
        OP_REQUIRES_OK(context, context->allocate_output(1, rx_shape, &grad_rx));
        OP_REQUIRES_OK(context, context->allocate_output(2, ry_shape, &grad_ry));
        OP_REQUIRES_OK(context, context->allocate_output(3, rz_shape, &grad_rz));
        OP_REQUIRES_OK(context, context->allocate_output(4, parentage_shape, &grad_par));
        OP_REQUIRES_OK(context, context->allocate_output(5, data_index_shape, &grad_didx));
        
        // call function for gradient
        HmfMeanpass3dGradFunctor<Device>()(
            context->eigen_device<Device>(),
            size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
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
            grad_par->flat<int>().data(),
            grad_didx->flat<int>().data(),
            buffers_full,
            buffers_imgs
        );
        
        //deallocate buffers - done automatically at lower levels
        //if(buffers_full != NULL)
        //    free(buffers_full);
        //if(buffers_imgs != NULL)
        //    free(buffers_imgs);
    }
};

REGISTER_OP("HmfMeanpass3d")
  .Input("data: float")
  .Input("rx: float")
  .Input("ry: float")
  .Input("rz: float")
  .Input("parentage: int32")
  .Input("data_index: int32")
  .Output("u: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs()-2; i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 5, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-2), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-1), 1, &input));
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("HmfMeanpass3dGrad")
  .Input("grad: float")
  .Input("data: float")
  .Input("rx: float")
  .Input("ry: float")
  .Input("rz: float")
  .Input("parentage: int32")
  .Input("data_index: int32")
  .Input("u: float")
  .Output("grad_data: float")
  .Output("grad_rx: float")
  .Output("grad_ry: float")
  .Output("grad_rz: float")
  .Output("grad_par: int32")
  .Output("grad_didx: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs()-3; i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 5, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-3), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-2), 1, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs()-1), 5, &input));
    for (size_t i = 0; i < c->num_inputs()-1; i++)
        c->set_output(i, c->input(i+1));
    return Status::OK();
  });


// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass3d").Device(DEVICE_CPU), HmfMeanpass3dOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass3dGrad").Device(DEVICE_CPU), HmfMeanpass3dGradOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass3d").Device(DEVICE_GPU), HmfMeanpass3dOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HmfMeanpass3dGrad").Device(DEVICE_GPU), HmfMeanpass3dGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA


