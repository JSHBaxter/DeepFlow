/// \file hmf_auglag1d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the augmented Lagrangian solver for an HMF 
/// segmentation model operation in Tensorflow.

#include "hmf_auglag1d.h"
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
#include "hmf_auglag1d_cpu_solver.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "hmf_auglag1d_gpu_solver.cc"
#endif

// Define the OpKernel class
// template parameter <float> is the datatype of the tensors.
template <typename Device>
class HmfAuglag1dOp : public OpKernel {
public:
    explicit HmfAuglag1dOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        
        // ensure all inputs are present
        DCHECK_EQ(4, context->num_inputs());

        // get the input tensors
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* parentage = &(context->input(2));
        const Tensor* data_index = &(context->input(3));

        // Ensure tensor is small enough to function
        OP_REQUIRES(context, data_cost->NumElements() <= tensorflow::kint32max / 16,
                    errors::InvalidArgument("Too many elements in tensor"));
        
        // check shapes of input and weights
        const DataType data_type = data_cost->dtype();
        const TensorShape& data_shape = data_cost->shape();
        const TensorShape& rx_shape = rx_cost->shape();
        const TensorShape& parentage_shape = parentage->shape();
        const TensorShape& data_index_shape = data_index->shape();
        int size_array[5] = {(int) data_shape.dim_size(0),
                             (int) data_shape.dim_size(1),
                             (int) data_shape.dim_size(2),
                             (int) rx_shape.dim_size(1),
                             (int) rx_shape.dim_size(2)};

        // check input is of correct rank
        DCHECK_EQ(data_shape.dims(), 3);
        DCHECK_EQ(rx_shape.dims(), 3);

        // check input is of correct size
        DCHECK_EQ(data_shape.dim_size(0), rx_shape.dim_size(0));

        // create output tensor
        Tensor* u = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, data_shape, &u));

        // create intermediate buffers as needed
        int n_s = size_array[2]*size_array[4];
        int n_i = size_array[2];
        int num_intermediates_full = HmfAuglag1dFunctor<Device>().num_buffers_full();
        int num_intermediates_images = HmfAuglag1dFunctor<Device>().num_buffers_images();
        float** buffers_full = NULL;
        float** buffers_imgs = NULL;
        get_temporary_buffers(context, buffers_full, n_s, num_intermediates_full, buffers_imgs, n_i, num_intermediates_images, data_cost);
        
        // call function
        HmfAuglag1dFunctor<Device>()(
            context->eigen_device<Device>(),
            size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
        
        //deallocate buffers
        clear_temporary_buffers(context, buffers_full, n_s, num_intermediates_full);
        clear_temporary_buffers(context, buffers_imgs, n_i, num_intermediates_images);
    }
};

REGISTER_OP("HmfAuglag1d")
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

// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("HmfAuglag1d").Device(DEVICE_CPU), HmfAuglag1dOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("HmfAuglag1d").Device(DEVICE_GPU), HmfAuglag1dOp<GPUDevice>);
#endif  // GOOGLE_CUDA


