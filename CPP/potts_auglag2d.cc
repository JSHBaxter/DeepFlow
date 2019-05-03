/// \file potts_auglag2d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the augmented Lagrangian solver for a Potts 
/// segmentation model operation in Tensorflow.

#include "potts_auglag2d.h"
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
#include "potts_auglag2d_cpu_solver.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "potts_auglag2d_gpu_solver.cc"
#endif

// Define the OpKernel class
// template parameter <float> is the datatype of the tensors.
template <typename Device>
class PottsAuglag2dOp : public OpKernel {
public:
    explicit PottsAuglag2dOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        
        // ensure all inputs are present
        DCHECK_EQ(3, context->num_inputs());

        // get the input tensors
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));

        // Ensure tensor is small enough to function
        OP_REQUIRES(context, data_cost->NumElements() <= tensorflow::kint32max / 16,
                    errors::InvalidArgument("Too many elements in tensor"));
        
        // check shapes of input and weights
        const DataType data_type = data_cost->dtype();
        const TensorShape& data_shape = data_cost->shape();
        const TensorShape& rx_shape = rx_cost->shape();
        const TensorShape& ry_shape = ry_cost->shape();
        int size_array[4] = {(int) data_shape.dim_size(0),
                             (int) data_shape.dim_size(1),
                             (int) data_shape.dim_size(2),
                             (int) data_shape.dim_size(3)};

        // check input is of rank 5
        DCHECK_EQ(data_shape.dims(), 4);
        DCHECK_EQ(rx_shape.dims(), 4);
        DCHECK_EQ(ry_shape.dims(), 4);

        // check input is of correct size
        // i.e. same for dim 0 and 1, rx is 1 smaller in 2, ry is 1 smaller in 3
        DCHECK_EQ(data_shape.dim_size(0), rx_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(1), rx_shape.dim_size(1));
        DCHECK_EQ(data_shape.dim_size(2), rx_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(3), rx_shape.dim_size(3));
        DCHECK_EQ(data_shape.dim_size(0), ry_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(1), ry_shape.dim_size(1));
        DCHECK_EQ(data_shape.dim_size(2), ry_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(3), ry_shape.dim_size(3));

        // create output tensor
        Tensor* u = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, data_shape, &u));

        // create intermediate buffers as needed
        int n_s = size_array[1]*size_array[2]*size_array[3];
        int n_i = size_array[2]*size_array[3];
        int num_intermediates_full = PottsAuglag2dFunctor<Device>().num_buffers_full();
        int num_intermediates_images = PottsAuglag2dFunctor<Device>().num_buffers_images();
        float** buffers_full = NULL;
        get_temporary_buffers(context, buffers_full, n_s, num_intermediates_full, data_cost);
        float** buffers_imgs = NULL;
        get_temporary_buffers(context, buffers_imgs, n_i, num_intermediates_images, data_cost);
        
        // call function
        PottsAuglag2dFunctor<Device>()(
            context->eigen_device<Device>(),
            size_array,
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

REGISTER_OP("PottsAuglag2d")
  .Input("data: float")
  .Input("rx: float")
  .Input("ry: float")
  .Output("u: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    for (size_t i = 0; i < c->num_inputs(); i++)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 4, &input));
    c->set_output(0, c->input(0));
    return Status::OK();
  });
// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("PottsAuglag2d").Device(DEVICE_CPU), PottsAuglag2dOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("PottsAuglag2d").Device(DEVICE_GPU), PottsAuglag2dOp<GPUDevice>);
#endif  // GOOGLE_CUDA


