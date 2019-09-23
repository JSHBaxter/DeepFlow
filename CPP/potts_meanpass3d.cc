/// \file potts_meanpass3d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the mean-field message passing approximate solver for a Potts 
/// segmentation model operation in Tensorflow.

#include "potts_meanpass3d.h"
#include "potts_meanpassNd.h"
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
#include "potts_meanpass3d_cpu_solver.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "potts_meanpass3d_gpu_solver.cc"
#endif

// Define the OpKernel class
template <typename Device>
class PottsMeanpass3dOp : public PottsMeanpassNdOp<Device> {
public:
    explicit PottsMeanpass3dOp(OpKernelConstruction* context) :
        PottsMeanpassNdOp<Device>(context, 3, 0) {}

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
        Tensor* new_u = this->u;
        
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
// template parameter <float> is the datatype of the tensors.
template <typename Device>
class PottsMeanpass3dGradOp : public OpKernel {
public:
    explicit PottsMeanpass3dGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        
        // ensure all inputs are present
        DCHECK_EQ(6, context->num_inputs());

        // get the input tensors
        const Tensor* grad = &(context->input(0));
        const Tensor* data_cost = &(context->input(1));
        const Tensor* rx_cost = &(context->input(2));
        const Tensor* ry_cost = &(context->input(3));
        const Tensor* rz_cost = &(context->input(4));
        const Tensor* u = &(context->input(5));

        // Ensure tensor is small enough to function
        OP_REQUIRES(context, data_cost->NumElements() <= tensorflow::kint32max / 16,
                    errors::InvalidArgument("Too many elements in tensor"));
        
        // check shapes of input and weights
        const DataType data_type = data_cost->dtype();
        const TensorShape& data_shape = data_cost->shape();
        const TensorShape& rx_shape = rx_cost->shape();
        const TensorShape& ry_shape = ry_cost->shape();
        const TensorShape& rz_shape = rz_cost->shape();
        int size_array[5] = {(int) data_shape.dim_size(0),
                             (int) data_shape.dim_size(1),
                             (int) data_shape.dim_size(2),
                             (int) data_shape.dim_size(3),
                             (int) data_shape.dim_size(4)};

        // check input is of rank 5
        DCHECK_EQ(data_shape.dims(), 5);
        DCHECK_EQ(rx_shape.dims(), 5);
        DCHECK_EQ(ry_shape.dims(), 5);
        DCHECK_EQ(rz_shape.dims(), 5);

        // check input is of correct size
        // i.e. same for dim 0 and 1, rx is 1 smaller in 2, ry is 1 smaller in 3
        DCHECK_EQ(data_shape.dim_size(0), rx_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(1), rx_shape.dim_size(1));
        DCHECK_EQ(data_shape.dim_size(2), rx_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(3), rx_shape.dim_size(3));
        DCHECK_EQ(data_shape.dim_size(4), rx_shape.dim_size(4));
        DCHECK_EQ(data_shape.dim_size(0), ry_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(1), ry_shape.dim_size(1));
        DCHECK_EQ(data_shape.dim_size(2), ry_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(3), ry_shape.dim_size(3));
        DCHECK_EQ(data_shape.dim_size(4), ry_shape.dim_size(4));
        DCHECK_EQ(data_shape.dim_size(0), rz_shape.dim_size(0));
        DCHECK_EQ(data_shape.dim_size(1), rz_shape.dim_size(1));
        DCHECK_EQ(data_shape.dim_size(2), rz_shape.dim_size(2));
        DCHECK_EQ(data_shape.dim_size(3), rz_shape.dim_size(3));
        DCHECK_EQ(data_shape.dim_size(4), rz_shape.dim_size(4));
        
        //get output tensors
        Tensor* grad_data = NULL;
        Tensor* grad_rx = NULL;
        Tensor* grad_ry = NULL;
        Tensor* grad_rz = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, data_shape, &grad_data));
        OP_REQUIRES_OK(context, context->allocate_output(1, rx_shape, &grad_rx));
        OP_REQUIRES_OK(context, context->allocate_output(2, ry_shape, &grad_ry));
        OP_REQUIRES_OK(context, context->allocate_output(3, rz_shape, &grad_rz));
        
        // create intermediate buffers as needed
        int n_s = size_array[1]*size_array[2]*size_array[3]*size_array[4];
        int n_i = size_array[2]*size_array[3]*size_array[4];
        int num_intermediates_full = PottsMeanpass3dGradFunctor<Device>().num_buffers_full();
        int num_intermediates_images = PottsMeanpass3dGradFunctor<Device>().num_buffers_images();
        float** buffers_full = NULL;
        float** buffers_imgs = NULL;
        get_temporary_buffers(context, buffers_full, n_s, num_intermediates_full, buffers_imgs, n_i, num_intermediates_images, data_cost);
        
        // call function for gradient
        PottsMeanpass3dGradFunctor<Device>()(
            context->eigen_device<Device>(),
            size_array,
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
        
        //deallocate buffers
        clear_temporary_buffers(context, buffers_full, n_s, num_intermediates_full);
        clear_temporary_buffers(context, buffers_imgs, n_i, num_intermediates_images);
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


// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass3d").Device(DEVICE_CPU), PottsMeanpass3dOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass3dGrad").Device(DEVICE_CPU), PottsMeanpass3dGradOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass3d").Device(DEVICE_GPU), PottsMeanpass3dOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PottsMeanpass3dGrad").Device(DEVICE_GPU), PottsMeanpass3dGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA


