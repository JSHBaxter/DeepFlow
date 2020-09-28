/// \file binary_auglag3d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the augmented Lagrangian solver for a Binary 
/// segmentation model operation in Tensorflow.

#include "regularNd.h"
#include "binary_auglag3d.h"
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
#include "binary_auglag3d_cpu_solver.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "binary_auglag3d_gpu_solver.cc"
#endif

// Define the OpKernel class
template <typename Device>
class BinaryAuglag3dOp : public RegularNdOp<Device> {
public:
    explicit BinaryAuglag3dOp(OpKernelConstruction* context) :
        RegularNdOp<Device>(context, 3, 4, 1) {}

protected:

    int Get_Num_Intermediates_Full() override {
        return BinaryAuglag3dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return BinaryAuglag3dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* rz_cost = &(context->input(3));
        Tensor* u = this->outputs[0];
        
        // call function
        BinaryAuglag3dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            rz_cost->flat<float>().data(),
            u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

REGISTER_OP("BinaryAuglag3d")
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
// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("BinaryAuglag3d").Device(DEVICE_CPU), BinaryAuglag3dOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("BinaryAuglag3d").Device(DEVICE_GPU), BinaryAuglag3dOp<GPUDevice>);
#endif  // GOOGLE_CUDA
