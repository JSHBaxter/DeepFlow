/// \file hmf_auglag3d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the augmented Lagrangian solver for an HMF 
/// segmentation model operation in Tensorflow.

#include "hmfNd.h"
#include "hmf_auglag3d.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <math.h>
#include <iostream>
using namespace tensorflow;

// Load the CPU kernels
using CPUDevice = Eigen::ThreadPoolDevice;
#include "hmf_auglag3d_cpu_solver.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
using GPUDevice = Eigen::GpuDevice;
#include "hmf_auglag3d_gpu_solver.cc"
#endif

// Define the OpKernel class
template <typename Device>
class HmfAuglag3dOp : public HmfNdOp<Device> {
public:
    HmfAuglag3dOp(OpKernelConstruction* context) : 
		HmfNdOp<Device>(context, 3, 4, 1, std::is_same<Device, GPUDevice>::value, false)	{}

protected:

    int Get_Num_Intermediates_Full() override {
        return HmfAuglag3dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return HmfAuglag3dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* rz_cost = &(context->input(3));
        const Tensor* parentage = &(context->input(4));
        const Tensor* data_index = &(context->input(5));
        Tensor* u = this->outputs[0];
		
        // call function
        HmfAuglag3dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
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
    }
};

REGISTER_OP("HmfAuglag3d")
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

// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("HmfAuglag3d").Device(DEVICE_CPU), HmfAuglag3dOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("HmfAuglag3d").Device(DEVICE_GPU), HmfAuglag3dOp<GPUDevice>);
#endif  // GOOGLE_CUDA
