/// \file hmf_auglag2d.cc
/// \author John S.H. Baxter
/// \brief Implementation of the augmented Lagrangian solver for an HMF 
/// segmentation model operation in Tensorflow.

#include "hmfNd.h"
#include "hmf_auglag2d.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <math.h>
#include <iostream>
using namespace tensorflow;

// Load the CPU kernels
using CPUDevice = Eigen::ThreadPoolDevice;
#include "hmf_auglag2d_cpu_solver.cc"

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
using GPUDevice = Eigen::GpuDevice;
#include "hmf_auglag2d_gpu_solver.cc"
#endif

// Define the OpKernel class
template <typename Device>
class HmfAuglag2dOp : public HmfNdOp<Device> {
public:
    HmfAuglag2dOp(OpKernelConstruction* context) : 
		HmfNdOp<Device>(context, 2, 3, 1, std::is_same<Device, GPUDevice>::value, false)	{}

protected:

    int Get_Num_Intermediates_Full() override {
        return HmfAuglag2dFunctor<Device>().num_buffers_full();
    }
    int Get_Num_Intermediates_Images() override {
        return HmfAuglag2dFunctor<Device>().num_buffers_images();
    }
    
    void CallFunction(OpKernelContext* context, float** buffers_full, float** buffers_imgs) override {
    
        const Tensor* data_cost = &(context->input(0));
        const Tensor* rx_cost = &(context->input(1));
        const Tensor* ry_cost = &(context->input(2));
        const Tensor* parentage = &(context->input(3));
        const Tensor* data_index = &(context->input(4));
        Tensor* u = this->outputs[0];
		
        // call function
        HmfAuglag2dFunctor<Device>()(
            context->eigen_device<Device>(),
            this->size_array,
            parentage->flat<int>().data(),
            data_index->flat<int>().data(),
            data_cost->flat<float>().data(),
            rx_cost->flat<float>().data(),
            ry_cost->flat<float>().data(),
            u->flat<float>().data(),
            buffers_full,
            buffers_imgs
        );
    }
};

REGISTER_OP("HmfAuglag2d")
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

// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("HmfAuglag2d").Device(DEVICE_CPU), HmfAuglag2dOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("HmfAuglag2d").Device(DEVICE_GPU), HmfAuglag2dOp<GPUDevice>);
#endif  // GOOGLE_CUDA
