/// \file taylor_series.cc
/// \author John S.H. Baxter
/// \brief Implementation of a Taylor series activation layer

#include "taylor_series.h"
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

// If we are using CUDA, include the GPU kernels
#if GOOGLE_CUDA
#include "taylor_series_gpu.cc"
#endif
#include "taylor_series_cpu.cc"

// Define the OpKernel class
template <typename Device, bool channels_first>
class TaylorSeriesOp : public OpKernel {
private:
	int* size_data;
	int* size_coeffs;
	int N;
	Tensor* output;
	
public:

    explicit TaylorSeriesOp(OpKernelConstruction* context) :
        OpKernel(context),
		size_data(NULL),
		size_coeffs(NULL),
		output(NULL){}

	~TaylorSeriesOp(){
		if(this->size_data)
			delete this->size_data;
		if(this->size_coeffs)
			delete this->size_coeffs;
	}

protected:

    void CallFunction(OpKernelContext* context) {
		//get input and output
        const Tensor* input_data = &(context->input(0));
        const Tensor* coeff_data = &(context->input(1));
		
		//call the actual operator
		TaylorSeriesFunctor<Device>()(
			context->eigen_device<Device>(),
			channels_first,
			this->size_data,
			this->size_coeffs,
			this->N,
			input_data->flat<float>().data(),
			coeff_data->flat<float>().data(),
			output->flat<float>().data()
		);
		
    }
	
	void CheckInputs(OpKernelContext* context){
		
		// ensure all inputs are present
		DCHECK_EQ(2, context->num_inputs());

		// get the input tensors
		const Tensor* data_input = &(context->input(0));
		const Tensor* coeff_input = &(context->input(1));

		// Ensure tensor is small enough to function
		OP_REQUIRES(context, data_input->NumElements() <= tensorflow::kint32max / 16,
					errors::InvalidArgument("Too many elements in tensor"));

		// check input ranks (exactly 2 for coeff, more than 1 for data)
		const DataType data_type = data_input->dtype();
		const TensorShape& data_shape = data_input->shape();
		const DataType coeff_type = coeff_input->dtype();
		const TensorShape& coeff_shape = coeff_input->shape();
		DCHECK_GT(data_shape.dims(), 1);
		DCHECK_EQ(coeff_shape.dims(), 2);
		this->N = data_shape.dims();
		if(channels_first)
			DCHECK_EQ(coeff_shape.dim_size(0), data_shape.dim_size(1));
		else
			DCHECK_EQ(coeff_shape.dim_size(0), data_shape.dim_size(data_shape.dims()-1));
		
		// populate size array
		if(this->size_data)
			delete this->size_data;
		if(this->size_coeffs)
			delete this->size_coeffs;
		this->size_data = new int[data_shape.dims()];
		this->size_coeffs = new int[coeff_shape.dims()];
		for(int i = 0; i < data_shape.dims(); i++)
			this->size_data[i] = data_shape.dim_size(i);
		for(int i = 0; i < coeff_shape.dims(); i++)
			this->size_coeffs[i] = coeff_shape.dim_size(i);
		
	}
	
	void GetOutputTensors(OpKernelContext* context){
		OP_REQUIRES_OK(context, context->allocate_output(0, (&(context->input(0)))->shape(), &(this->output)));
	}
	
	void Compute(OpKernelContext* context){
		this->CheckInputs(context);
		this->GetOutputTensors(context);
		this->CallFunction(context);
	}
};

// Define the OpKernel class
template <typename Device, bool channels_first>
class TaylorSeriesGradOp : public OpKernel {
private:
	int* size_data;
	int* size_coeffs;
	int N;
	Tensor* g_input;
	Tensor* g_coeffs;
	
public:

    explicit TaylorSeriesGradOp(OpKernelConstruction* context) :
        OpKernel(context),
		size_data(NULL),
		size_coeffs(NULL),
		g_input(NULL),
		g_coeffs(NULL){}

	~TaylorSeriesGradOp(){
		if(this->size_data)
			delete this->size_data;
		if(this->size_coeffs)
			delete this->size_coeffs;
	}

protected:

    void CallFunction(OpKernelContext* context) {
		//get input and output
        const Tensor* input_data = &(context->input(0));
        const Tensor* coeff_data = &(context->input(1));
        const Tensor* grad_data = &(context->input(2));
		
		//call the actual operator
		TaylorSeriesGradFunctor<Device>()(
			context->eigen_device<Device>(),
			channels_first,
			this->size_data,
			this->size_coeffs,
			this->N,
			input_data->flat<float>().data(),
			coeff_data->flat<float>().data(),
			grad_data->flat<float>().data(),
			g_input->flat<float>().data(),
			g_coeffs->flat<float>().data()
		);
		
    }
	
	void CheckInputs(OpKernelContext* context){
		
		// ensure all inputs are present
		DCHECK_EQ(3, context->num_inputs());

		// get the input tensors
		const Tensor* data_input = &(context->input(0));
		const Tensor* coeff_input = &(context->input(1));
		const Tensor* grad_input = &(context->input(2));

		// Ensure tensor is small enough to function
		OP_REQUIRES(context, data_input->NumElements() <= tensorflow::kint32max / 16,
					errors::InvalidArgument("Too many elements in tensor"));

		// check input ranks (exactly 2 for coeff, more than 1 for data)
		const DataType data_type = data_input->dtype();
		const TensorShape& data_shape = data_input->shape();
		const DataType coeff_type = coeff_input->dtype();
		const TensorShape& coeff_shape = coeff_input->shape();
		const DataType grad_type = grad_input->dtype();
		const TensorShape& grad_shape = grad_input->shape();
		DCHECK_GT(data_shape.dims(), 1);
		DCHECK_EQ(data_shape.dims(), grad_shape.dims());
		DCHECK_EQ(coeff_shape.dims(), 2);
		this->N = data_shape.dims();
		for(int i = 0; i < this->N; i++)
			DCHECK_EQ(data_shape.dim_size(i), grad_shape.dim_size(i));
		if(channels_first)
			DCHECK_EQ(coeff_shape.dim_size(0), data_shape.dim_size(1));
		else
			DCHECK_EQ(coeff_shape.dim_size(0), data_shape.dim_size(data_shape.dims()-1));
		
		// populate size array
		if(this->size_data)
			delete this->size_data;
		if(this->size_coeffs)
			delete this->size_coeffs;
		this->size_data = new int[data_shape.dims()];
		this->size_coeffs = new int[coeff_shape.dims()];
		for(int i = 0; i < data_shape.dims(); i++)
			this->size_data[i] = data_shape.dim_size(i);
		for(int i = 0; i < coeff_shape.dims(); i++)
			this->size_coeffs[i] = coeff_shape.dim_size(i);
		
	}
	
	void GetOutputTensors(OpKernelContext* context){
		OP_REQUIRES_OK(context, context->allocate_output(0, (&(context->input(0)))->shape(), &(this->g_input)));
		OP_REQUIRES_OK(context, context->allocate_output(1, (&(context->input(1)))->shape(), &(this->g_coeffs)));
	}
	
	void Compute(OpKernelContext* context){
		this->CheckInputs(context);
		this->GetOutputTensors(context);
		this->CallFunction(context);
	}
};

#define _TAYLOR_SERIES_REGISTER_OP_(name_str) \
REGISTER_OP(name_str)\
  .Input("input: float")\
  .Input("coeff: float")\
  .Output("output: float")\
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {\
    ::tensorflow::shape_inference::ShapeHandle input;\
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input));\
    c->set_output(0, c->input(0));\
    return Status::OK();\
  });
_TAYLOR_SERIES_REGISTER_OP_("TaylorSeriesNCS");
_TAYLOR_SERIES_REGISTER_OP_("TaylorSeriesNSC");
  
#define _TAYLOR_SERIES_GRAD_REGISTER_OP_(name_str) \
REGISTER_OP(name_str)\
  .Input("input: float")\
  .Input("coeff: float")\
  .Input("grad: float")\
  .Output("g_input: float")\
  .Output("g_coeffs: float")\
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {\
    ::tensorflow::shape_inference::ShapeHandle input;\
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input));\
    c->set_output(0, c->input(0));\
    c->set_output(1, c->input(1));\
    return Status::OK();\
  });
_TAYLOR_SERIES_GRAD_REGISTER_OP_("TaylorSeriesGradNCS");
_TAYLOR_SERIES_GRAD_REGISTER_OP_("TaylorSeriesGradNSC");

// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("TaylorSeriesNCS").Device(DEVICE_CPU), TaylorSeriesOp<CPUDevice,true>);
REGISTER_KERNEL_BUILDER(Name("TaylorSeriesNSC").Device(DEVICE_CPU), TaylorSeriesOp<CPUDevice,false>);
REGISTER_KERNEL_BUILDER(Name("TaylorSeriesGradNCS").Device(DEVICE_CPU), TaylorSeriesGradOp<CPUDevice,true>);
REGISTER_KERNEL_BUILDER(Name("TaylorSeriesGradNSC").Device(DEVICE_CPU), TaylorSeriesGradOp<CPUDevice,false>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("TaylorSeriesNCS").Device(DEVICE_GPU), TaylorSeriesOp<GPUDevice,true>);
REGISTER_KERNEL_BUILDER(Name("TaylorSeriesNSC").Device(DEVICE_GPU), TaylorSeriesOp<GPUDevice,false>);
REGISTER_KERNEL_BUILDER(Name("TaylorSeriesGradNCS").Device(DEVICE_GPU), TaylorSeriesGradOp<GPUDevice,true>);
REGISTER_KERNEL_BUILDER(Name("TaylorSeriesGradNSC").Device(DEVICE_GPU), TaylorSeriesGradOp<GPUDevice,false>);
#endif  // GOOGLE_CUDA
