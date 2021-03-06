/// \file regularNd.cc
/// \author John S.H. Baxter
/// \brief Tensorflow-facing code for "regular" solvers, i.e. inputs are all same-sized

#include "regularNd.h"
#include "tf_memory_utils.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <math.h>
#include <iostream>
using namespace tensorflow;

template <typename Device>
RegularNdOp<Device>::RegularNdOp(OpKernelConstruction* context, int N, int num_inputs, int num_outputs) : 
    OpKernel(context),
    N(N),
	O(num_outputs),
	I(num_inputs),
    n_s(0),
    n_i(0),
    size_array(0),
    outputs(NULL)
{}

template <typename Device>
RegularNdOp<Device>::~RegularNdOp(){
    if(this->size_array)
        delete this->size_array;
    if(this->outputs)
        delete this->outputs;
}

template <typename Device>
void RegularNdOp<Device>::Compute(OpKernelContext* context) {
    this->CheckInputs(context);
    this->GetOutputTensors(context);
    
    //allocate temprary buffers
    float** buffers_full = NULL;
    float** buffers_imgs = NULL;
    get_temporary_buffers(context, buffers_full, this->n_s, this->Get_Num_Intermediates_Full(), 
                                   buffers_imgs, this->n_i, this->Get_Num_Intermediates_Images(), &(context->input(0)));
    
    //pass down to child to find and run method
    this->CallFunction(context, buffers_full, buffers_imgs);
        
    //deallocate buffers
    clear_temporary_buffers(context, buffers_full, this->n_s, this->Get_Num_Intermediates_Full());
    clear_temporary_buffers(context, buffers_imgs, this->n_i, this->Get_Num_Intermediates_Images());
}

template <typename Device>
void RegularNdOp<Device>::CheckInputs(OpKernelContext* context) {

    // ensure all inputs are present
    DCHECK_EQ(I, context->num_inputs());

    // get the input tensors
    const Tensor* data_cost = &(context->input(0));

    // Ensure tensor is small enough to function
    OP_REQUIRES(context, data_cost->NumElements() <= tensorflow::kint32max / 16,
                errors::InvalidArgument("Too many elements in tensor"));

    // check input is of rank N+2
    const DataType data_type = data_cost->dtype();
    const TensorShape& data_shape = data_cost->shape();
    for(int i = 0; i < I; i++)
        DCHECK_EQ((&(context->input(i)))->shape().dims(), N+2);

    // check shapes of input and weights
    for(int i = 1; i < I; i++) {
        const TensorShape& other_shape = (&(context->input(i)))->shape();
        for(int j = 0; j < N+2; j++)
            DCHECK_EQ(data_shape.dim_size(j), other_shape.dim_size(j));
    }

    // populate size array structure
    this->size_array = new int[N+2];
    for(int i = 0; i < N+2; i++)
        this->size_array[i] = (int) data_shape.dim_size(i);
    this->n_s = 1;
    for(int i = 1; i < N+2; i++)
        this->n_s *= size_array[i];
    this->n_i = 1;
    for(int i = 2; i < N+2; i++)
        this->n_i = size_array[i];
        
}

template <typename Device>
void RegularNdOp<Device>::GetOutputTensors(OpKernelContext* context) {
    if(outputs)
        delete outputs;
    outputs = new Tensor*[O];
    for(int i = 0; i < O; i++){
        outputs[i] = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(i, (&(context->input(0)))->shape(), &(outputs[i])));
    }
}
