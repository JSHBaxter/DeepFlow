/// \file potts_meanpassNd.cc
/// \author John S.H. Baxter
/// \brief Implementation of the mean-field message passing approximate solver for a Potts 
/// segmentation model operation in Tensorflow.

#include "potts_meanpassNd.h"
#include "tf_memory_utils.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <math.h>
#include <iostream>
using namespace tensorflow;

template <typename Device>
PottsMeanpassNdOp<Device>::PottsMeanpassNdOp(OpKernelConstruction* context, int N, int with_init) : 
    OpKernel(context),
    N(N),
    n_s(0),
    n_i(0),
    size_array(0),
    u(NULL),
    I(with_init)
{}

template <typename Device>
PottsMeanpassNdOp<Device>::~PottsMeanpassNdOp(){
    if(this->size_array)
        delete this->size_array;
}

template <typename Device>
void PottsMeanpassNdOp<Device>::Compute(OpKernelContext* context) {
    this->CheckInputs(context);
    this->GetOutputTensors(context);
    
    //allocate temprary buffers
    float** buffers_full = NULL;
    get_temporary_buffers(context, buffers_full, this->n_s, this->Get_Num_Intermediates_Full(), &(context->input(0)));
    float** buffers_imgs = NULL;
    get_temporary_buffers(context, buffers_imgs, this->n_i, this->Get_Num_Intermediates_Images(), &(context->input(0)));
    
    //pass down to child to find and run method
    this->CallFunction(context, buffers_full, buffers_imgs);
        
    //deallocate buffers
    clear_temporary_buffers(context, buffers_full, this->n_s, this->Get_Num_Intermediates_Full());
    clear_temporary_buffers(context, buffers_imgs, this->n_i, this->Get_Num_Intermediates_Images());
}

template <typename Device>
void PottsMeanpassNdOp<Device>::CheckInputs(OpKernelContext* context) {

    // ensure all inputs are present
    DCHECK_EQ(N+I+1, context->num_inputs());

    // get the input tensors
    const Tensor* data_cost = &(context->input(0));

    // Ensure tensor is small enough to function
    OP_REQUIRES(context, data_cost->NumElements() <= tensorflow::kint32max / 16,
                errors::InvalidArgument("Too many elements in tensor"));

    // check input is of rank N+2
    const DataType data_type = data_cost->dtype();
    const TensorShape& data_shape = data_cost->shape();
    DCHECK_EQ(data_shape.dims(), N+2);
    for(int i = 0; i < N+I; i++)
        DCHECK_EQ((&(context->input(i+1)))->shape().dims(), N+2);

    // check shapes of input and weights
    for(int i = 0; i < N+I; i++) {
        const TensorShape& other_shape = (&(context->input(i+1)))->shape();
        for(int j = 0; j < N+2; j++)
            DCHECK_EQ(data_shape.dim_size(j), other_shape.dim_size(j));
    }

    // populate size array structure
    this->size_array = new int[N+2];
    for(int i = 0; i < N+2; i++)
        this->size_array[i] = (int) data_shape.dim_size(i);
    this->n_s = 1;
    this->n_i = 1;
    for(int i = 1; i < N+2; i++)
        this->n_s *= size_array[i];
    for(int i = 2; i < N+2; i++)
        this->n_i = size_array[2]*size_array[3]*size_array[4];
        
}

template <typename Device>
void PottsMeanpassNdOp<Device>::GetOutputTensors(OpKernelContext* context) {
    u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, (&(context->input(0)))->shape(), &u));
}



template <typename Device>
PottsMeanpassNdGradOp<Device>::PottsMeanpassNdGradOp(OpKernelConstruction* context, int N, int with_init) : 
    OpKernel(context),
    N(N),
    n_s(0),
    n_i(0),
    size_array(0),
    grads(0),
    I(with_init)
{}

template <typename Device>
PottsMeanpassNdGradOp<Device>::~PottsMeanpassNdGradOp(){
    if(this->size_array)
        delete this->size_array;
    if(this->grads)
        delete this->grads;
}


template <typename Device>
void PottsMeanpassNdGradOp<Device>::Compute(OpKernelContext* context) {
    this->CheckInputs(context);
    this->GetOutputTensors(context);
    
    //allocate temprary buffers
    float** buffers_full = NULL;
    get_temporary_buffers(context, buffers_full, this->n_s, this->Get_Num_Intermediates_Full(), &(context->input(0)));
    float** buffers_imgs = NULL;
    get_temporary_buffers(context, buffers_imgs, this->n_i, this->Get_Num_Intermediates_Images(), &(context->input(0)));
    
    //pass down to child to find and run method
    this->CallFunction(context, buffers_full, buffers_imgs);
        
    //deallocate buffers
    clear_temporary_buffers(context, buffers_full, this->n_s, this->Get_Num_Intermediates_Full());
    clear_temporary_buffers(context, buffers_imgs, this->n_i, this->Get_Num_Intermediates_Images());
}

template <typename Device>
void PottsMeanpassNdGradOp<Device>::CheckInputs(OpKernelContext* context) {

    // ensure all inputs are present
    DCHECK_EQ(N+I+3, context->num_inputs());

    // get the input tensors
    const Tensor* data_cost = &(context->input(0));

    // Ensure tensor is small enough to function
    OP_REQUIRES(context, data_cost->NumElements() <= tensorflow::kint32max / 16,
                errors::InvalidArgument("Too many elements in tensor"));

    // check input is of rank N+2
    const DataType data_type = data_cost->dtype();
    const TensorShape& data_shape = data_cost->shape();
    DCHECK_EQ(data_shape.dims(), N+2);
    for(int i = 0; i < N+I+2; i++)
        DCHECK_EQ((&(context->input(i+1)))->shape().dims(), N+2);

    // check shapes of input and weights
    for(int i = 0; i < N+I+2; i++) {
        const TensorShape& other_shape = (&(context->input(i+1)))->shape();
        for(int j = 0; j < N+2; j++)
            DCHECK_EQ(data_shape.dim_size(j), other_shape.dim_size(j));
    }

    // populate size array structure
    this->size_array = new int[N+2];
    for(int i = 0; i < N+2; i++)
        this->size_array[i] = (int) data_shape.dim_size(i);
    this->n_s = 1;
    this->n_i = 1;
    for(int i = 1; i < N+2; i++)
        this->n_s *= size_array[i];
    for(int i = 2; i < N+2; i++)
        this->n_i = size_array[2]*size_array[3]*size_array[4];
        
}

template <typename Device>
void PottsMeanpassNdGradOp<Device>::GetOutputTensors(OpKernelContext* context) {
    if(grads)
        delete grads;
    grads = new Tensor*[N+I+1];
    for(int i = 0; i < N+I+1; i++)
        OP_REQUIRES_OK(context, context->allocate_output(i, (&(context->input(0)))->shape(), &(grads[i])));
}
