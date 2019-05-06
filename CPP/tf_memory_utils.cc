#include "tf_memory_utils.h"

using namespace tensorflow;

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

void get_temporary_buffers(OpKernelContext* context, float**& buffers_full, const int size, const int number, const Tensor* d){
    buffers_full = (number > 0) ? new float*[number]: NULL;

    Tensor buffer;
    TensorShape full_shape;
    full_shape.AddDim(size*number);
    OP_REQUIRES_OK(context, context->allocate_temp(d->dtype(), full_shape, &buffer));
    float* data_ptr = buffer.flat<float>().data();
    for(int b = 0; b < number; b++)
        buffers_full[b] = data_ptr + b*size;
    
}

void get_temporary_buffers(OpKernelContext* context, float**& buffers1, const int size1, const int number1, float**& buffers2, const int size2, const int number2, const Tensor* d){
    buffers1 = (number1 > 0) ? new float*[number1]: NULL;
    buffers2 = (number2 > 0) ? new float*[number2]: NULL;

    Tensor buffer;
    TensorShape full_shape;
    full_shape.AddDim(size1*number1+size2*number2);
    OP_REQUIRES_OK(context, context->allocate_temp(d->dtype(), full_shape, &buffer));
    float* data_ptr = buffer.flat<float>().data();
    for(int b = 0; b < number1; b++, data_ptr += b*size1)
        buffers1[b] = data_ptr;
    for(int b = 0; b < number2; b++, data_ptr += b*size2)
        buffers2[b] = data_ptr;
    
}

void clear_temporary_buffers(OpKernelContext* context, float** buffers, const int size, const int number){
    if( buffers != NULL )
        delete buffers;
}