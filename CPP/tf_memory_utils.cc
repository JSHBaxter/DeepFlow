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

void clear_temporary_buffers(OpKernelContext* context, float** buffers, const int size, const int number){
    if( buffers != NULL )
        delete buffers;
}