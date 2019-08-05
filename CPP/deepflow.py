import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import os

flow_path = os.path.dirname(os.path.realpath(__file__))
module = tf.load_op_library(flow_path + '/flow.so')
@ops.RegisterGradient("BinaryMeanpass3d")
def _binary_meanpass3d_grad_cc(op, grad):
    """
    The gradient for `binary_meanpass3d` using the operation implemented in C++.

    :param op: `binary_meanpass3d` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `binary_meanpass3d` op.
    :return: gradients with respect to the input of `binary_meanpass3d`.
    """

    return module.binary_meanpass3d_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.outputs[0])    

@ops.RegisterGradient("BinaryMeanpass2d")
def _binary_meanpass2d_grad_cc(op, grad):
    """
    The gradient for `binary_meanpass2d` using the operation implemented in C++.

    :param op: `binary_meanpass2d` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `binary_meanpass2d` op.
    :return: gradients with respect to the input of `binary_meanpass2d`.
    """

    return module.binary_meanpass2d_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.outputs[0])    

@ops.RegisterGradient("BinaryMeanpass1d")
def _binary_meanpass1d_grad_cc(op, grad):
    """
    The gradient for `binary_meanpass1d` using the operation implemented in C++.

    :param op: `binary_meanpass1d` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `binary_meanpass1d` op.
    :return: gradients with respect to the input of `binary_meanpass1d`.
    """

    return module.binary_meanpass1d_grad(grad, op.inputs[0], op.inputs[1], op.outputs[0])    

@ops.RegisterGradient("PottsMeanpass3d")
def _potts_meanpass3d_grad_cc(op, grad):
    """
    The gradient for `potts_meanpass3d` using the operation implemented in C++.

    :param op: `potts_meanpass3d` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `potts_meanpass3d` op.
    :return: gradients with respect to the input of `potts_meanpass3d`.
    """

    return module.potts_meanpass3d_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.outputs[0])    

@ops.RegisterGradient("PottsMeanpass2d")
def _potts_meanpass2d_grad_cc(op, grad):
    """
    The gradient for `potts_meanpass2d` using the operation implemented in C++.

    :param op: `potts_meanpass2d` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `potts_meanpass2d` op.
    :return: gradients with respect to the input of `potts_meanpass2d`.
    """

    return module.potts_meanpass2d_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.outputs[0])    

@ops.RegisterGradient("PottsMeanpass1d")
def _potts_meanpass1d_grad_cc(op, grad):
    """
    The gradient for `potts_meanpass1d` using the operation implemented in C++.

    :param op: `potts_meanpass1d` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `potts_meanpass1d` op.
    :return: gradients with respect to the input of `potts_meanpass1d`.
    """

    return module.potts_meanpass1d_grad(grad, op.inputs[0], op.inputs[1], op.outputs[0])    

@ops.RegisterGradient("HmfMeanpass3d")
def _hmf_meanpass3d_grad_cc(op, grad):
    """
    The gradient for `hmf_meanpass3d` using the operation implemented in C++.

    :param op: `hmf_meanpass3d` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `hmf_meanpass3d` op.
    :return: gradients with respect to the input of `hmf_meanpass3d`.
    """
    gradient = module.hmf_meanpass3d_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4], op.inputs[5], op.outputs[0])
    return gradient

@ops.RegisterGradient("HmfMeanpass2d")
def _hmf_meanpass2d_grad_cc(op, grad):
    """
    The gradient for `hmf_meanpass2d` using the operation implemented in C++.

    :param op: `hmf_meanpass2d` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `hmf_meanpass2d` op.
    :return: gradients with respect to the input of `hmf_meanpass2d`.
    """
    gradient = module.hmf_meanpass3d_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4], op.outputs[0])
    return gradient

@ops.RegisterGradient("HmfMeanpass1d")
def _hmf_meanpass1d_grad_cc(op, grad):
    """
    The gradient for `hmf_meanpass3d` using the operation implemented in C++.

    :param op: `hmf_meanpass1d` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `hmf_meanpass1d` op.
    :return: gradients with respect to the input of `hmf_meanpass1d`.
    """
    gradient = module.hmf_meanpass3d_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.outputs[0])
    return gradient


