import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import os

flow_path = os.path.dirname(os.path.realpath(__file__))
module = tf.load_op_library(flow_path + '/flow.so')

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


