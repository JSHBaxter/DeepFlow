#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.
"""

import unittest
import numpy as np
import tensorflow as tf
import deepflow

flow_module = deepflow.module

class InnerProductOpTest(unittest.TestCase):
    def test_runAndPrintOutput(self):
        
        b = 16
        x = 16*16*16
        c = 4
        t = 8
        iters = 1000

        for i in range(1):
            coeffs_i = np.random.randn(c,t)*0.01

            for devicename in ['CPU','GPU']:
                
                if devicename == 'CPU':
                    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 0})
                else:
                    config = tf.ConfigProto(log_device_placement=False)

                data = tf.placeholder(tf.float32, shape = (b,x,c))
                coeffs = tf.placeholder(tf.float32, shape = (c,t))
                grad = tf.placeholder(tf.float32, shape = (b,x,c))
                
                with tf.Session(config=config) as sess:
                    
                    if devicename == 'CPU':
                        taylor_series = flow_module.taylor_series_nsc(data,coeffs)
                        taylor_series_grad = flow_module.taylor_series_grad_nsc(data,coeffs,grad)
                    else:
                        taylor_series = flow_module.taylor_series_nsc(data,coeffs)
                        taylor_series_grad = flow_module.taylor_series_grad_nsc(data,coeffs,grad)
                    
                    coeffs_d = coeffs_i.copy()
                    print(coeffs_d)
                    
                    for iter in range(iters):
                        input_d = np.random.randn(b,x,c)*2
                        output_d = np.exp(input_d)
                        output_ts = sess.run(taylor_series, feed_dict = {data: input_d, coeffs: coeffs_d})
                        grad_d = np.sign(output_d-output_ts)/(b*c*x)
                        grad_input, grad_coeffs = sess.run(taylor_series_grad, feed_dict = {data: input_d, coeffs: coeffs_d, grad: grad_d})
                        coeffs_d += 0.1*grad_coeffs
                        
                        print(coeffs_d.T)


if __name__ == '__main__':
    print(dir(flow_module))
    unittest.main()
