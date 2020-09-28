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
        
        b = 1
        c = 4
        x = 10
        t = 32

        for i in range(1):
            input_d = 10*np.random.randn(b,c,x)
            output_d = np.exp(input_d)
            coeffs_d = np.ones((c,t))

            for devicename in ['CPU','GPU']:
                
                if devicename == 'CPU':
                    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 0})

                    data = tf.placeholder(tf.float32, shape = (b,x,c))
                    coeffs = tf.placeholder(tf.float32, shape = (c,t))

                else:
                    config = tf.ConfigProto(log_device_placement=False)

                    data = tf.placeholder(tf.float32, shape = (b,c,x))
                    coeffs = tf.placeholder(tf.float32, shape = (c,t))
                    
                
                with tf.Session(config=config) as sess:
                    
                    if devicename == 'CPU':
                        flow = flow_module.taylor_series_nsc(data,coeffs)
                    else:
                        flow = flow_module.taylor_series_ncs(data,coeffs)
                        
                    if devicename == 'CPU':
                        output_ts = sess.run(flow, feed_dict = {data: np.transpose(input_d,[0,2,1]), coeffs: coeffs_d})
                    else:
                        output_ts = sess.run(flow, feed_dict = {data: input_d, coeffs: coeffs_d})
            
                if devicename == 'CPU':
                    print("CPU")
                    print(input_d[0,:,:])
                    print(output_d[0,:,:])
                    print(np.transpose(output_ts,[0,2,1])[0,:,:])
                    print((output_d-np.transpose(output_ts,[0,2,1]))[0,:,:])
                else:
                    print("GPU")
                    print(input_d[0,:,:])
                    print(output_d[0,:,:])
                    print(output_ts[0,:,:])
                    print((output_d-output_ts)[0,:,:])


if __name__ == '__main__':
    print(dir(flow_module))
    unittest.main()
