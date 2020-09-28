#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.
"""

import unittest
import numpy as np
import tensorflow as tf
import deepflow

meanpass_module = deepflow.module

class InnerProductOpTest(unittest.TestCase):
    def test_runAndPrintOutput(self):
        
        #tf.eagerly()
        
        b = 1
        c = 4
        x = 20

        for i in range(1):
            input_d = 0.01*(np.random.rand(b,c,x)-0.5).astype(float)
            input_rx = 1*np.ones((b,c,x)).astype(float)
            #input_rx[:,:,:,:,2] *= 0
            #input_rx[:,:,:,:,3] *= 0
            #input_rx[:,:,:,:,4] *= 0
            #input_rx[:,:,:,:,5] *= 0
            #input_rx += 0.2*(np.random.rand(b,c,x))

            for devicename in ['/CPU:0','/GPU:0']:
                
                if devicename == '/CPU:0':
                    data = tf.convert_to_tensor(np.transpose(input_d,[0,2,1]),dtype=tf.float32)
                    rx = tf.convert_to_tensor(np.transpose(input_rx,[0,2,1]),dtype=tf.float32)

                else:
                    data = tf.convert_to_tensor(input_d,dtype=tf.float32)
                    rx = tf.convert_to_tensor(input_rx,dtype=tf.float32)
                    
                
                with tf.device(devicename):

                    forward = meanpass_module.binary_auglag1d(data,rx).numpy()
                    forward_mean = meanpass_module.binary_meanpass1d(data,rx).numpy()
                    #grad_flow = tf.gradients(flow, (data,rx))

                    if devicename == '/CPU:0':
                        forward = np.transpose(forward,[0,2,1])
                        forward_mean = np.transpose(forward_mean,[0,2,1])

                print( input_d.shape )
                print( (np.round(input_d[0,0,:]*100)).astype(int) )
                print( (np.round(input_d[0,1,:]*100)).astype(int) )
                print( (np.round(input_d[0,2,:]*100)).astype(int) )
                print( (np.round(input_d[0,3,:]*100)).astype(int) )

                print( '\n' )

                forward = np.exp(forward);
                print( forward.shape )
                print( (np.round(forward[0,0,:]*100)).astype(int) )
                print( (np.round(forward[0,1,:]*100)).astype(int) )
                print( (np.round(forward[0,2,:]*100)).astype(int) )
                print( (np.round(forward[0,3,:]*100)).astype(int) )
                print( '\n' )
                print( '\n' )

                forward_mean = 1.0 / (1.0 + np.exp(-forward_mean))
                print( forward_mean.shape )
                print( (np.round(forward_mean[0,0,:]*100)).astype(int) )
                print( (np.round(forward_mean[0,1,:]*100)).astype(int) )
                print( (np.round(forward_mean[0,2,:]*100)).astype(int) )
                print( (np.round(forward_mean[0,3,:]*100)).astype(int) )
                print( '\n' )
                print( '\n' )

                #sumprob = np.sum(forward,axis=1)
                #print( (np.round(sumprob[0,:]*100)).astype(int) )

                #forward0 = np.exp(forward0);
                #print( forward0.shape )
                #print( (np.round(forward0[0,0,:,:,:]*100)).astype(int) )
                #print( (np.round(forward0[0,1,:,:,:]*100)).astype(int) )
                #print( (np.round(forward0[0,2,:,:,:]*100)).astype(int) )
                #print( (np.round(forward0[0,3,:,:,:]*100)).astype(int) )

                #print( '\n' )

                #sumprob = np.sum(forward0,axis=1)
                #print( (np.round(sumprob[0,:,:,:]*100)).astype(int) )

                #print( gradient[0].shape )
                #print( (np.round(gradient[0][0,0,:,:,:]*100)).astype(int) )
                #print( (np.round(gradient[0][0,1,:,:,:]*100)).astype(int) )
                #print( (np.round(gradient[0][0,2,:,:,:]*100)).astype(int) )
                #print( (np.round(gradient[0][0,3,:,:,:]*100)).astype(int) )
                

                #np.testing.assert_array_equal(gradient_tf, gradient_inner_product)


if __name__ == '__main__':
    unittest.main()
