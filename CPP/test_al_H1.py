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
        
        b = 1
        c = 4
        r = 2*c-2;
        x = 10

        parentage_list = []
        for i in range(r):
            parentage_list.append( int(i//2) - 1)
        print(parentage_list)
        data_index_list = [-1 for i in range(c-2)] + [i for i in range(c)]
        print(data_index_list)
        parentage = tf.convert_to_tensor(parentage_list, dtype=tf.int32)
        data_index = tf.convert_to_tensor(data_index_list, dtype=tf.int32)
        
        for i in range(1):
            input_d = 0.1*np.random.rand(b,c,x)
            input_rx = 2*np.ones((b,r,x))
            input_rx[:,0:c,:] = 2.00001
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
                    forward = meanpass_module.hmf_auglag1d(data,rx,parentage,data_index)
                    forward_mean = meanpass_module.hmf_meanpass1d(data,rx,parentage,data_index)
                    forward_mean_init = meanpass_module.hmf_meanpass1d_with_init(data,rx,forward,parentage,data_index)
                    #grad_flow = tf.gradients(flow, (data,rx))

                    if devicename == '/CPU:0':
                        forward = np.transpose(forward,[0,2,1])
                        forward_mean = np.transpose(forward_mean,[0,2,1])
                        forward_mean_init = np.transpose(forward_mean_init,[0,2,1])
                        

                print( input_d.shape )
                for i in range(c):
                    print( (np.round(input_d[0,i,:]*10000)).astype(int) )
                print( '\n' )

                forward = np.exp(forward);
                for i in range(c):
                    print( (np.round(forward[0,i,:]*100)).astype(int) )
                print( '\n' )

                sumprob = np.sum(forward,axis=1)
                print( (np.round(sumprob[0,:]*100)).astype(int) )
                print( '\n' )
                print( '\n' )

                forward_mean = np.exp(forward_mean) / np.sum(np.exp(forward_mean),axis=1,keepdims=True)
                for i in range(c):
                    print( (np.round(forward_mean[0,i,:]*100)).astype(int) )
                print( '\n' )
                sumprob = np.sum(forward_mean,axis=1)
                print( (np.round(sumprob[0,:]*100)).astype(int) )
                print( '\n' )
                print( '\n' )

                forward_mean_init = np.exp(forward_mean_init) / np.sum(np.exp(forward_mean_init),axis=1,keepdims=True)
                for i in range(c):
                    print( (np.round(forward_mean_init[0,i,:]*100)).astype(int) )
                print( '\n' )
                sumprob = np.sum(forward_mean_init,axis=1)
                print( (np.round(sumprob[0,:]*100)).astype(int) )
                print( '\n' )
                print( '\n' )

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
