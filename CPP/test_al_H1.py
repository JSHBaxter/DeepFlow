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
        c = 8
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
            input_rx = 0*np.ones((b,r,x))
            input_rx[:,0:c,:] = 10
            print(input_rx)
            #input_rx[:,:,:,:,2] *= 0
            #input_rx[:,:,:,:,3] *= 0
            #input_rx[:,:,:,:,4] *= 0
            #input_rx[:,:,:,:,5] *= 0
            #input_rx += 0.2*(np.random.rand(b,c,x))

            for devicename in ['CPU','GPU']:
                
                if devicename == 'CPU':
                    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 0})

                    data = tf.placeholder(tf.float32, shape = (b,x,c))
                    rx = tf.placeholder(tf.float32, shape = (b,x,r))

                else:
                    config = tf.ConfigProto(log_device_placement=False)

                    data = tf.placeholder(tf.float32, shape = (b,c,x))
                    rx = tf.placeholder(tf.float32, shape = (b,r,x))
                    
                
                with tf.Session(config=config) as sess:

                    flow = meanpass_module.hmf_auglag1d(data,rx,parentage,data_index)
                    flow_mean = meanpass_module.hmf_meanpass1d(data,rx,parentage,data_index)
                    #grad_flow = tf.gradients(flow, (data,rx))

                    if devicename == 'CPU':
                        forward = sess.run(flow, feed_dict = {data: np.transpose(input_d,[0,2,1]), rx: np.transpose(input_rx,[0,2,1])})
                        forward = np.transpose(forward,[0,2,1])
                        forward_mean = sess.run(flow_mean, feed_dict = {data: np.transpose(input_d,[0,2,1]), rx: np.transpose(input_rx,[0,2,1])})
                        forward_mean = np.transpose(forward_mean,[0,2,1])
                        #gradient = sess.run(grad_flow, feed_dict = {data: np.transpose(input_d,[0,2,3,4,1]), rx: np.transpose(input_rx,[0,2,3,4,1]), ry: np.transpose(input_ry,[0,2,3,4,1]), rz: np.transpose(input_rz,[0,2,3,4,1])})
                        #gradient = [np.transpose(g,[0,4,1,2,3]) for g in gradient]
                    else:
                        forward = sess.run(flow, feed_dict = {data: input_d, rx: input_rx})
                        forward_mean = sess.run(flow_mean, feed_dict = {data: input_d, rx: input_rx})
                        #forward0 = sess.run(flow, feed_dict = {data: input_d, rx: 0*input_rx, ry: 0*input_ry, rz: 0*input_rz})
                        #gradient = sess.run(grad_flow, feed_dict = {data: input_d, rx: input_rx, ry: input_ry, rz: input_rz})

                print( input_d.shape )
                for i in range(c):
                    print( (np.round(input_d[0,i,:]*10000)).astype(int) )

                print( '\n' )

                #forward = np.exp(forward);
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