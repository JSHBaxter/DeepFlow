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
        x = 2
        y = 10
        z = 10

        for i in range(1):
            mesh_y, mesh_x, mesh_z = np.meshgrid(np.arange(y),np.arange(x),np.arange(z));
            input_d = 0.1*np.random.rand(b,c,x,y,z)
            #input_d[:,1,:,:] = 0.1*(mesh_y-y//2)+0.1*(np.random.rand(x,y)-0.5)
            input_rx = 0.1*np.ones((b,c,x,y,z))
            #input_rx[:,:,:,:,2] *= 0
            #input_rx[:,:,:,:,3] *= 0
            #input_rx[:,:,:,:,4] *= 0
            #input_rx[:,:,:,:,5] *= 0
            #input_rx += 0.2*(np.random.rand(b,c,x,y,z))
            input_ry = 0.1*np.ones((b,c,x,y,z))
            #input_ry[:,:,:,:,2] *= 0
            #input_ry[:,:,:,:,3] *= 0
            #input_ry[:,:,:,:,4] *= 0
            #input_ry[:,:,:,:,5] *= 0
            #input_ry += 0.2*(np.random.rand(b,c,x,y,z))
            input_rz = 0.1*np.ones((b,c,x,y,z))
            #input_rz[:,:,:,:,2] *= 0
            #input_rz[:,:,:,:,3] *= 0
            #input_rz[:,:,:,:,4] *= 0
            #input_rz[:,:,:,:,5] *= 0
            #input_rz += 0.2*(np.random.rand(b,c,x,y,z))

            for devicename in ['GPU', 'CPU']:
            #for devicename in ['GPU']:
                
                if devicename == 'CPU':
                    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 0})

                    data = tf.placeholder(tf.float32, shape = (b,x,y,z,c))
                    rx = tf.placeholder(tf.float32, shape = (b,x,y,z,c))
                    ry = tf.placeholder(tf.float32, shape = (b,x,y,z,c))
                    rz = tf.placeholder(tf.float32, shape = (b,x,y,z,c))

                else:
                    config = tf.ConfigProto(log_device_placement=False)

                    data = tf.placeholder(tf.float32, shape = (b,c,x,y,z))
                    rx = tf.placeholder(tf.float32, shape = (b,c,x,y,z))
                    ry = tf.placeholder(tf.float32, shape = (b,c,x,y,z))
                    rz = tf.placeholder(tf.float32, shape = (b,c,x,y,z))
                    
                
                with tf.Session(config=config) as sess:

                    flow = meanpass_module.potts_auglag3d(data,rx,ry,rz)
                    flow_mean = meanpass_module.potts_meanpass3d(data,rx,ry,rz)
                    #grad_flow = tf.gradients(flow, (data,rx,ry,rz))

                    if devicename == 'CPU':
                        forward = sess.run(flow, feed_dict = {data: np.transpose(input_d,[0,2,3,4,1]), rx: np.transpose(input_rx,[0,2,3,4,1]), ry: np.transpose(input_ry,[0,2,3,4,1]), rz: np.transpose(input_rz,[0,2,3,4,1])})
                        forward = np.transpose(forward,[0,4,1,2,3])
                        forward_mean = sess.run(flow_mean, feed_dict = {data: np.transpose(input_d,[0,2,3,4,1]), rx: np.transpose(input_rx,[0,2,3,4,1]), ry: np.transpose(input_ry,[0,2,3,4,1]), rz: np.transpose(input_rz,[0,2,3,4,1])})
                        forward_mean = np.transpose(forward_mean,[0,4,1,2,3])
                        #gradient = sess.run(grad_flow, feed_dict = {data: np.transpose(input_d,[0,2,3,4,1]), rx: np.transpose(input_rx,[0,2,3,4,1]), ry: np.transpose(input_ry,[0,2,3,4,1]), rz: np.transpose(input_rz,[0,2,3,4,1])})
                        #gradient = [np.transpose(g,[0,4,1,2,3]) for g in gradient]
                    else:
                        forward = sess.run(flow, feed_dict = {data: input_d, rx: input_rx, ry: input_ry, rz: input_rz})
                        forward_mean = sess.run(flow_mean, feed_dict = {data: input_d, rx: input_rx, ry: input_ry, rz: input_rz})
                        #forward0 = sess.run(flow, feed_dict = {data: input_d, rx: 0*input_rx, ry: 0*input_ry, rz: 0*input_rz})
                        #gradient = sess.run(grad_flow, feed_dict = {data: input_d, rx: input_rx, ry: input_ry, rz: input_rz})

                print( input_d.shape )
                print( (np.round(input_d[0,0,:,:,:]*100)).astype(int) )
                print( (np.round(input_d[0,1,:,:,:]*100)).astype(int) )
                print( (np.round(input_d[0,2,:,:,:]*100)).astype(int) )
                print( (np.round(input_d[0,3,:,:,:]*100)).astype(int) )

                print( '\n' )

                forward = np.exp(forward);
                print( forward.shape )
                print( (np.round(forward[0,0,:,:,:]*100)).astype(int) )
                print( (np.round(forward[0,1,:,:,:]*100)).astype(int) )
                print( (np.round(forward[0,2,:,:,:]*100)).astype(int) )
                print( (np.round(forward[0,3,:,:,:]*100)).astype(int) )
                print( '\n' )

                forward_mean = np.exp(forward_mean) / np.sum(np.exp(forward_mean),axis=1,keepdims=True);
                print( forward_mean.shape )
                print( (np.round(forward_mean[0,0,:,:,:]*100)).astype(int) )
                print( (np.round(forward_mean[0,1,:,:,:]*100)).astype(int) )
                print( (np.round(forward_mean[0,2,:,:,:]*100)).astype(int) )
                print( (np.round(forward_mean[0,3,:,:,:]*100)).astype(int) )
                print( '\n' )

                sumprob = np.sum(forward,axis=1)
                print( (np.round(sumprob[0,:,:,:]*100)).astype(int) )
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
