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
        r = 6
        c = 4
        x = 1
        y = 5
        z = 5

        parentage = tf.convert_to_tensor([-1, -1, 0, 0, 1, 1], dtype=tf.int32)
        data_index = tf.convert_to_tensor([-1, -1, 0, 1, 2, 3], dtype=tf.int32)
        #parentage = tf.convert_to_tensor([-1, -1, -1, -1], dtype=tf.int32)
        #data_index = tf.convert_to_tensor([0, 1, 2, 3], dtype=tf.int32)

        for i in range(1):
            mesh_y, mesh_x, mesh_z = np.meshgrid(np.arange(y),np.arange(x),np.arange(z));
            input_d = 0.0994*np.random.rand(b,c,x,y,z)
            #input_d[:,1,:,:] = 0.1*(mesh_y-y//2)+0.1*(np.random.rand(x,y)-0.5)
            input_rx = 1*np.ones((b,r,x,y,z))
            #input_rx[:,:,:,:,2] *= 0
            #input_rx[:,:,:,:,3] *= 0
            #input_rx[:,:,:,:,4] *= 0
            #input_rx[:,:,:,:,5] *= 0
            input_rx += 0.0*(np.random.rand(b,r,x,y,z))
            input_ry = 1*np.ones((b,r,x,y,z))
            #input_ry[:,:,:,:,2] *= 0
            #input_ry[:,:,:,:,3] *= 0
            #input_ry[:,:,:,:,4] *= 0
            #input_ry[:,:,:,:,5] *= 0
            input_ry += 0.0*(np.random.rand(b,r,x,y,z))
            input_rz = 1*np.ones((b,r,x,y,z))
            #input_rz[:,:,:,:,2] *= 0
            #input_rz[:,:,:,:,3] *= 0
            #input_rz[:,:,:,:,4] *= 0
            #input_rz[:,:,:,:,5] *= 0
            input_rz += 0.0*(np.random.rand(b,r,x,y,z))

            for devicename in ['GPU', 'CPU']:
                
                if devicename == 'CPU':
                    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 0})

                    data = tf.placeholder(tf.float32, shape = (b,x,y,z,c))
                    rx = tf.placeholder(tf.float32, shape = (b,x,y,z,r))
                    ry = tf.placeholder(tf.float32, shape = (b,x,y,z,r))
                    rz = tf.placeholder(tf.float32, shape = (b,x,y,z,r))

                else:
                    config = tf.ConfigProto(log_device_placement=False)

                    data = tf.placeholder(tf.float32, shape = (b,c,x,y,z))
                    rx = tf.placeholder(tf.float32, shape = (b,r,x,y,z))
                    ry = tf.placeholder(tf.float32, shape = (b,r,x,y,z))
                    rz = tf.placeholder(tf.float32, shape = (b,r,x,y,z))
                    
                
                with tf.Session(config=config) as sess:

                    flow = meanpass_module.hmf_meanpass3d(data,rx,ry,rz,parentage,data_index)
                    grad_flow = tf.gradients(flow, (data,rx,ry,rz,parentage,data_index))

                    if devicename == 'CPU':
                        prob = sess.run(tf.math.softmax(flow,axis=-1), feed_dict = {data: np.transpose(input_d,[0,2,3,4,1]), rx: np.transpose(input_rx,[0,2,3,4,1]), ry: np.transpose(input_ry,[0,2,3,4,1]), rz: np.transpose(input_rz,[0,2,3,4,1])})
                        prob = np.transpose(prob,[0,4,1,2,3])
                        forward = sess.run(flow, feed_dict = {data: np.transpose(input_d,[0,2,3,4,1]), rx: np.transpose(input_rx,[0,2,3,4,1]), ry: np.transpose(input_ry,[0,2,3,4,1]), rz: np.transpose(input_rz,[0,2,3,4,1])})
                        forward = np.transpose(forward,[0,4,1,2,3])
                        #gradient = sess.run(grad_flow, feed_dict = {data: input_d, rx: input_rx, ry: input_ry, rz: input_rz})
                    else:
                        prob = sess.run(tf.math.softmax(flow,axis=1), feed_dict = {data: input_d, rx: input_rx, ry: input_ry, rz: input_rz})
                        forward = sess.run(flow, feed_dict = {data: input_d, rx: input_rx, ry: input_ry, rz: input_rz})
                        #gradient = sess.run(grad_flow, feed_dict = {data: input_d, rx: input_rx, ry: input_ry, rz: input_rz})

                print( input_d.shape )
                print( (np.round(input_d[0,0,:,:,:]*100)).astype(int) )
                print( (np.round(input_d[0,1,:,:,:]*100)).astype(int) )
                print( (np.round(input_d[0,2,:,:,:]*100)).astype(int) )
                print( (np.round(input_d[0,3,:,:,:]*100)).astype(int) )

                print( '\n' )

                print( forward.shape )
                print( (np.round(forward[0,0,:,:,:]*100)).astype(int) )
                print( (np.round(forward[0,1,:,:,:]*100)).astype(int) )
                print( (np.round(forward[0,2,:,:,:]*100)).astype(int) )
                print( (np.round(forward[0,3,:,:,:]*100)).astype(int) )

                print( '\n' )

                print( prob.shape )
                #print( (np.round(prob[0,:,:,:,0]*100)).astype(int) )
                #print( (np.round(prob[0,:,:,:,1]*100)).astype(int) )
                #print( (np.round(prob[0,:,:,:,2]*100)).astype(int) )
                #print( (np.round(prob[0,:,:,:,3]*100)).astype(int) )
                print( (np.round((prob[0,0,:,:,:]+prob[0,1,:,:,:])*100)).astype(int) )
                print( (np.round((prob[0,2,:,:,:]+prob[0,3,:,:,:])*100)).astype(int) )
                #print( (np.round((prob[0,:,:,:,2]+prob[0,:,:,:,3])*100)).astype(int) )
                #print( gradient[0][0,:,:,:,0] )

                #np.testing.assert_array_equal(gradient_tf, gradient_inner_product)


if __name__ == '__main__':
    unittest.main()
