#!/usr/bin/env python3

import unittest
import numpy as np
import tensorflow as tf
import deepflow

meanpass_module = deepflow.module

class InnerProductOpTest(unittest.TestCase):
    def test_runAndPrintOutput(self):
        
        b = 1
        c = 4
        x = 10
        y = 10

        for i in range(1):
            mesh_y, mesh_x = np.meshgrid(np.arange(y),np.arange(x));
            input_d = 0.01*np.random.rand(b,c,x,y)
            input_rx = 1.0*np.ones((b,c,x,y))
            input_ry = 1.0*np.ones((b,c,x,y))
            
            for devicename in ['/CPU:0','/GPU:0']:
                
                if devicename == '/CPU:0':
                    data = tf.convert_to_tensor(np.transpose(input_d,[0,2,3,1]),dtype=tf.float32)
                    rx = tf.convert_to_tensor(np.transpose(input_rx,[0,2,3,1]),dtype=tf.float32)
                    ry = tf.convert_to_tensor(np.transpose(input_ry,[0,2,3,1]),dtype=tf.float32)

                else:
                    data = tf.convert_to_tensor(input_d,dtype=tf.float32)
                    rx = tf.convert_to_tensor(input_rx,dtype=tf.float32)
                    ry = tf.convert_to_tensor(input_ry,dtype=tf.float32)
                    
                
                with tf.device(devicename):

                    forward = meanpass_module.potts_auglag2d(data,rx,ry)
                    forward_mean = meanpass_module.potts_meanpass2d(data,rx,ry)
                    #grad_flow = tf.gradients(flow, (data,rx))

                    if devicename == '/CPU:0':
                        forward = np.transpose(forward,[0,3,1,2])
                        forward_mean = np.transpose(forward_mean,[0,3,1,2])

                print( input_d.shape )
                print( (np.round(input_d[0,0,:,:]*100)).astype(int) )
                print( (np.round(input_d[0,1,:,:]*100)).astype(int) )
                print( (np.round(input_d[0,2,:,:]*100)).astype(int) )
                print( (np.round(input_d[0,3,:,:]*100)).astype(int) )

                print( '\n' )

                forward = np.exp(forward);
                print( forward.shape )
                print( (np.round(forward[0,0,:,:]*100)).astype(int) )
                print( (np.round(forward[0,1,:,:]*100)).astype(int) )
                print( (np.round(forward[0,2,:,:]*100)).astype(int) )
                print( (np.round(forward[0,3,:,:]*100)).astype(int) )
                print( '\n' )
                print( '\n' )

                forward_mean = np.exp(forward_mean) / np.sum(np.exp(forward_mean),axis=1,keepdims=True)
                print( forward_mean.shape )
                print( (np.round(forward_mean[0,0,:,:]*100)).astype(int) )
                print( (np.round(forward_mean[0,1,:,:]*100)).astype(int) )
                print( (np.round(forward_mean[0,2,:,:]*100)).astype(int) )
                print( (np.round(forward_mean[0,3,:,:]*100)).astype(int) )
                print( '\n' )
                print( '\n' )

                sumprob = np.sum(forward,axis=1)
                print( (np.round(sumprob[0,:,:]*100)).astype(int) )

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
