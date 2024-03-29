#!/usr/bin/env python3

import unittest
import numpy as np
import tensorflow as tf
import deepflow

import matplotlib.pyplot as plot

meanpass_module = deepflow.module


class InnerProductOpTest(unittest.TestCase):
    def test_runAndPrintOutput(self):
        
        b = 1
        c = 3
        x = 50
        y = 50

        for i in range(1):
            mesh_c, mesh_b, mesh_x, mesh_y = np.meshgrid(np.arange(c),np.arange(b),np.arange(x),np.arange(y));
            circle = ((mesh_x-(x-1)/2)**2+(mesh_y-(y-1)/2)**2)**0.5 / (max(x,y)/2)
            input_d = 0.5*np.ones((b,c,x,y))+0.25*np.random.randn(b,c,x,y)
            input_rx = 0.125*np.random.randn(b,c,x,y)
            input_ry = 0.125*np.random.randn(b,c,x,y)
            gold = np.zeros((b,c,x,y))
            gold[ np.logical_and(circle < (mesh_c+0.5)/3, circle > (mesh_c-0.5)/3) ] = 1.0
            input_d += 0.5*gold
            input_rx += gold
            input_ry += gold
            gold[ np.logical_and(circle > (2+0.5)/3, mesh_c == 1) ] = 1.0
            input_rx[:,:,1:,:] -= gold[:,:,:-1,:]
            input_ry[:,:,:,1:] -= gold[:,:,:,:-1]
            input_rx = 0.5*(1-abs(input_rx))
            input_ry = 0.5*(1-abs(input_ry))

            print( input_d.shape )
            print( (np.round(input_d[0,0,:,:]*100)).astype(int) )
            print( (np.round(input_d[0,1,:,:]*100)).astype(int) )
            print( (np.round(input_d[0,2,:,:]*100)).astype(int) )
            plot.imshow(np.transpose(input_d[0,:,:,:],(2,1,0)))
            plot.show()
            plot.imshow(np.transpose(input_rx[0,:,:,:],(2,1,0)))
            plot.show()
            plot.imshow(np.transpose(input_ry[0,:,:,:],(2,1,0)))
            plot.show()

            
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
                    forward = np.exp(forward);
                        
                    forward_mean = meanpass_module.potts_meanpass2d(data,rx,ry)
                    if devicename == '/CPU:0':
                        forward_mean = np.transpose(forward_mean,[0,3,1,2])
                    forward_mean = np.exp(forward_mean) / np.sum(np.exp(forward_mean),axis=1,keepdims=True)
                    
                    forward_mean_init = meanpass_module.potts_meanpass2d_with_init(data,rx,ry,forward)
                    if devicename == '/CPU:0':
                        forward_mean_init = np.transpose(forward_mean_init,[0,3,1,2])
                        forward = np.transpose(forward,[0,3,1,2])
                    forward_mean_init = np.exp(forward_mean_init) / np.sum(np.exp(forward_mean_init),axis=1,keepdims=True)
                    
                    error = forward_mean-gold
                    if devicename == '/CPU:0':
                        error = np.transpose(error,[0,2,3,1])
                        forward_mean = np.transpose(forward_mean,[0,2,3,1])
                    print(error.shape)
                    grad_d, grad_rx, grad_ry = meanpass_module.potts_meanpass2d_grad(error,data,rx,ry,forward_mean)
                    if devicename == '/CPU:0':
                        grad_d = np.transpose(grad_d,[0,3,1,2])
                        grad_rx = np.transpose(grad_rx,[0,3,1,2])
                        grad_ry = np.transpose(grad_ry,[0,3,1,2])

                print( '\n' )
                
                forward_sf = np.exp(input_d) / np.sum(np.exp(input_d),axis=1,keepdims=True)
                print( (np.round(forward_sf[0,0,:,:]*100)).astype(int) )
                print( (np.round(forward_sf[0,1,:,:]*100)).astype(int) )
                print( (np.round(forward_sf[0,2,:,:]*100)).astype(int) )
                plot.imshow(np.transpose(forward_sf[0,:,:,:],(2,1,0)))
                plot.show()

                print( '\n' )

                print( forward.shape )
                print( (np.round(forward[0,0,:,:]*100)).astype(int) )
                print( (np.round(forward[0,1,:,:]*100)).astype(int) )
                print( (np.round(forward[0,2,:,:]*100)).astype(int) )
                print( '\n' )
                sumprob = np.sum(forward,axis=1)
                print( (np.round(sumprob[0,:,:]*100)).astype(int) )
                print( '\n' )
                print( '\n' )
                plot.imshow(np.transpose(forward[0,:,:,:],(2,1,0)))
                plot.show()

                print( forward_mean.shape )
                print( (np.round(forward_mean[0,0,:,:]*100)).astype(int) )
                print( (np.round(forward_mean[0,1,:,:]*100)).astype(int) )
                print( (np.round(forward_mean[0,2,:,:]*100)).astype(int) )
                print( '\n' )
                print( '\n' )
                plot.imshow(np.transpose(forward_mean[0,:,:,:],(2,1,0)))
                plot.show()

                print( forward_mean_init.shape )
                print( (np.round(forward_mean_init[0,0,:,:]*100)).astype(int) )
                print( (np.round(forward_mean_init[0,1,:,:]*100)).astype(int) )
                print( (np.round(forward_mean_init[0,2,:,:]*100)).astype(int) )
                print( '\n' )
                print( '\n' )
                plot.imshow(np.transpose(forward_mean_init[0,:,:,:],(2,1,0)))
                plot.show()

                print( grad_d.shape )
                print( (np.round(grad_d[0,0,:,:]*100)).astype(int) )
                print( (np.round(grad_d[0,1,:,:]*100)).astype(int) )
                print( (np.round(grad_d[0,2,:,:]*100)).astype(int) )
                print( '\n' )
                print( '\n' )
                plot.imshow(np.transpose(grad_d[0,:,:,:],(2,1,0)))
                plot.show()
                
                plot.imshow(np.transpose(0.5+0.5*(forward_mean_init-forward_sf)[0,:,:,:],(2,1,0)))
                plot.show()
                plot.imshow(np.transpose(0.5+0.5*(forward_mean_init-forward_mean)[0,:,:,:],(2,1,0)))
                plot.show()

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
