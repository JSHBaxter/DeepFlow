#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.
"""

import numpy as np
import tensorflow as tf
import deepflow
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filt
import time

flow_module = deepflow.module

size = 256
snr = 0.5

filtsize = 3
gx, gy = np.meshgrid(range(size),range(size))
circle1 = ((gx-size/2)**2 +(gy-size/2)**2 < (size/3)**2).astype('float32')
circle1_blur = filt.uniform_filter(circle1,filtsize)
circle2 = ((gx-size/2)**2 +(gy-size/2)**2 < (size/4)**2).astype('float32')
circle2_blur = filt.uniform_filter(circle2,filtsize)

ground_truth = np.zeros((3,size,size),'float32')
ground_truth[0,:,:] = 1-circle1
ground_truth[1,:,:] = circle1-circle2
ground_truth[2,:,:] = circle2

input_data = np.copy(ground_truth-0.5)
input_data = 0.25*(input_data + np.random.normal(size=input_data.shape) / snr)
input_data[0,:,:] = filt.uniform_filter(input_data[0,:,:],filtsize*2+1)
input_data[1,:,:] = filt.uniform_filter(input_data[1,:,:],filtsize*2+1)
input_data[2,:,:] = filt.uniform_filter(input_data[2,:,:],filtsize*2+1)
input_data = np.expand_dims(input_data,0)

input_ry = np.zeros((4,size,size),'float32')
input_ry[0,:,:] = circle1_blur
input_ry[0,:,:-1] -= circle1_blur[:,1:]
input_ry[1,:,:] = circle1_blur
input_ry[1,:,:-1] -= circle1_blur[:,1:]
input_ry[2,:,:] = circle1_blur-circle2_blur
input_ry[2,:,:-1] -= (circle1_blur-circle2_blur)[:,1:]
input_ry[3,:,:] = circle2_blur
input_ry[3,:,:-1] -= circle2_blur[:,1:]
input_ry = 0.5*(1-filtsize*abs(input_ry))+0.00
input_ry[3,:,:] = 0
input_ry = np.expand_dims(input_ry,0)

input_rx = np.zeros((4,size,size),'float32')
input_rx[0,:,:] = circle1_blur
input_rx[0,:-1,:] -= circle1_blur[1:,:]
input_rx[1,:,:] = circle1_blur
input_rx[1,:-1,:] -= circle1_blur[1:,:]
input_rx[2,:,:] = circle1_blur-circle2_blur
input_rx[2,:-1,:] -= (circle1_blur-circle2_blur)[1:,:]
input_rx[3,:,:] = circle2_blur
input_rx[3,:-1,:] -= circle2_blur[1:,:]
input_rx = 0.5*(1-filtsize*abs(input_rx))+0.00
input_rx[3,:,:] = 0
input_rx = np.expand_dims(input_rx,0)

config = tf.ConfigProto(log_device_placement=False)
#config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 0})
with tf.Session(config=config) as sess:

	parentage = tf.convert_to_tensor([-1, -1, 0, 0], dtype=tf.int32)
	data_index = tf.convert_to_tensor([-1, 0, 1, 2], dtype=tf.int32)

	data = tf.placeholder(tf.float32, shape = (1,3,size,size))
	rx = tf.placeholder(tf.float32, shape = (1,4,size,size))
	ry = tf.placeholder(tf.float32, shape = (1,4,size,size))
	grad = tf.placeholder(tf.float32, shape = (1,3,size,size))
	u = tf.placeholder(tf.float32, shape = (1,3,size,size))
	
	def sigmoid(a):
		return 1.0 / (1.0 + np.exp(-a))
		
	def softmax(a):
		if a.shape[0] > 1:
			return np.exp(a-np.max(a)) / np.sum(np.exp(a-np.max(a)), axis=0, keepdims=True)	
		return np.exp(a-np.max(a)) / np.sum(np.exp(a-np.max(a)), axis=1, keepdims=True)	
		
	def grad_softmax(g,a):
		s = softmax(a)
		ret_g = np.zeros(g.shape)
		for i in range(s.shape[1]):
			for j in range(s.shape[1]):
				if j == i:
					ret_g[:,i,:,:] += g[:,i,:,:]*s[:,i,:,:]*(1-s[:,i,:,:])
				else:
					ret_g[:,i,:,:] -= g[:,j,:,:]*s[:,i,:,:]*s[:,j,:,:]
		return ret_g
		
	plt.imshow(np.transpose(input_data[0,:,:,:]+0.5,(1,2,0)))
	plt.show()
	
	flow_al = tf.exp(flow_module.hmf_auglag2d(data,rx,ry,parentage,data_index))
	print("Running AL...")
	t = time.time()
	forward_al = sess.run(flow_al, feed_dict = {data: input_data, rx: input_rx, ry: input_ry})
	elapsed = time.time() - t
	plt.imshow(np.transpose(forward_al[0,:,:,:],(1,2,0)))
	plt.show()
	print("\t ... done " + str(elapsed))
		
	flow_mp = flow_module.hmf_meanpass2d(data,rx,ry,parentage,data_index)
	print("Running MP...")
	t = time.time()
	forward_mp = sess.run(flow_mp, feed_dict = {data: input_data, rx: input_rx, ry: input_ry})
	elapsed = time.time() - t
	plt.imshow(np.transpose(softmax(forward_mp)[0,:,:,:],(1,2,0)))
	plt.show()
	print("\t ... done " + str(elapsed))
	
	flow_g = flow_module.hmf_meanpass2d_grad(grad,data,rx,ry,parentage,data_index,u)
	loss_grad = grad_softmax(softmax(forward_mp)-ground_truth, forward_mp)
	plt.imshow(loss_grad[0,0,:,:])
	plt.show()
	
	print("Running MP grad...")
	t = time.time()
	backward_mp_d, backward_mp_rx, backward_mp_ry, _, _ = sess.run(flow_g, feed_dict = {grad: loss_grad, data: input_data, rx: input_rx, ry: input_ry, u: forward_mp})
	elapsed = time.time() - t
	print("\t ... done " + str(elapsed))
		
	plt.imshow(backward_mp_d[0,0,:,:])
	plt.show()
	plt.imshow(backward_mp_rx[0,0,:,:])
	plt.show()
	plt.imshow(backward_mp_rx[0,1,:,:])
	plt.show()
	plt.imshow(backward_mp_rx[0,2,:,:])
	plt.show()
	plt.imshow(backward_mp_rx[0,3,:,:])
	plt.show()
	
	