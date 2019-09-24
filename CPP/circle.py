#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.
"""

import numpy as np
import tensorflow as tf
import deepflow
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filt

flow_module = deepflow.module

size = 16
snr = 0.25

gx, gy = np.meshgrid(range(size),range(size))
circle = ((gx-size/2)**2 +(gy-size/2)**2 < (size/4)**2).astype('float32')
circle_blur = filt.uniform_filter(circle,3)

input_data = 0.5*((circle_blur-0.5) + np.random.normal(size=circle.shape) / snr)
input_data = filt.uniform_filter(input_data,11)
input_data = np.expand_dims(np.expand_dims(input_data,0),0)

input_ry = np.copy(circle_blur)
input_ry[:,:-1] -= circle_blur[:,1:]
input_ry = (1-3*abs(input_ry))
input_ry = np.expand_dims(np.expand_dims(input_ry,0),0)

input_rx = np.copy(circle_blur)
input_rx[:-1,:] -= circle_blur[1:,:]
input_rx = (1-3*abs(input_rx))
input_rx = np.expand_dims(np.expand_dims(input_rx,0),0)

circle = np.expand_dims(np.expand_dims(circle,0),0)

config = tf.ConfigProto(log_device_placement=False)
#config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 0})
with tf.Session(config=config) as sess:
	data = tf.placeholder(tf.float32, shape = (1,1,size,size))
	rx = tf.placeholder(tf.float32, shape = (1,1,size,size))
	ry = tf.placeholder(tf.float32, shape = (1,1,size,size))
	grad = tf.placeholder(tf.float32, shape = (1,1,size,size))
	u = tf.placeholder(tf.float32, shape = (1,1,size,size))
	
	def sigmoid(a):
		return 1.0 / (1.0 + np.exp(-a))
		
	flow_mp = flow_module.binary_meanpass2d(data,rx,ry)
	forward_mp = sess.run(flow_mp, feed_dict = {data: input_data, rx: input_rx, ry: input_ry})
	flow_g = flow_module.binary_meanpass2d_grad(grad,data,rx,ry,u)
	loss_grad = (sigmoid(forward_mp)-circle)*sigmoid(forward_mp)*(1-sigmoid(forward_mp))
	backward_mp_d, backward_mp_rx, backward_mp_ry = sess.run(flow_g, feed_dict = {grad: loss_grad, data: input_data, rx: input_rx, ry: input_ry, u: forward_mp})
	forward_mp = sigmoid(forward_mp)
	flow_al = tf.exp(flow_module.binary_auglag2d(data,rx,ry))
	forward_al = sess.run(flow_al, feed_dict = {data: input_data, rx: input_rx, ry: input_ry})
	
	plt.imshow(circle.squeeze(),vmin=0,vmax=1)
	plt.axis('off')
	plt.savefig("circle_gold.png",bbox_inches='tight',pad_inches=0)
	plt.show()
	plt.imshow(input_data.squeeze(),vmin=-0.25,vmax=0.25)
	plt.axis('off')
	plt.savefig("circle_data_term.png",bbox_inches='tight',pad_inches=0)
	plt.show()
	plt.imshow(input_rx.squeeze(),vmin=0,vmax=0.5)
	plt.axis('off')
	plt.savefig("circle_rx_term.png",bbox_inches='tight',pad_inches=0)
	plt.show()
	plt.imshow(input_ry.squeeze(),vmin=0,vmax=0.5)
	plt.axis('off')
	plt.savefig("circle_ry_term.png",bbox_inches='tight',pad_inches=0)
	plt.show()
	plt.imshow(forward_mp.squeeze(),vmin=0,vmax=1)
	plt.axis('off')
	plt.savefig("circle_mean_pass.png",bbox_inches='tight',pad_inches=0)
	plt.show()
	plt.imshow(forward_al.squeeze(),vmin=0,vmax=1)
	plt.axis('off')
	plt.savefig("circle_auglag.png",bbox_inches='tight',pad_inches=0)
	plt.show()
	
	plt.imshow(backward_mp_d.squeeze(),vmin=-0.05,vmax=0.05)
	plt.axis('off')
	plt.savefig("circle_data_grad.png",bbox_inches='tight',pad_inches=0)
	plt.show()
	plt.imshow(backward_mp_rx.squeeze(),vmin=-0.05,vmax=0.05)
	plt.axis('off')
	plt.savefig("circle_rx_grad.png",bbox_inches='tight',pad_inches=0)
	plt.show()
	plt.imshow(backward_mp_ry.squeeze(),vmin=-0.05,vmax=0.05)
	plt.axis('off')
	plt.savefig("circle_ry_grad.png",bbox_inches='tight',pad_inches=0)
	plt.show()
	
	plt.hist(np.concatenate((backward_mp_rx.flatten(),backward_mp_ry.flatten())),bins=100,log=True)
	plt.savefig("circle_r_grad_hist.png",bbox_inches='tight',pad_inches=0)
	plt.show()