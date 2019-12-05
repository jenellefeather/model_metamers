"""
Contains functions used to build the cnns for audio networks. 

This code was used for the following paper:
  Metamers of neural networks reveal divergence from human perceptual systems
  Jenelle Feather, Alex Durango, Ray Gonzalez, Josh McDermott
  In Advances in Neural Information Processing Systems (2019)
  PDF: https://papers.nips.cc/paper/9198-metamers-of-neural-networks-reveal-divergence-from-human-perceptual-systems.pdf
"""

import tensorflow as tf
import functools
import pickle
import numpy as np
import os
import json


def multi_fc_top_classification(input_tensor, n_classes_dict, name, **kwargs):
    """
    Builds multiple FC layers (and appends integer names) for each of the tasks specified in n_classes_dict.  
    
    Args 
    ----
    input_tensor (tensorflow tensor) : the input layer for each of the added fc layers
    n_classes_dict (dict) : contains the number of classes (number of FC units) for each of the tasks 
    name (string) : name of the fc_layer, function appends integers to name for each task

    Outputs
    -------
    output_layer_dict (dictionary) : dictionary containing each of the output fc layers

    """
    output_layer_dict = {}
    all_keys_tasks = list(n_classes_dict.keys())
    all_keys_tasks.sort() # so that when we reload things are in the same order
    for num_classes_task_idx, num_classes_task_name in enumerate(all_keys_tasks):
        task_name = '%s_%s'%(name, num_classes_task_idx)
        output_layer_dict[num_classes_task_name] = tf.layers.dense(input_tensor, units=n_classes_dict[num_classes_task_name], name=task_name, **kwargs)

    return output_layer_dict


def hanning_pooling(input_layer, strides=2, pool_size=8, padding='SAME', name=None, sqrt_window=False, normalize=False):
    """
    Add a layer using a hanning kernel for pooling

    Parameters
    ----------
    input_layer : tensorflow tensor
        layer to add hanning pooling to
    strides : int
        proportion downsampling
    top_node : string
        specify the node after which the spectemp filters will be added and used as input for the FFT.
    sqrt_window : boolean
        if true, takes the sqrt of the window (old version), normal window generation has sqrt_window=False
    normalize : boolean
        if true, divide the filter by the sum of its values, so that the smoothed signal is the same amplitude as the original.
    name : False or string
        name for the layer. If false appends "_hpool" to the top_node name

    Returns
    -------
    output_layer : tensorflow tensor
        input_layer with hanning pooling applied
    """
    n_channels = input_layer.get_shape().as_list()[3]
    hanning_window_tensor = make_hanning_kernel_tensor_no_depthwise(n_channels, strides=strides, pool_size=pool_size, sqrt_window=sqrt_window, normalize=normalize, name='%s_hpool_kernel'%name)
    if type(strides)!=list and type(strides)==int:
        strides = [strides, strides] # using square filters
    output_layer = tf.nn.conv2d(input_layer, filter=hanning_window_tensor, strides=[1, strides[0], strides[1], 1], padding=padding, name=name)
    return output_layer


def make_hanning_kernel_tensor_no_depthwise(n_channels, strides=2, pool_size=8, sqrt_window=False, normalize=False, name=None):
    """
    Make a tensor containing the symmetric 2d hanning kernel to use for the pooling filters
    For strides=2, using pool_size=8 gives a reduction of -24.131545969216841 at 0.25 cycles
    For strides=3, using pool_size=12 gives a reduction of -28.607805482176282 at 1/6 cycles

    This version uses the normal conv2d operation and fills most of the smoothing tensor with zeros. Depthwise convolution
    does not have a second order gradient, and cannot be used with some functions.

    Parameters
    ----------
    n_channels : int
        number of channels to copy the kernel into
    strides : int
        proportion downsampling
    pool_size : int
        how large of a window to use
    sqrt_window : boolean
        if true, takes the sqrt of the window (old version), normal window generation has sqrt_window=False
    normalize : boolean
        if true, divide the filter by the sum of its values, so that the smoothed signal is the same amplitude as the original.
    name : False or string
        name for the layer. If false appends "_hpool" to the top_node name


    Returns
    -------
    hanning_tensor : tensorflow tensor
        tensorflow tensor containing the hanning kernel with size [1 pool_size pool_size n_channels]

    """
    hanning_kernel = make_hanning_kernel(strides=strides,pool_size=pool_size,sqrt_window=sqrt_window, normalize=normalize).astype(np.float32)
    hanning_kernel = np.expand_dims(np.expand_dims(hanning_kernel,0),0) * np.expand_dims(np.expand_dims(np.eye(n_channels),3),3) # [width, width, n_channels, n_channels]
    hanning_tensor = tf.constant(hanning_kernel, dtype=tf.float32, name=name)
    hanning_tensor = tf.transpose(hanning_tensor, [2,3,0,1])
    return hanning_tensor


def make_hanning_kernel(strides=2, pool_size=8, sqrt_window=False, normalize=False):
    """
    Make the symmetric 2d hanning kernel to use for the pooling filters
    For strides=2, using pool_size=8 gives a reduction of -24.131545969216841 at 0.25 cycles
    For strides=3, using pool_size=12 gives a reduction of -28.607805482176282 at 1/6 cycles

    Parameters
    ----------
    strides : int
        proportion downsampling
    pool_size : int
        how large of a window to use
    sqrt_window : boolean
        if true, takes the sqrt of the window (old version), normal window generation has sqrt_window=False
    normalize : boolean
        if true, divide the filter by the sum of its values, so that the smoothed signal is the same amplitude as the original.

    Returns
    -------
    two_dimensional_kernel : numpy array
        hanning kernel in 2d to use as a kernel for filtering

    """

    if type(strides)!=list and type(strides)==int:
        strides = [strides, strides] # using square filters
 
    if type(pool_size)!=list and type(pool_size)==int: 
        if pool_size > 1:
            window = 0.5 * (1 - np.cos(2.0 * np.pi * (np.arange(pool_size)) / (pool_size - 1)))
            if sqrt_window: 
                two_dimensional_kernel = np.sqrt(np.outer(window, window))
            else: 
                two_dimensional_kernel = np.outer(window, window)
        else: 
            window = np.ones((1,1))
            two_dimensional_kernel = window # [1x1 kernel]
    elif type(pool_size)==list:
        if pool_size[0] > 1:
            window_h = np.expand_dims(0.5 * (1 - np.cos(2.0 * np.pi * (np.arange(pool_size[0])) / (pool_size[0] - 1))),0)
        else:
            window_h = np.ones((1,1))
        if pool_size[1] > 1:
            window_w = np.expand_dims(0.5 * (1 - np.cos(2.0 * np.pi * (np.arange(pool_size[1])) / (pool_size[1] - 1))),1)
        else:
            window_w = np.ones((1,1))
 
        if sqrt_window:
            two_dimensional_kernel = np.sqrt(np.outer(window_h, window_w))
        else:  
            two_dimensional_kernel = np.outer(window_h, window_w)

    if normalize:
        two_dimensional_kernel = two_dimensional_kernel/(sum(two_dimensional_kernel.ravel()))        
    
    return two_dimensional_kernel


