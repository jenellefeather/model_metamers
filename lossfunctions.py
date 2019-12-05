"""
Contains the loss functions that are used for generating metamers. 
In the refenced NeurIPS paper, the 'raw_pixel' L2 loss was used for all 
experiments and the TV regularization was not included in the generation
proceedure. 

This code was used for the following paper:
  Metamers of neural networks reveal divergence from human perceptual systems
  Jenelle Feather, Alex Durango, Ray Gonzalez, Josh McDermott
  In Advances in Neural Information Processing Systems (2019)
  PDF: https://papers.nips.cc/paper/9198-metamers-of-neural-networks-reveal-divergence-from-human-perceptual-systems.pdf
"""

import functools
import tensorflow as tf
import numpy as np

def generate_loss_functions(LOSS_TYPE='raw_pixels', SHAPE_NORMALIZE=True):
    """
    Wrapper to create loss functions and functions to measure statistics for plotting. 
    
    Returns
    -------
    loss_function : func
        creates a list of tensorflow losses to use for the matched statistics
        Parameters
        ----------
        layer : tensorflow tensor
           nets['<layer>'] that will be used to compute the loss. Should be [1 height, width, number]
        comparison_features : numpy array
            the output from nets['<layer>'].eval(feed_dict = {<input_feed>}) to use for the loss function
        update_losses : list
            a list of the losses computed so far, or an empty list
        
        Returns
        -------
        update_losses : list
            a list of the losses computed so far        
    
        measure_stats_function : func
        takes a numpy array for a layer and measures the statistics for the layer, will always be unraveled for plotting purposes. 
        Parameters
        ----------
        comparison_features : numpy array
            the output from nets['<layer>'].eval(feed_dict = {<input_feed>}) over which you will compute the specified statistics
        
        Returns
        -------
        measured_statistics : numpy array
            the statistics measued for the layer with the specified function
        
        
    """
    if LOSS_TYPE=='raw_pixels':
        loss_function = functools.partial(match_pixel_loss, SHAPE_NORMALIZE=SHAPE_NORMALIZE)
        measure_stats_function = match_pixel_measure
        return loss_function, measure_stats_function
    elif LOSS_TYPE=='raw_pixels_l1':
        loss_function = match_pixel_loss_l1
        measure_stats_function = match_pixel_measure
        return loss_function, measure_stats_function
    elif LOSS_TYPE=='total_variation':
        loss_function = total_variation_loss_function
        measure_stats_function = measure_total_variation_loss
        return loss_function, measure_stats_function


def total_variation_loss_function(layer, comparison_features, update_losses=[], layer_weight=1.):
    """
    Returns the total variation loss for layer. Useful for regularization. 

    Parameters
    ----------
    layer : tensorflow tensor
        nets['<layer>'] that will be used to compute the correlation matrix. Should be [1 height, width, 1]
    comparison_features : numpy array (not used) 
        the output from nets['<layer>'].eval(feed_dict = {<input_feed>}) over which you will compute the correlation 
        matrix, dimensions are [1 <height> <width> 1] or [<height> <width> 1]
    update_losses : list
         a list of the losses computed so far, or an empty list
    layer_weight : int or float
        amount to weight the layer loss

    Returns
    -------
    update_losses : list
        a list with the losses

    """
    del comparison_features
    update_losses.append(tf.reduce_sum(layer_weight * tf.image.total_variation(layer)))
    return update_losses
    

def measure_total_variation_loss(layer):
    # numpy port of https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/image_ops_impl.py#L2320-L2388
    ndims = len(layer.shape)

    if ndims == 3:
      # The input is a single image with shape [height, width, channels].

      # Calculate the difference of neighboring pixel-values.
      # The images are shifted one pixel along the height and width by slicing.
      pixel_dif1 = layer[1:, :, :] - layer[:-1, :, :]
      pixel_dif2 = layer[:, 1:, :] - layer[:, :-1, :]

      # Sum for all axis. (None is an alias for all axis.)
      sum_axis = None
    elif ndims == 4:
      # The input is a batch of images with shape:
      # [batch, height, width, channels].

      # Calculate the difference of neighboring pixel-values.
      # The images are shifted one pixel along the height and width by slicing.
      pixel_dif1 = layer[:, 1:, :, :] - layer[:, :-1, :, :]
      pixel_dif2 = layer[:, :, 1:, :] - layer[:, :, :-1, :]

      # Only sum for the last 3 axis.
      # This results in a 1-D tensor with the total variation for each image.
      sum_axis = (1, 2, 3)
    else:
      raise ValueError('\'images\' must be either 3 or 4-dimensional.')

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = (
        np.sum(np.abs(pixel_dif1), axis=sum_axis) +
        np.sum(np.abs(pixel_dif2), axis=sum_axis))

    return tot_var
   

def match_pixel_measure(comparison_features):
    """
    Stats function when measuring pixels -- just unravel the comparison features. 
    
    Parameters
    ----------
    comparison_features : numpy array
        the output from nets['<layer>'].eval(feed_dict = {<input_feed>}) to use for the statistics
    
    Returns
    -------
    update_losses : list
        a list of the losses computed so far   
    
    """
    measure_stats=np.ravel(comparison_features)
    return measure_stats


def match_pixel_loss(layer, comparison_features, update_losses=[], SHAPE_NORMALIZE=False):
    """
    Matches on the raw pixel values, used for metamer generation.
    
    Parameters
    ----------
    layer : tensorflow tensor
        nets['<layer>'] that will be used to compute the loss. Should be [1 height, width, number]
    comparison_features : numpy array
        the output from nets['<layer>'].eval(feed_dict = {<input_feed>}) to use for the loss function
    update_losses : list
        a list of the losses computed so far, or an empty list
    shape_normalize : boolean
       
    
    Returns
    -------
    update_losses : list
        a list of the losses computed so far   
        
    """
    if len(layer.get_shape()) == 4:
        batch, height, width, channels = layer.get_shape()
        channels = channels.value
        height = height.value
        width = width.value
        batch = batch.value
        shape_norm_value = batch*channels*height*width
    elif len(layer.get_shape()) == 3:
        batch, width, channels = layer.get_shape()
        channels = channels.value
        width = width.value
        batch = batch.value
        shape_norm_value = batch * width * channels
    else:
        batch, number = layer.get_shape()
        shape_norm_value = batch.value * number.value
        

    if SHAPE_NORMALIZE: # use the full size of the layer for the normalization
        normalization = shape_norm_value  # check this. its not quite right.
    else:
        normalization = 1

    update_losses.append(tf.nn.l2_loss(comparison_features-layer)/normalization)
    return update_losses

def match_pixel_loss_l1(layer, comparison_features, update_losses=[]):
    """
    Matches on the raw pixel values with an l1 loss. 

    Parameters
    ----------
    layer : tensorflow tensor
        nets['<layer>'] that will be used to compute the loss. Should be [1 height, width, number]
    comparison_features : numpy array
        the output from nets['<layer>'].eval(feed_dict = {<input_feed>}) to use for the loss function
    update_losses : list
        a list of the losses computed so far, or an empty list    Returns

    Returns
    -------
    update_losses : list
        a list of the losses computed so far

    """
    update_losses.append(tf.reduce_sum(tf.abs(comparison_features-layer)))
    return update_losses

