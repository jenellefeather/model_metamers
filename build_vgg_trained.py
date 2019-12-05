"""
Builds a VGG-19 network from the tfslim repo, including preprocessing steps, 
and sets up a dictionary with pointers to the activations to use when making
loss functions for model metamer generation. 

This code was used for the following paper:
  Metamers of neural networks reveal divergence from human perceptual systems
  Jenelle Feather, Alex Durango, Ray Gonzalez, Josh McDermott
  In Advances in Neural Information Processing Systems (2019)
  PDF: https://papers.nips.cc/paper/9198-metamers-of-neural-networks-reveal-divergence-from-human-perceptual-systems.pdf
"""

import sys
import tensorflow as tf
import numpy as np
from imagenet_classes import class_names
import vgg
import metamer_helpers

@tf.custom_gradient
def jittered_relu_grad(x):
    y = tf.nn.relu(x)
    def grad(dy): #clip the zeros.
        dy_shape = dy.get_shape()
        return tf.where(x<=0, dy, dy)  # tf.clip_by_value(dy, -0.1, 0.1)
    return y, grad


# Copied from the vgg preprocessing library in tfslim, modified to work with [BATCH,X,X,X] tensors
def mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.
  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)
  Note that the rank of `image` must be known.
  Args:
    image: a tensor of size [height, width, C] or [batch, height, width, C].
    means: a C-vector of values to subtract from each channel.
  Returns:
    the centered image.
  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  if image.get_shape().ndims == 3:
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
      channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)
  if image.get_shape().ndims == 4:
    return tf.subtract(image, np.expand_dims(np.expand_dims(np.expand_dims(means,0),0),0))

  else: 
    raise ValueError('Input must be of size [height, width, C>0]')


def add_jitter_relu_to_layer_vgg(nets, layer_name):
  """
  Specify the layer that we are adding a jittered relu to, and contains the logic for 
  adding the relu. This is custom built for each network.
  """
  input_layers = {'conv1_2':'conv1_2_prerelu',
                  'conv2_2':'conv2_2_prerelu',
                  'conv3_4':'conv3_4_prerelu',
                  'conv4_4':'conv4_4_prerelu',
                  'conv5_4':'conv5_4_prerelu',
                  'fc6':'fc6_prerelu',
                  'fc7':'fc7_prerelu'}

  if 'conv' in layer_name:
    matched_layer_name = '%s_jittered_relu'%layer_name
    nets[matched_layer_name] = jittered_relu_grad(nets[input_layers[layer_name]])
  if 'fc' in layer_name:
    matched_layer_name = '%s_jittered_relu'%layer_name
    nets[matched_layer_name] = jittered_relu_grad(nets[input_layers[layer_name]])
  elif 'Mixed' in layer_name: 
    branches = []
    print(input_layers[layer_name])
    for branch in input_layers[layer_name]:
      print(branch)
      print(nets[branch])
      if type(nets[branch]) is list:
        intermediate_concat = []
        for intermediate in nets[branch]:
          intermediate_concat.append(jittered_relu_grad(intermediate))
          branches.append(tf.concat(axis=3, values=intermediate_concat))
      else:
        print(nets[branch])
        branches.append(jittered_relu_grad(nets[branch]))
    print(branches)
    matched_layer_name = '%s_jittered_relu'%layer_name
    nets[matched_layer_name] = tf.concat(axis=3, values=branches)
  else: # don't use the relu
    matched_layer_name = layer_name
  return nets, matched_layer_name


def main():

  session = tf.Session()


  ### This section deals with preprocessing for the VGG network ###

  # Used for variable clipping. For imagenet metamers we always optimize a varible input 
  # bounded between 0-1 and then rescale before going into the network.
  min_image = 0 
  max_image = 1 

  # Parameters for preprocessing VGG, the input images are between 0-255
  subtract_value = 0
  multiply_value = 255

  # The mean channel values used for VGG preprocessing
  means = [123.68, 116.779, 103.939]

  # Make an input variable for the network (will be optimized)
  # Include constraint to maintain variable between min_image and max_image
  imgs = tf.Variable(np.random.random([1, 224, 224, 3]), dtype=tf.float32, 
                     constraint=lambda t: tf.clip_by_value(t, min_image, max_image))

  # apply the vgg preprocessing for loaded checkpoint
  img_preproc = tf.subtract(imgs, subtract_value)
  img_preproc = tf.multiply(img_preproc, multiply_value)
  img_preproc = mean_image_subtraction(img_preproc, means)


  ### Now build the model, and load the saved checkpoint ###

  # Get vgg.py from https://github.com/tensorflow/models/tree/master/research/slim
  logits, nets = vgg.vgg_19(img_preproc, is_training=False, scope='vgg_19')

  # Make a saver and load the checkpoint
  # model checkpoint http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
  saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19'))
  saver.restore(session, 'vgg_19.ckpt')


  ### Include pointers to the layers we will use for metamer generation ###
  ### Much of this block is bookkeeping to make accessing intermediate layers easier ###

  # The layers that were used in Feather et al. 2019 ('jitted_relu' was removed for figure labels)
  metamer_layers = ['conv1_2_jittered_relu',
                    'conv2_2_jittered_relu',
                    'conv3_4_jittered_relu',
                    'conv4_4_jittered_relu',
                    'conv5_4_jittered_relu',
                    'fc6_jittered_relu',
                    'fc7_jittered_relu',
                    'fc8'] 

  nets['input_image'] = imgs 
  nets['image_prepoc'] = img_preproc
  nets['logits'] = logits
  nets['min_image'] = min_image
  nets['max_image'] = max_image
  nets['subtract_value'] = subtract_value
  nets['multiply_value'] = multiply_value
  nets['visualization_input'] = nets['input_image']
  nets['predictions'] = tf.nn.softmax(logits)
  nets['class_labels_key'] = class_names
  nets['imagenet_logits'] = nets['logits']

  # Some of the tfslim networks have an offset for the class index. The saved VGG model does not.
  nets['class_index_offset'] = int(0) 


  ### The following dictionaries and code are used to grab the activations of the conv layers ###
  ### before the non-linearity is applied, so that we can apply the modified gradient ReLU.   ###
  ### The activations after the modified gradient ReLU will be the same as the activations    ###
  ### after the normal gradient ReLU.                                                         ###

  # Get the pre-relu layers and add them to nets, format <layer_name_in_nets>:<name_in_graph>
  layers_pre_relu = {'conv1_2_prerelu':'vgg_19/conv1/conv1_2/BiasAdd:0',
                     'conv2_2_prerelu':'vgg_19/conv2/conv2_2/BiasAdd:0',
                     'conv3_4_prerelu':'vgg_19/conv3/conv3_4/BiasAdd:0',
                     'conv4_4_prerelu':'vgg_19/conv4/conv4_4/BiasAdd:0',
                     'conv5_4_prerelu':'vgg_19/conv5/conv5_4/BiasAdd:0',
                     'fc6_prerelu':'vgg_19/fc6/BiasAdd:0',
                     'fc7_prerelu':'vgg_19/fc7/BiasAdd:0'
                    }

  # remap some of the names in nets for easy access
  # for VGG, all of these values are after the non-linearity is applied 
  nets['conv1_2'] = nets['vgg_19/conv1/conv1_2']
  nets['conv2_2'] = nets['vgg_19/conv2/conv2_2']
  nets['conv3_4'] = nets['vgg_19/conv3/conv3_4']
  nets['conv4_4'] = nets['vgg_19/conv4/conv4_4']
  nets['conv5_4'] = nets['vgg_19/conv5/conv5_4']
  nets['fc6'] = nets['vgg_19/fc6']
  nets['fc7'] = nets['vgg_19/fc7']
  nets['fc8'] = nets['vgg_19/fc8']

  add_jitter_layers = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4', 'fc6', 'fc7']

  # This logic is helpful for networks such as ResNet, where we modify the relu for all of the layers 
  # that will be concatenated at the end of the block. 
  for layer_key, layer_name in layers_pre_relu.items():
    if type(layer_name) is list:
        concat_layers_mixed = []
        # Some of the mixed layers have layers that are mixed... need to jitter the relu for each? 
        for concat_layer in layer_name:
            concat_layers_mixed.append(tf.get_default_graph().get_tensor_by_name(concat_layer))
        nets[layer_key] = concat_layers_mixed
    else: 
        nets[layer_key] = tf.get_default_graph().get_tensor_by_name(layer_name)

  for layer in add_jitter_layers:
      nets, net_layer_name = add_jitter_relu_to_layer_vgg(nets, layer)


  # Initialize the input variable, check if other things aren't initialized (useful for generating 
  # metamers from a random network)
  uninitialized = tf.report_uninitialized_variables().eval(session=session)
  print('##### \n UNINITIALIZED VARIABLES ARE:')
  print(uninitialized)
  print('#####')
  all_variables = tf.global_variables()
  init_op = tf.variables_initializer([var for var in all_variables if 
                any([var_name.decode('utf-8') in var.name for var_name in uninitialized.tolist()])])
  init_op.run(session=session)


  ### Remaining code block runs some sanity checks with an example image. ###

  # Pull in an example image that is classified correctly (it is an airplane from imagenet).
  image_path = 'assets/airplane.png'
  image_class = 'airliner'
  image_dict = metamer_helpers.use_image_path_specified_image(image_path,
                                                              image_class=image_class,
                                                              im_shape=224)

  # Normalize between 0-1, since our variables are normalized to those values.
  image_dict['image'] = (image_dict['image']-image_dict['min_value_image_set']) / (
                        image_dict['max_value_image_set']-image_dict['min_value_image_set'])

  eval_predictions = session.run(nets['predictions'], 
                                 feed_dict={imgs: [image_dict['image']]}).ravel()
  sorted_predictions = np.argsort(eval_predictions)[::-1]
  prediction_check_msg = 'Predicted image for airliner example is %s with %f prob' % (
                         class_names[sorted_predictions[0] + nets['class_index_offset']], 
                         eval_predictions[sorted_predictions[0]])
  predicted_class = class_names[sorted_predictions[0] + nets['class_index_offset']]
  assert predicted_class==image_class, prediction_check_msg
    
  # Make sure that the activations are the same between the normal relu and the modified gradient
  # relu for an example layer. 
  same_layers = {'normal_relu':nets['conv3_4'], 
                 'modified_grad_relu':nets['conv3_4_jittered_relu']}
  check_relu = session.run(same_layers, feed_dict={imgs: [image_dict['image']]})
  relu_check_msg = ('The activations after the modified gradient ReLU do not '
                    'match the activations after the normal gradient ReLU.')
  assert np.all(check_relu['normal_relu'] == check_relu['modified_grad_relu']), relu_check_msg

  return nets, session, metamer_layers

if __name__== "__main__":
  main()
