"""
Contains functions that are used for image loading and metamer generation. 
This code was used for the following paper: 
  Metamers of neural networks reveal divergence from human perceptual systems
  Jenelle Feather, Alex Durango, Ray Gonzalez, Josh McDermott
  In Advances in Neural Information Processing Systems (2019)
  PDF: https://papers.nips.cc/paper/9198-metamers-of-neural-networks-reveal-divergence-from-human-perceptual-systems.pdf
"""

import scipy.io.wavfile as wav
import pickle
import numpy as np
import scipy
from scipy.misc import imread, imresize
import resampy
import tensorflow as tf

def use_audio_path_specified_audio(wav_path, wav_word, 
                                   rms_normalize=None, 
                                   SR=20000):
  """
  Loads an example wav specified by wav_path

  Inputs: 
   wav_path (string) : filepath to the audio to load
   wav_word (string) : label for the audio in wav_path
   rms_normalize (float) : the rms value to set the audio to 
   SR (int) : sampling rate of the desired audio. The file at 
     wav_path will be resampled to this value

  Output: 
    audio_dict (dictionary) : a dictionary containing the loaded 
      audio and preprocessing parameters
  """

  metamer_word_encodings = pickle.load(open('assets/metamer_word_encodings.pckl', 'rb'))
  word_to_int = metamer_word_encodings['word_to_word_idx']

  print("Loading: %s"%wav_path)
  SR_loaded, wav_f = scipy.io.wavfile.read(wav_path)
  if SR_loaded != SR:
    wav_f = resampy.resample(wav_f, SR_loaded, SR)
    SR_loaded = SR

  if rms_normalize is not None:
    wav_f = wav_f - np.mean(wav_f.ravel())
    wav_f = wav_f/(np.sqrt(np.mean(wav_f.ravel()**2)))*rms_normalize
    rms = rms_normalize
  else:
    rms = np.sqrt(np.mean(wav_f.ravel()**2))

  audio_dict={}

  audio_dict['wav'] = wav_f
  audio_dict['SR'] = SR
  audio_dict['word_int'] = word_to_int[wav_word]
  audio_dict['word'] = wav_word
  audio_dict['rms'] = rms
  audio_dict['filename'] = wav_path
  audio_dict['filename_short'] = wav_path.split('/')[-1]
  audio_dict['correct_response'] = wav_word

  return audio_dict


def image_center_crop(input_img, im_shape=224):
  """
  Returns a square portion of the input image, rescaled to im_shape
  
  Inputs: 
    input_img (np array) : an image stored as [H,W,C] 
    im_shape (int) : the square shape to crop the image

  Outputs: 
    input_img (np array) : The resized numpy array [im_shape, im_shape, C]
  """
  image_shape_hw = input_img.shape[0:2]
  smallest_dim = min(image_shape_hw)
  cropped_img = input_img[int((image_shape_hw[0]/2-smallest_dim/2)):int((image_shape_hw[0]/2+smallest_dim/2)), 
                int((image_shape_hw[1]/2-smallest_dim/2)):int((image_shape_hw[1]/2+smallest_dim/2)), :]
  # Now that it is square, resize it 
  cropped_img = imresize(cropped_img, (im_shape, im_shape))
  return cropped_img


def use_image_path_specified_image(image_path, image_class=None, im_shape=224):
  """
  Loads and image from a specified path and performs preprocessing. 

  Inputs:
    image_path (string) : filepath to an image
    image_class (string) : the class name for the image
    im_shape (int) : the height of the image (square)

  Outputs:
    image_dict (dictionary) : contains the input image and preprocessing info
  """

  print("Loading: %s"%image_path)
  input_img = scipy.misc.imread(image_path, mode='RGB')
  input_img = image_center_crop(input_img, im_shape=im_shape)

  image_dict = {}
  image_dict['image'] = input_img
  image_dict['shape'] = im_shape
  image_dict['filename'] = image_path
  image_dict['filename_short'] = image_path.split('/')[-1]
  image_dict['correct_response'] = image_class
  image_dict['max_value_image_set'] = 255
  image_dict['min_value_image_set'] = 0
  return image_dict


def _make_pink_noise(T,rms_normalization=False):
    """
    Makes a segment of pink noise length T and returns a numpy array with the values

    Parameters
    ----------
    T : int
        length of the pink noise to generate
    rms_normalization : float
        normalization factor for the pink noise, ie the rms of a test signal. default no normalization.

    Returns
    -------
    pink_noise : numpy array
        numpy array containing pink noise of length T
    rms_pint : float
        the rms of the pink noise

    """
    uneven = T%2
    X = np.random.randn(T//2+1+uneven) + 1j * np.random.randn(T//2+1+uneven)
    S = np.sqrt(np.arange(len(X))+1.)
    pink_noise = (np.fft.irfft(X/S)).real
    if uneven:
        pink_noise = pink_noise[:-1]
    rms_pink = np.sqrt(np.mean(pink_noise**2))
    if rms_normalization: # basic normalization of pink noise
        pink_noise = (rms_normalization/rms_pink)*pink_noise
        rms_pink = np.sqrt(np.mean(pink_noise**2))
    return pink_noise, rms_pink


def make_initialization_op_pink_audio(input_tensor, input_noise_shape, 
                                      audio_scaling=0.001, rms_normalize=False):
  """ 
  Makes a tensorflow operation to reassign the input audio noise, using pink noise. 

  This function is particularly helpful when generating multiple metamers from 
  the same random seed without rebuilding the tensorflow graph. 

  Inputs:
    input_tensor (tensorflow tensor) : the variable tensor that will be optimized
      during metamer generation
    input_noise_shape (float) : the length of the input audio 
    audio_scaling (float) : multiplicative value to directly scale the 
      initialized pink noise. Usually set less than 1, as a way to scale the noise. 
    rms_normalization (float) : whether to generate the pink noise at a specific
      rms value. Normally set to match the rms of the input audio to the network. 

  Outputs: 
    input_tensor (tensorflow tensor) : the same as input to the function.
    input_noise_assign_op (tensorflow op) : when called with sess.run will
      reinitialize input_tensor to the noise value.
  """
  pink_noise, rmspink = _make_pink_noise(input_noise_shape, 
                                        rms_normalization=rms_normalize)
  noise_initialization = (np.expand_dims(pink_noise,0)-np.mean(pink_noise))*audio_scaling
  input_noise_assign_op = tf.assign(input_tensor, noise_initialization)
  return input_tensor, input_noise_assign_op


def make_initialization_op_random_image(input_tensor, max_image, min_image, 
                                        scaling_factor=0.1, im_shape=224):
  """
  Generates a random noise image to use as an initialization for the 
  metamer variable input during the optimization. 

  This function is particularly useful when generating multiple metamers from the 
  same random seed without rebuilding the tensorflow graph. 

  Inputs: 
    input_tensor (tensorflow tensor) : the variable tensor that will be optimized
      during metamer generation
    max_image (float) : the maximum value that the variable can be, used to 
      center the noise
    min_image (float) : the minimum value that the varibale can be, used to 
      center the noise 
    scaling_factor (float) : how much to scale the input

  Outputs: 
    input_tensor (tensorflow tensor) : the same as input to the function. 
    input_noise_assign_op (tensorflow op) : when called with sess.run will 
      reinitialize input_tensor to the noise value.
  """
    
  # Make an op that reassins the input tensor to noise
  if type(im_shape)!=list:
    im_shape = [im_shape, im_shape, 3]
  noise = np.random.random([1, im_shape[0], im_shape[1], im_shape[2]])*scaling_factor + (max_image-min_image)/2
  input_noise_assign_op = tf.assign(input_tensor, noise)
  return input_tensor, input_noise_assign_op


def run_optimization(loss, sess, input_tensor, iterations_adam, log_loss_every_num, 
                     starting_learning_rate_adam=0.001, 
                     additional_output_tensors=None, 
                     adam_exponential_decay=0.95):
  """
  Sets up the optimizer for the metamer generation and runs the optimization. 

  Inputs: 
    loss (tensorflow tensor) : the loss to minimize to generate the output
    sess (tensorflow session) : the tensorflow session where the
      optimization will occur
    input_tensor (tensorflow tensor) : the variable tensor that will be optimized
      during metamer generation
    iterations_adam (int) : the number of iterations to run the adam optimizer
    log_loss_every_num (int) : will print the loss after each of these iterations
      and save the synthetic variable to a dictionary
    starting_learning_rate_adam (float) : learning rate for adam optimizer
    additional_output_tensors (list of tensorflow tensors) : If not None, these 
      tensors will be evaluated every log_loss_every_num iterations and saved to 
      the track_output dictionary
    adam_exponential_decay (float) : Sets the exponential decay for the adam optimizer

  Outputs: 
    total_loss (list) : tracks the loss during the optimization
    track_ites (list) : logs the iteration where we saved loss values
    track_output (dictionary) : saves evaluated tensors during the optmization. 
      Always contains the tensor that is being optimized, and can include additional 
      tensors if `additional_output_tensors` is specified.
    
  """
  if additional_output_tensors is not None: 
    additional_output_tensors['optimized_tensor'] = input_tensor

  # set up the optimizers--adam has a decaying learning rate
  step = tf.Variable(0, trainable=False)

  rate = tf.train.exponential_decay(starting_learning_rate_adam, step, 1000, adam_exponential_decay)
  optm = tf.train.AdamOptimizer(rate).minimize(loss, var_list=[input_tensor], global_step=step)

  # set up some values that we will track with each iteration
  total_loss = np.full(iterations_adam//log_loss_every_num+1,np.nan)
  track_iters = np.full(iterations_adam//log_loss_every_num+1,np.nan)
  track_output = {}
  
  # initialize any variables that were uninitialized before (ie the adam optimize variables)
  uninitialized = sess.run(tf.report_uninitialized_variables())
  all_variables = tf.global_variables()
  init_op = tf.variables_initializer([var for var in all_variables if any([var_name.decode('utf-8') in var.name for var_name in uninitialized.tolist()])])
  sess.run(init_op)
  
  # go through the adam optimization
  for i in range(iterations_adam+1):
    if i % log_loss_every_num == 0 or i==iterations_adam:
      total_loss[i//log_loss_every_num] = sess.run(loss)
      track_iters[i//log_loss_every_num] = i
      # TODO: check the R^2 values here? 
      if i % (log_loss_every_num) == 0:
        if additional_output_tensors is not None:
          track_output[i] = sess.run(additional_output_tensors)
        else:
          track_output[i] = sess.run(input_tensor)
      print("[%d/%d], loss=%f" % (i, iterations_adam, total_loss[i//log_loss_every_num])) # this seems okay.

    if i != iterations_adam: # log the loss for the last iteration but don't run the optimization
      sess.run(optm)
  
  return total_loss, track_iters, track_output

