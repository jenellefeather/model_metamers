import sys
import tfcochleagram
import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wav
import pickle 
import sys 
import json
import os 
import scipy
import matplotlib.pylab as plt
import audio_cnn_helpers
import metamer_helpers

# Jittered relu grad is only applied to the metamer generation layer. 
# This modification to the gradient helps with optimization for the final layer. 
@tf.custom_gradient
def jittered_relu_grad(x):
    y = tf.nn.relu(x)
    def grad(dy): #clip the zeros.
        dy_shape = dy.get_shape()
        # Normal relu gradient is equivalent to tf.where(x<=0, 0*dy, 1*dy)
        return tf.where(x<=0, dy, dy)
    return y, grad

# Build our network
def build_net(_):
    pckl_file = 'word_network_aliased.pckl'
    ckpt_path = 'word_aliased.ckpt'

    # Parameters to build the cochleagram input, same as used for training
    signal_rate = 20000
    signal_length_s = 2
    COCH_PARAMS = {
        "ENV_SR":200,
        "HIGH_LIM":8000,
        "LOW_LIM":20,
        "N":50,
        "SAMPLE_FACTOR":4,
        "compression":"clipped_point3",
        "rFFT":True,
        "reshape_kell2018":False,
        "erb_filter_kwargs":{'no_lowpass':False, 'no_highpass':False},
        # Chosen to normalize a dataset a while ago and used to train these models
        "scale_before_compression":796.87416837456942
    }

    net_name = 'word_aliased'
    
    # Load pickle containing the network specification
    with open(pckl_file, 'rb') as f:
        pckled_network = pickle.load(f)
   
    # Make a variable input tensor (will be optimized)
    input_tensor = tf.Variable(np.ones([1,signal_rate*signal_length_s]), 
                               dtype=tf.float32)
    trainable = False
    training = False

    nets = {'input_signal':input_tensor}

    # Start a session so that we can easily load the variables. 
    sess = tf.Session()

    # Make the cochleagram graph (input into the word neural network)
    with tf.variable_scope('cochlear_network'):
        coch_container = tfcochleagram.cochleagram_graph(nets,
                                                         signal_rate,
                                                         **COCH_PARAMS)
        input_tensor = nets['cochleagram']

    # Builds the network from the saved pckl for the audio network
    with tf.variable_scope('brain_network'):
        for layer_idx, layer in enumerate(pckled_network['layer_list']):
            layer_name = pckled_network['graph_architecture'][layer_idx]['args']['name']
            layer_type = pckled_network['graph_architecture'][layer_idx]['layer_type']
            if layer_type == 'tf.layers.batch_normalization':
                nets[layer_name]= layer(input_tensor, trainable=trainable, training=training)
            elif layer_type == 'tf.layers.dropout':
                nets[layer_name] = layer(input_tensor, training=training)
            elif layer_type == 'tf.layers.conv2d':
                nets[layer_name] = layer(input_tensor, trainable=trainable)
            else: 
                nets[layer_name] = layer(input_tensor)
            input_tensor = nets[layer_name]

    # Load all of the variables in the scope "brain_network" (excludes the input signal)
    brain_globals = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='brain_network')
    brain_locals = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='brain_network')

    # Load a checkpoint
    saver = tf.train.Saver(var_list=brain_locals+brain_globals)
    saver.restore(sess, ckpt_path)

    nets['visualization_input'] = nets['cochleagram']
    nets['logits'] = nets['fc_top']['/stimuli/word_int']
    nets['predictions'] = tf.nn.softmax(nets['logits'])

    # For experiments in Feather et al. 2019 we generated metamers matched to the RELU after each conv 
    # of fully connected layer. 
    # This code applies a modified gradient relu after each.  
    for pre_layer in ['conv_0', 'conv_1', 'conv_2', 'conv_3', 'conv_4', 'fc_intermediate']:
        layer_pre_relu = nets[pre_layer]
        nets['%s_jittered_relu'%pre_layer] = jittered_relu_grad(layer_pre_relu)

    # Choose the layers for the optimization 
    metamer_gen_layers = ['visualization_input', 
                  'conv_0_jittered_relu', # After the relu activation
                  'conv_1_jittered_relu', 
                  'conv_2_jittered_relu', 
                  'conv_3_jittered_relu', 
                  'conv_4_jittered_relu', 
                  'fc_intermediate_jittered_relu', 
                  'logits']

    # Load in the encodings for this network
    word_and_speaker_encodings = pickle.load(open('assets/metamer_word_encodings.pckl', 'rb'))
    nets['idx_to_label'] = word_and_speaker_encodings['word_idx_to_word']
    class_names = nets['idx_to_label']
    nets['class_index_offset'] = 0


    ### Remaining code block runs some sanity checks with an example sound. ###

    # Pull in an example sound that is classified correctly (it contains the word "human")
    audio_path = 'assets/human_audio_resampled.wav'
    wav_word = 'human'
    audio_dict = metamer_helpers.use_audio_path_specified_audio(audio_path,
                                                                wav_word,
                                                                rms_normalize=0.1)
    eval_predictions = sess.run(nets['predictions'],
                                   feed_dict={nets['input_signal']: [audio_dict['wav']]}).ravel()
    sorted_predictions = np.argsort(eval_predictions)[::-1]
    prediction_check_msg = 'Predicted word for human example is %s with %f prob' % (
                           class_names[sorted_predictions[0] + nets['class_index_offset']],
                           eval_predictions[sorted_predictions[0]])
    predicted_class = class_names[sorted_predictions[0] + nets['class_index_offset']]
    assert predicted_class==wav_word, prediction_check_msg

    # Make sure that the activations are the same between the normal relu and the modified gradient
    # relu for an example layer.
    same_layers = {'normal_relu':nets['relu_3'],
                   'modified_grad_relu':nets['conv_3_jittered_relu']}
    check_relu = sess.run(same_layers, feed_dict={nets['input_signal']: [audio_dict['wav']]})
    relu_check_msg = ('The activations after the modified gradient ReLU do not '
                      'match the activations after the normal gradient ReLU.')
    assert np.all(check_relu['normal_relu'] == check_relu['modified_grad_relu']), relu_check_msg

    return nets, sess, metamer_gen_layers


def main():
    nets, session, metamer_gen_layers = build_net('_')
    return nets, session, metamer_gen_layers

if __name__== "__main__":
    main()
