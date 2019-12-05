"""
This script grabs random pairs of images and computes a distance measure
between their activation layers. Specifically, this computes and stores 
the Spearman R values, to match those used to construct the null in the 
corresponding NeurIPS paper. When called from the command line, a random
seed is taken as input in order to choose random image pairs. 1 million 
pairs were used to construct the null distribution for the NeurIPS paper. 

Note: This will not run without specifying a path to imagenet 
tfrecords (specifically this was run on those processed with tfslim).

This code is based off of that used for the following paper:
  Metamers of neural networks reveal divergence from human perceptual systems
  Jenelle Feather, Alex Durango, Ray Gonzalez, Josh McDermott
  In Advances in Neural Information Processing Systems (2019)
  PDF: https://papers.nips.cc/paper/9198-metamers-of-neural-networks-reveal-divergence-from-human-perceptual-systems.pdf
"""

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
import build_vgg_trained as build_network
import time
import glob
from imagenet import get_split
import argparse
from metamer_helpers import image_center_crop

parser = argparse.ArgumentParser()
parser.add_argument("seed", help="the random seed to use for dataset shuffle", type=int)
args = parser.parse_args()

split_name = 'train'
dataset_dir = <PATH_TO_TFSLIM_IMAGENET_TFRECORDS>

# Recommended to run multiple of these scripts in parallel
num_images = 4000 #(1-million images)/(250 parallel runs)

random_seed = args.seed #set random seed (different for each run)
im_shape = 224

slim = tf.contrib.slim
#build the network
nets, sess, metamer_layers = build_network.main()

# metamer_layers is a list of strings which are the keys of nets
layers_activations = [nets[layer] for layer in metamer_layers] 
dataset = get_split(split_name, dataset_dir)
data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, 
                                                               shuffle=True, 
                                                               seed=random_seed)
image_raw, label = data_provider.get(['image', 'label'])
image = tf.expand_dims(image_center_crop(image_raw, im_shape=im_shape), 0)

#all of the losses. shape=(num_image_pairs, num_layers)
spearman_r = []
with slim.queues.QueueRunners(sess):
    for i in range(num_images):
        
        #use the first 2 images then update the images_list index 
        image1 = sess.run(image)
        image2 = sess.run(image)
        spearman_r_pair = []
        
        layer_image1_activations = sess.run(layers_activations, 
                                            feed_dict={nets['input_image']:image1})
        layer_image2_activations = sess.run(layers_activations, 
                                            feed_dict={nets['input_image']:image2})
        
        for activations1, activations2 in zip(layer_image1_activations, 
                                              layer_image2_activations):
            activations1 = activations1.ravel()
            activations2 = activations2.ravel()
            spearman_r_pair.append(spearmanr(activations1, activations2)[0].0)
        
        if i%500==0:
            print('{:f} percent done'.format(i/num_images*100))
        
        spearman_r.append(spearman_r_pair) 
    
    spearman_r = np.array(spearman_r)

np.save('null_distributions/random_vgg19/spearman_r_%s.npy'%args.seed, spearman_r)
