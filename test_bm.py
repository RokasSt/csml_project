""" 
Author: Rokas Stanislovas
MSc Project: Likelihood Approximations
for Energy-Based Models
MSc Computational Statistics and 
Machine Learning
"""

import numpy as np
import argparse
import shutil
import os
import sys
import json
import tensorflow as tf
from   tensorflow.examples.tutorials.mnist import input_data
import theano
import theano.tensor as T
import datetime
import utils
import argparse
import timeit
from   model_classes import BoltzmannMachine

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

test_images      = np.round(mnist.test.images)

test_labels      = mnist.test.labels

train_images     = np.round(mnist.train.images)

train_labels     = mnist.train.labels

num_train_images = train_images.shape[0]

input_dim        = train_images.shape[1]

assert input_dim == 784

validate_images  = np.round(mnist.validation.images)

validate_labels  = mnist.validation.labels

arg_parser       = argparse.ArgumentParser()

arg_parser.add_argument('--path_to_params', type=str,required= True)

arg_parser.add_argument('--num_steps', type = str, required = True)

arg_parser.add_argument('--num_samples', type = str, required = True)

arg_parser.add_argument('--num_chains', type = str, required = True)

FLAGS, _           = arg_parser.parse_known_args()

path_to_params     = FLAGS.path_to_params

num_steps          = int(FLAGS.num_steps)

num_samples        = int(FLAGS.num_samples)

num_chains         = int(FLAGS.num_chains)

save_to_path       = os.path.split(path_to_params)[0]

save_to_path       = os.path.join(save_to_path,"samples.jpeg")

bm = BoltzmannMachine(num_vars = input_dim, 
                      training_inputs = train_images,
                      test_inputs     = test_images,
                      algorithm       = None,
                      batch_size      = None,
                      num_samples     = None,
                      training        = False)
                      
bm.load_model_params(full_path = path_to_params)

start_time = timeit.default_timer()

bm.sample_from_bm(num_chains   = num_chains, 
                  num_samples  = num_samples, 
                  num_steps    = num_steps, 
                  save_to_path = save_to_path)
                   
end_time = timeit.default_timer()
                   
print ('Image generation took %f minutes' % ((end_time - start_time)/ 60.))
                      
