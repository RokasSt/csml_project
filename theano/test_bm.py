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
import os
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

arg_parser.add_argument('--num_samples', type = str, required = True)

arg_parser.add_argument('--num_chains', type = str, required = True)

arg_parser.add_argument('--trained_subset', type = str, required = True)

arg_parser.add_argument('--num_steps', type = str, required = True)

FLAGS, _         = arg_parser.parse_known_args()

path_to_params   = FLAGS.path_to_params

num_samples      = int(FLAGS.num_samples)

num_chains       = int(FLAGS.num_chains)

trained_subset   = int(FLAGS.trained_subset)

num_steps        = int(FLAGS.num_steps)

split_path       = os.path.split(path_to_params)

if bool(trained_subset):
    
   indices =np.loadtxt(os.path.join(split_path[0],"LEARNT_INSTANCES.dat"))
   
   indices = np.asarray(indices, dtype = np.int64)
   
   test_inputs = train_images[indices,:]
   
   #override a given num_chains
   
   num_chains  = len(indices)
   
else:
    
   test_inputs = test_images
   
if "INIT" in split_path[1]:
    
   save_to_path = os.path.join(split_path[0],"samples_init.jpeg")
   
else:
    
   save_to_path = os.path.join(split_path[0],"samples.jpeg")

bm = BoltzmannMachine(num_vars        = input_dim, 
                      training        = False)
                      
bm.load_model_params(full_path = path_to_params)

start_time = timeit.default_timer()

if bool(trained_subset):
    
   if num_chains ==1:
       
      test_inputs = np.reshape(test_inputs,[1,len(test_inputs)])

   bm.test_relative_probability(inputs = test_inputs, trained= True)

   rand_samples = bm.np_rand_gen.binomial(n=1,p=0.5, 
                              size = (test_inputs.shape[0], 784))
              
   rand_samples = np.asarray(rand_samples, 
                             dtype = theano.config.floatX)

   bm.test_relative_probability(inputs = rand_samples, trained = False)

bm.sample_from_bm(test_inputs  = test_inputs,
                  num_chains   = num_chains, 
                  num_samples  = num_samples,
                  num_steps    = num_steps,
                  save_to_path = save_to_path)
                   
end_time = timeit.default_timer()
                   
print('Image generation took %f minutes'%((end_time - start_time)/60.))


                   

                   

                      
