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

print("Importing data:")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

test_images      = mnist.test.images

test_labels      = mnist.test.labels

train_images     = mnist.train.images

train_labels     = mnist.train.labels

num_train_images = train_images.shape[0]

input_dim        = train_images.shape[1]

assert input_dim == 784

validate_images  = mnist.validation.images

validate_labels  = mnist.validation.labels

arg_parser       = argparse.ArgumentParser()

arg_parser.add_argument('--num_epochs', type=str,required= True)

arg_parser.add_argument('--algorithm', type=str,required= True)

arg_parser.add_argument('--num_samples', type = str, required = True)

arg_parser.add_argument('--batch_size', type= str, required=  True)

arg_parser.add_argument('--learning_rate', type = str, required = True)

arg_parser.add_argument('--experiment', type = str, required = True)

FLAGS, _        = arg_parser.parse_known_args()

algorithm       = FLAGS.algorithm

num_epochs      = int(FLAGS.num_epochs)

batch_size      = int(FLAGS.batch_size)

learning_rate   = float(FLAGS.learning_rate)

num_samples     = int(FLAGS.num_samples)

experiment_tag  = FLAGS.experiment

dir_name ="logs_%s"%experiment_tag

specs = (algorithm,
         str(learning_rate),
         batch_size, 
         num_samples, 
         datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))

exp_tag = "%s_LR%sBS%dNS%d_%s"%specs
        
exp_path = os.path.join(dir_name,exp_tag)
        
os.makedirs(exp_path)

num_iterations = num_train_images // batch_size

losses = []

### add definitions computational graph
bm = BoltzmannMachine(num_vars = input_dim, 
                      training_inputs = train_images,
                      test_inputs     = test_images,
                      algorithm       = algorithm)

bm.add_objective(batch_size, num_samples)

bm.add_grad_updates(learning_rate)   

bm.add_pseudo_cost_measure(batch_size)

init_chains = bm.init_chains()

optimize = bm.optimization_step()
### end adding definitions

start_time = timeit.default_timer()

epoch_time0 = start_time

for epoch_index in range(num_epochs):
    
    permuted_inds = np.random.permutation(num_train_images)
    
    avg_cost_val = []
    
    for iter_index in range(num_iterations):
    
        selected_inds = permuted_inds[batch_size*iter_index:batch_size*(iter_index+1)]
        
        if algorithm =="CSS":
        
           sampled_indices = bm.select_samples(selected_inds, num_samples)
        
           approx_minibatch_cost = optimize(sampled_indices, list(selected_inds))
           
        if algorithm =="CD1":
           
           init_chains(list(selected_inds))
           
           approx_minibatch_cost = optimize(list(selected_inds))
        
        avg_cost_val.append(approx_minibatch_cost)
        
        if iter_index % 50 ==0:
            
           print('Training epoch %d ---- Iter %d ---- cost value: %f'
           %(epoch_index, iter_index, approx_minibatch_cost))
        
    avg_cost_val = np.mean(avg_cost_val)
    
    losses.append(avg_cost_val)
        
    print('Training epoch %d ----- average cost value: %f' %(epoch_index, avg_cost_val))
    
    epoch_time = timeit.default_timer()
    
    print ('Training epoch took %f minutes' % ( (epoch_time - epoch_time0) / 60.))
    
    bm.save_model_params(os.path.join(exp_path,"TRAINED_PARAMS.model"))
    
    epoch_time0 = epoch_time
    
training_time = epoch_time0 - start_time

print ('Training took %f minutes' % ( training_time / 60.))
    
np.savetxt(os.path.join(exp_path,"TRAIN_LOSSES.dat"), losses)

np.savetxt(os.path.join(exp_path,"TRAINING_TIME.dat"), np.array([training_time]))
    
    
    
    
    
    
    

  









