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

report_step      = 1

test_mode        = False

num_to_test      = 50

test_images      = np.round(mnist.test.images)

test_labels      = mnist.test.labels

train_images     = np.round(mnist.train.images)

train_labels     = mnist.train.labels

N_train          = train_images.shape[0]

input_dim        = train_images.shape[1]

assert input_dim == 784

validate_images  = np.round(mnist.validation.images)

validate_labels  = mnist.validation.labels

arg_parser       = argparse.ArgumentParser()

arg_parser.add_argument('--num_epochs', type=str,required= True)

arg_parser.add_argument('--algorithm', type=str,required= True)

arg_parser.add_argument('--num_samples', type = str, required = False)

arg_parser.add_argument('--num_steps', type = str, required = False)

arg_parser.add_argument('--include_all', type = str, required = False)

arg_parser.add_argument('--batch_size', type= str, required=  True)

arg_parser.add_argument('--learning_rate', type = str, required = True)

arg_parser.add_argument('--experiment', type = str, required = True)

FLAGS, _        = arg_parser.parse_known_args()

algorithm       = FLAGS.algorithm

num_epochs      = int(FLAGS.num_epochs)

batch_size      = int(FLAGS.batch_size)

learning_rate   = float(FLAGS.learning_rate)

experiment_tag  = FLAGS.experiment

dir_name ="logs_%s"%algorithm

print("Algorithm : %s"%algorithm)
print("Experiment: %s"%experiment_tag)

if algorithm == "CSS":
    
   num_samples = int(FLAGS.num_samples)

   specs = (str(learning_rate),
            batch_size, 
            num_samples, 
            datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))

   exp_tag = "LR%sBS%dNS%d_%s"%specs
   
   num_steps = None
   
   include_all = None
   
if algorithm == "CD1":
    
   num_steps = int(FLAGS.num_steps)
   
   specs = (str(learning_rate),
            batch_size,
            num_steps,
            datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))

   exp_tag = "LR%sBS%dNST%d_%s"%specs
   
   num_samples = None 
   
   include_all = None
   
if algorithm == "CSS_MF":
    
   num_steps    = int(FLAGS.num_steps)
   
   num_samples  = int(FLAGS.num_samples)
   
   include_all = int(FLAGS.include_all)
   
   specs = (str(learning_rate),
            batch_size,
            num_steps,
            num_samples,
            include_all,
            datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))
            
   include_all = bool(include_all)

   exp_tag = "LR%sBS%dNST%dNS%dINC%d_%s"%specs
   
num_iterations = N_train // batch_size

losses = []

bm = BoltzmannMachine(num_vars        = input_dim, 
                      training_inputs = train_images,
                      algorithm       = algorithm,
                      batch_size      = batch_size,
                      learning_rate   = learning_rate,
                      num_samples     = num_samples,
                      num_steps       = num_steps,
                      include_all    = include_all)
                      
                      
if test_mode:
    
   print("Testing")
   bm.test_compute_energy_time(num_to_test)
   bm.test_get_mf_samples()
   sys.exit()
   
else:
    
   exp_path = os.path.join(dir_name,exp_tag)
        
   os.makedirs(exp_path)
                      
cd_sampling, optimize = bm.add_graph()

start_time = timeit.default_timer()

epoch_time0 = start_time

for epoch_index in range(num_epochs):
    
    permuted_inds = np.random.permutation(N_train)
    
    avg_cost_val = []
    
    for iter_index in range(num_iterations):
        
        iter_start_time = timeit.default_timer()
    
        minibatch_inds = permuted_inds[batch_size*iter_index:batch_size*(iter_index+1)]
        
        if algorithm =="CSS":
           
           sampled_indices = bm.select_samples(minibatch_inds)
        
           approx_minibatch_cost = optimize(sampled_indices, list(minibatch_inds))
           
        if algorithm =="CSS_MF":
            
           mf_t0 = timeit.default_timer()
           bm.do_mf_updates()
           mf_t1 = timeit.default_timer()
           print("4 steps of MF updates took --- %f"%((mf_t1 -mf_t0)/60.0))
           
           if include_all:
              opt_t0 = timeit.default_timer()
              ###
              approx_minibatch_cost = optimize(range(N_train), list(minibatch_inds))
              ###
              opt_t1 = timeit.default_timer()
              print("Optimization step took --- %f"%((opt_t1 - opt_t0)/60.0))
           else:
              opt_t0 = timeit.default_timer()
              ###
              approx_minibatch_cost = optimize(list(minibatch_inds))
              ###
              opt_t1 = timeit.default_timer()
              print("Optimization step took --- %f"%((opt_t1 - opt_t0)/60.0))
           
        if algorithm =="CD1":
            
           mf_sample, cd_sample = cd_sampling(list(minibatch_inds))
           
           bm.x_gibbs.set_value(np.transpose(cd_sample)) 
           
           approx_minibatch_cost = optimize(list(minibatch_inds))
        
        avg_cost_val.append(approx_minibatch_cost)
        
        if iter_index % report_step ==0:
            
           print('Training epoch %d ---- Iter %d ---- cost value: %f'
           %(epoch_index, iter_index, approx_minibatch_cost))
           
        iter_end_time = timeit.default_timer()
        
        print('Training iteration took %f minutes'%
        ((iter_end_time - iter_start_time) / 60.))
        
    avg_cost_val = np.mean(avg_cost_val)
    
    losses.append(avg_cost_val)
        
    print('Training epoch %d ----- average cost value: %f' %(epoch_index, avg_cost_val))
    
    epoch_time = timeit.default_timer()
    
    print ('Training epoch took %f minutes' % ( (epoch_time - epoch_time0) / 60.))
    
    bm.save_model_params(os.path.join(exp_path,"TRAINED_PARAMS.model"))
    
    epoch_time0 = epoch_time
    
training_time = epoch_time0 - start_time

print('Training took %f minutes'%(training_time/ 60.))
    
np.savetxt(os.path.join(exp_path,"TRAIN_LOSSES.dat"), losses)

np.savetxt(os.path.join(exp_path,"TRAINING_TIME.dat"), np.array([training_time]))
    
    
    
    
    
    
    

  









