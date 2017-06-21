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
import datetime
import utils
import argparse
import timeit
import os

print("Importing data:")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

report_step      = 1

save_init_weights= True

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

arg_parser.add_argument('--num_epochs', type = str,required= True)

arg_parser.add_argument('--algorithm', type = str,required= True)

arg_parser.add_argument('--num_samples', type = str, required = False)

arg_parser.add_argument('--num_steps', type = str, required = False)

arg_parser.add_argument('--num_data', type = str, required = False)

arg_parser.add_argument('--resample', type = str, required = False)

arg_parser.add_argument('--batch_size', type = str, required=  True)

arg_parser.add_argument('--learning_rate', type = str, required = True)

arg_parser.add_argument('--experiment', type = str, required = True)

arg_parser.add_argument('--use_gpu', type = str, required = True)

FLAGS, _        = arg_parser.parse_known_args()

algorithm       = FLAGS.algorithm

num_epochs      = int(FLAGS.num_epochs)

use_gpu         = int(FLAGS.use_gpu)

batch_size      = int(FLAGS.batch_size)

learning_rate   = float(FLAGS.learning_rate)

experiment_tag  = FLAGS.experiment

dir_name ="logs_%s"%algorithm

print("Algorithm : %s"%algorithm)
print("Experiment: %s"%experiment_tag)

if algorithm == "CD1":
    
   assert FLAGS.num_steps != None 
    
   num_cd_steps = int(FLAGS.num_steps)
   
   specs = (str(learning_rate),
            batch_size,
            num_steps,
            datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))

   exp_tag = "LR%sBS%dNST%d_%s"%specs
   
   num_samples = None 
   
   num_data    = None
   
   resample    = None
   
if algorithm   == "CSS":
   
   assert FLAGS.num_samples != None
   
   assert FLAGS.num_data    != None
    
   num_cd_steps   = None
   
   num_samples    = int(FLAGS.num_samples)
   
   num_data       = int(FLAGS.num_data)
   
   if num_samples ==0:
   
      specs = (str(learning_rate),
               batch_size,
               num_samples,
               num_data,
               datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))
   
      exp_tag = "LR%sBS%dNS%dNUM_DATA%d_%s"%specs
      
      resample = None
      
   if num_samples > 0:
       
      assert FLAGS.resample != None
       
      resample = int(FLAGS.resample)
       
      specs = (str(learning_rate),
               batch_size,
               num_samples,
               resample,
               num_data,
               datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))
   
      exp_tag = "LR%sBS%dNS%dRS%dNUM_DATA%d_%s"%specs
      
      resample = bool(resample)
   
num_iterations = N_train // batch_size

losses = []

if bool(use_gpu):    
   print("Will attempt to use GPU")
   os.environ['THEANO_FLAGS'] = 'device=cuda'
   
import theano
import theano.tensor as T
from   model_classes import BoltzmannMachine

bm = BoltzmannMachine(num_vars        = input_dim, 
                      training_inputs = train_images,
                      algorithm       = algorithm,
                      batch_size      = batch_size,
                      learning_rate   = learning_rate,
                      num_samples     = num_samples,
                      num_cd_steps    = num_cd_steps,
                      num_data        = num_data,
                      unique_samples  = resample)
    
exp_path = os.path.join(dir_name,exp_tag)
        
os.makedirs(exp_path)
   
if save_init_weights:
    
   bm.save_model_params(os.path.join(exp_path,"INIT_PARAMS.model"))
   print("saved initial weights of fully visible Boltzmann Machine")
   
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
           
           if num_samples > 0:
              
              #mf_t0 = timeit.default_timer()
              #bm.do_mf_updates(num_steps =4)
              #mf_t1 = timeit.default_timer()
              #print("4 steps of MF updates took --- %f"%((mf_t1 -mf_t0)/60.0))
              # alternatively, try reinitiate distribution.
              
              reinit_mf = bm.np_rand_gen.uniform(0,1, size = (bm.num_vars, bm.num_samples))
        
              reinit_mf = np.asarray(reinit_mf, dtype = theano.config.floatX)
              
              bm.mf_params.set_value(reinit_mf)
              
           sampled_indices = bm.select_data(minibatch_inds)
           
           opt_t0 = timeit.default_timer()
           ###
           approx_minibatch_cost = optimize(sampled_indices, 
                                            list(minibatch_inds))
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
    
training_time = (epoch_time0 - start_time)/60.0

print('Training took %f minutes'%training_time)
    
np.savetxt(os.path.join(exp_path,"TRAIN_LOSSES.dat"), losses)

np.savetxt(os.path.join(exp_path,"TRAINING_TIME.dat"), np.array([training_time]))
    
    
    
    
    
    
    

  









