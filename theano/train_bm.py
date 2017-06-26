""" 
Author: Rokas Stanislovas
MSc Project: Likelihood Approximations
for Energy-Based Models
MSc Computational Statistics and 
Machine Learning
"""

import numpy as np
import subprocess
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

report_step          = 1

test_mode            = False

save_images          = True

save_best_params     = True

test_add_complements = False

test_comp_energies   = False

save_init_weights    = True

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

arg_parser.add_argument('--mf_steps', type = str, required = False)

arg_parser.add_argument('--data_samples', type = str, required = False)

arg_parser.add_argument('--learn_subset', type = str, required = False)

arg_parser.add_argument('--is_uniform', type = str, required = False)

arg_parser.add_argument('--resample', type = str, required = False)

arg_parser.add_argument('--momentum', type = str, required = False)

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

if FLAGS.learn_subset != None:
    
   num_learn = int(FLAGS.learn_subset)
   
   print("Will train on only %d randomly selected images"%num_learn)
   
   train_inds   = np.random.choice(range(N_train), 
                                   num_learn,
                                   replace=False)
   
   train_images = train_images[train_inds,:]
   
   N_train      = num_learn
   
   assert batch_size <= num_learn
   
else:
    
   num_learn = None
   
if FLAGS.momentum != None:
    
   momentum = float(FLAGS.momentum)
   
   if momentum != 0.0:
       
      use_momentum  = True
      
   else:
       
      use_momentum  = False
      momentum      = 0.0
      
else:
    
   use_momentum = False
   momentum     = 0.0
   
if test_mode:
    
   ## currently test_mode carries out 
   ## tests on gradient computation only
   ## to compare implicit and explicit 
   ## implementations (without using momentum term)  
   
   use_momentum = False
   
if algorithm == "CD1":
    
   assert FLAGS.num_steps != None 
    
   num_cd_steps = int(FLAGS.num_steps)
   
   specs = (str(learning_rate),
            str(momentum),
            batch_size,
            num_steps,
            datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))

   exp_tag = "LR%sM%sBS%dNST%d_%s"%specs
   
   num_samples = None 
   
   data_samples    = None
   
   resample    = None
   
   is_uniform  = False
   
if algorithm   == "CSS":
   
   assert FLAGS.num_samples  != None
   
   assert FLAGS.data_samples != None
    
   num_cd_steps   = None
   
   num_samples    = int(FLAGS.num_samples)
   
   data_samples   = int(FLAGS.data_samples)
   
   if num_samples ==0:
   
      specs = (str(learning_rate),
               str(momentum),
               batch_size,
               num_samples,
               data_sampples,
               datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))
   
      exp_tag = "LR%sM%sBS%dNS%dDATA%d_%s"%specs
      
      resample = None
      
      mf_steps = 0
      
   is_uniform = False
      
   if num_samples > 0:
       
      if FLAGS.is_uniform != None:
          
         is_uniform = bool(int(FLAGS.is_uniform))
       
      if FLAGS.mf_steps != None:
          
         mf_steps = int(FLAGS.mf_steps)
         
      else:
          
         mf_steps = 0
       
      assert FLAGS.resample != None
       
      resample = int(FLAGS.resample)
       
      specs = (str(learning_rate),
               str(momentum),
               batch_size,
               num_samples,
               resample,
               data_samples,
               mf_steps,
               datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))
   
      exp_tag = "LR%sM%sBS%dNS%dRS%dDATA%dMF%d_%s"%specs
      
      resample = bool(resample)
      
report_p_tilda = False # uses pseudo likelihood by default
   
if (num_learn < 50) and (data_samples ==0) and (resample != 1):
   
   print("Will report p_tilda for this experiment")
   report_p_tilda = True
      
else:
       
   print("Will report pseudo likelihoods for this experiment")
   
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
                      num_samples     = num_samples,
                      num_cd_steps    = num_cd_steps,
                      data_samples    = data_samples,
                      unique_samples  = resample,
                      is_uniform      = is_uniform,
                      mf_steps        = mf_steps,
                      use_momentum    = use_momentum,
                      report_p_tilda  = report_p_tilda,
                      test_mode       = test_mode)
                      
if test_add_complements:
    
   print("Testing mode")
   test_function = bm.test_add_complementary_term()
   
   t0 = timeit.default_timer()
   
   test_function()
   
   t1 = timeit.default_timer()
   
   print("Computation took --- %f minutes"%((t1 - t0)/60.0))
   
   sys.exit()
   
elif test_comp_energies:
    
   print("Testing mode")
   minibatch_inds = range(batch_size)
   
   sampled_indices = bm.select_data(minibatch_inds)
   
   test_function = bm.test_compute_energy()
   
   if data_samples < N_train:
   
      print("Need to compute energies of %d terms"%(batch_size*data_samples))
   
      assert len(sampled_indices) == batch_size*data_samples
      
   if data_samples == N_train:
       
      print("Need to compute energies of %d terms"%(data_samples))
   
      assert len(sampled_indices) == data_samples
   
   t0 = timeit.default_timer()
   
   approx_out = test_function(sampled_indices)
   
   t1 = timeit.default_timer()
   
   print("Computation took --- %f minutes"%((t1 - t0)/60.0))
   
   sys.exit()
   
elif test_mode:
    
   print("Testing mode")
   cd_sampling, optimize = bm.add_graph()
   
else:
    
   print("Training mode")
   cd_sampling, optimize = bm.add_graph()
   
   exp_path = os.path.join(dir_name,exp_tag)
        
   os.makedirs(exp_path)
   
   if FLAGS.learn_subset != None:
    
      np.savetxt(os.path.join(exp_path,"LEARNT_INSTANCES.dat"), train_inds)
      
   if save_images:
      print("Saving training images")
      np.savetxt(os.path.join(exp_path,"TRAIN_IMAGES.dat"), train_images)
   
   if save_init_weights:
    
      bm.save_model_params(os.path.join(exp_path,"INIT_PARAMS.model"))
      print("saved initial weights of fully visible Boltzmann Machine")
      
start_time = timeit.default_timer()

epoch_time0 = start_time

lowest_cost = np.inf

for epoch_index in range(num_epochs):
    
    permuted_inds = np.random.permutation(N_train)
    
    # put different learning_rate rules (per epoch) for now here:
    
    lrate_epoch = (1.0/(1+epoch_index/100.))*learning_rate/batch_size
      
    # lrate_epoch = (0.9**epoch_index)*learning_rate  # 0.99
    
    # lrate_epoch  = learning_rate
    
    if use_momentum:
    
       momentum_epoch = momentum
    
    print("Learning rate for epoch %d --- %f"%(epoch_index,lrate_epoch))
    
    avg_cost_val = []
    
    for i in range(num_iterations):
        
        iter_start_time = timeit.default_timer()
    
        minibatch_inds = permuted_inds[batch_size*i:batch_size*(i+1)]
        
        if algorithm =="CSS":
           
           if (num_samples > 0) and (is_uniform != True):
               
              if bool(mf_steps):
                 print("Updating MF parameters")
                 mf_t0 = timeit.default_timer()
              
                 bm.do_mf_updates(num_steps = mf_steps)
              
                 mf_t1 = timeit.default_timer()
                 print("4 steps of MF updates took --- %f"%
                 ((mf_t1 -mf_t0)/60.0))
              
              else:
                  
                 # sample with different probabilities 
                 reinit_mf = bm.np_rand_gen.uniform(0,1,\
                 size = (bm.num_vars, bm.num_samples))
        
                 reinit_mf = np.asarray(reinit_mf, 
                                        dtype = theano.config.floatX)
              
                 bm.mf_params.set_value(reinit_mf)
              
           opt_t0 = timeit.default_timer()
           ###
           if data_samples > 0:
               
              sampled_indices = bm.select_data(minibatch_inds)
              
              sampling_var = sampled_indices
                                               
           if data_samples ==0 and is_uniform:
               
              if resample > 0:
                 
                 is_samples = bm.np_rand_gen.binomial(n=1,p=0.5, 
                 size = (bm.num_samples*batch_size, bm.num_vars))
                 
              else:
               
                 is_samples = bm.np_rand_gen.binomial(n=1,p=0.5, 
                 size = (bm.num_samples, bm.num_vars))
              
              sampling_var = np.asarray(is_samples, 
                                        dtype = theano.config.floatX)
              
              if test_mode:
                      
                 t0 = timeit.default_timer()
                 
                 approx_cost, p_tilda = optimize(sampling_var, 
                                                 list(minibatch_inds),
                                                 lrate_epoch)
                                                  
                 t1 = timeit.default_timer()  
                 print("Gradient computation with implementation 1 took"+\
                 " --- %f minutes"%((t1 - t0)/60.0))
                 
                 W_implicit = np.asarray(bm.W.get_value())
              
                 b_implicit = np.asarray(bm.b.get_value())
                 
                 t0 = timeit.default_timer()
                 bm.test_grad_computations(is_samples, list(minibatch_inds))
                 t1 = timeit.default_timer()
                 print("Gradient computation with implementation 2 took "+\
                 "--- %f minutes"%((t1 - t0)/60.0))
              
                 W_explicit = np.asarray(bm.W.get_value())
              
                 b_explicit = np.asarray(bm.b.get_value())
                 
                 print("Equivalence of W updates in two implementations:")
                 print((np.round(W_implicit,12) == np.round(W_explicit,12)).all())
                 print("Equivalence of b updates in two implementations:")
                 print((np.round(b_implicit,12) == np.round(b_explicit,12)).all())
                 sys.exit()
                 
           if use_momentum:
              
              approx_cost, p_tilda = optimize(sampling_var, 
                                              list(minibatch_inds),
                                              lrate_epoch,
                                              momentum_epoch)
              
           else:
                  
              approx_cost, p_tilda = optimize(sampling_var, 
                                              list(minibatch_inds),
                                              lrate_epoch)   
           ###
           opt_t1 = timeit.default_timer()
           print("Optimization step took --- %f minutes"%
           ((opt_t1 - opt_t0)/60.0))
            
        if algorithm =="CD1":
            
           mf_sample, cd_sample = cd_sampling(list(minibatch_inds))
           
           bm.x_gibbs.set_value(np.transpose(cd_sample)) 
           
           approx_cost, p_tilda = optimize(list(minibatch_inds),
                                           lrate_epoch)
        
        avg_cost_val.append(approx_cost)
        
        if save_best_params and (abs(approx_cost) < lowest_cost):
        
           bm.save_model_params(os.path.join(exp_path,"TRAINED_PARAMS.model")) 
           
           lowest_cost = abs(approx_cost) 
        
        if i % report_step ==0:
            
           print('Training epoch %d ---- Iter %d ---- cost value: %f'
           %(epoch_index, i, approx_cost))
           
           if report_p_tilda:
               
              print("p_tilda:")
              print(p_tilda)
           
        iter_end_time = timeit.default_timer()
        
        print('Training iteration took %f minutes'%
        ((iter_end_time - iter_start_time) / 60.))
        
    avg_cost_val = np.mean(avg_cost_val)
    
    losses.append(avg_cost_val)
        
    print('Training epoch %d ---- average cost value: %f'
    %(epoch_index, avg_cost_val))
    
    epoch_time = timeit.default_timer()
    
    print ('Training epoch took %f minutes'%((epoch_time - epoch_time0)/60.))
    
    if not save_best_params:
    
       bm.save_model_params(os.path.join(exp_path,"TRAINED_PARAMS.model"))
    
    epoch_time0 = epoch_time
    
training_time = (epoch_time0 - start_time)/60.0

print('Training took %f minutes'%training_time)
    
np.savetxt(os.path.join(exp_path,"TRAIN_LOSSES.dat"), losses)

np.savetxt(os.path.join(exp_path,"TRAINING_TIME.dat"), np.array([training_time]))

if algorithm == "CSS":

   if (FLAGS.learn_subset != None) and (num_learn > 0):

      for file_tag in ['INIT','TRAINED']:
          
          if file_tag == 'TRAINED':
              
             print("Testing trained model")
             
          if file_tag == 'INIT':
              
            print("Testing initialized model")

          command_string =("python test_bm.py "+\
          "--path_to_params %s/%s_PARAMS.model "+\
          "--num_samples 8 --num_chains %d --trained_subset 1 --num_steps 100 --use_mf_sampler 1")\
          %(exp_path,file_tag, num_learn)

          subprocess.call(command_string ,shell=True)


    
    
    
    
    
    
    

  









