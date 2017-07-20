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

test_grad_mode       = False

save_images          = True

save_best_params     = True

save_end_params      = True

test_add_complements = False

test_comp_energies   = False

save_init_weights    = True

save_every_epoch     = False

report_pseudo_cost   = True

report_w_norm        = True

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

arg_parser.add_argument('--resample', type = str, required = False)

arg_parser.add_argument('--momentum', type = str, required = False)

arg_parser.add_argument('--num_hidden', type = str, required = False)

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
   
   if num_learn > 0 :
   
      print("Will train on only %d randomly selected images"%num_learn)
   
      train_inds   = np.random.choice(range(N_train), 
                                      num_learn,
                                      replace=False)
   
      train_images = train_images[train_inds,:]
   
      N_train      = num_learn
   
      assert batch_size <= num_learn
      
   else:
       
      num_learn = None
         
else:
    
   num_learn = None
##################################   
num_iters = N_train // batch_size

losses = []
##################################   
if FLAGS.num_hidden != None:

   num_hidden = int(FLAGS.num_hidden)
   
else:
    
   num_hidden = 0
   
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
   
if test_grad_mode:
    
   ## currently test_mode carries out 
   ## tests on gradient computation only
   ## to compare implicit and explicit 
   ## implementations (without using momentum term)  
   
   use_momentum = False
   
algorithm_dict = {}
   
if algorithm == "CD" or algorithm == "PCD":
    
   assert FLAGS.num_steps != None 
    
   num_cd_steps = int(FLAGS.num_steps)
   
   algorithm_dict['num_cd_steps'] = num_cd_steps
   
   specs = (num_hidden,
            str(learning_rate),
            str(momentum),
            batch_size,
            num_cd_steps,
            datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))

   exp_tag = "RH%dLR%sM%sBS%dNST%d_%s"%specs
   
   algorithm_dict['num_samples']  = None 
   
   algorithm_dict['data_samples'] = None
   
   algorithm_dict['resample']     = None
   
   algorithm_dict['mf_steps']     = None
   
report_p_tilda = False
   
if algorithm   == "CSS":
   
   assert FLAGS.num_samples  != None
   
   assert FLAGS.data_samples != None
    
   algorithm_dict['num_cd_steps'] = None
   
   algorithm_dict['num_samples']    = int(FLAGS.num_samples)
   
   algorithm_dict['data_samples']   = int(FLAGS.data_samples)
   
   algorithm_dict['alpha']  = 0.995
   
   algorithm_dict['mf_steps'] = 0
   
   if algorithm_dict['num_samples'] == 0:
   
      specs = (num_hidden,
               str(learning_rate),
               str(momentum),
               batch_size,
               algorithm_dict['num_samples'] ,
               algorithm_dict['data_samples'],
               datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))
   
      exp_tag = "RH%dLR%sM%sBS%dNS%dDATA%d_%s"%specs
      
      algorithm_dict['resample'] = None
      
   if algorithm_dict['num_samples'] > 0:
      
      FLAGS.mf_steps != None:
          
         algorithm_dict['mf_steps'] = int(FLAGS.mf_steps)
      
      assert FLAGS.resample != None
       
      algorithm_dict['resample'] = int(FLAGS.resample)
       
      specs = (num_hidden,
               str(learning_rate),
               str(momentum),
               batch_size,
               algorithm_dict['num_samples'],
               algorithm_dict['resample'],
               algorithm_dict['data_samples'],
               algorithm_dict['mf_steps'],
               datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" ))
   
      exp_tag = "RH%dLR%sM%sBS%dNS%dRS%dDATA%dMF%d_%s"%specs
      
      algorithm_dict['resample'] = bool(algorithm_dict['resample'])
      
   if (num_learn < 50) and (algorithm_dict['data_samples'] ==0) \
   and (algorithm_dict['resample'] != True):
   
      print("Will report p_tilda for this experiment")
      report_p_tilda = True
      
      p_tilda_all = np.zeros([num_epochs*num_iters//report_step,batch_size])
      
      p_t_i = 0
      
if bool(use_gpu):    
   print("Will attempt to use GPU")
   os.environ['THEANO_FLAGS'] = 'device=cuda'
   
import theano
import theano.tensor as T
from   model_classes import BoltzmannMachine

bm = BoltzmannMachine(num_vars        = input_dim, 
                      num_hidden      = num_hidden,
                      training_inputs = train_images,
                      algorithm       = algorithm,
                      algorithm_dict  = algorithm_dict,
                      batch_size      = batch_size,
                      use_momentum    = use_momentum,
                      report_p_tilda  = report_p_tilda,
                      test_mode       = test_grad_mode)
                      
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
   
elif test_grad_mode:
    
   print("Testing mode")
   
else:
    
   print("Training mode")
   
   bm.add_graph()
   
   exp_path = os.path.join(dir_name,exp_tag)
        
   os.makedirs(exp_path)
   
   if num_learn != None:
    
      np.savetxt(os.path.join(exp_path,"LEARNT_INSTANCES.dat"), train_inds)
      
   if save_images:
      print("Saving training images")
      np.savetxt(os.path.join(exp_path,"TRAIN_IMAGES.dat"), train_images)
   
   if save_init_weights:
    
      bm.save_model_params(os.path.join(exp_path,"INIT_PARAMS.model"))
      print("saved initial weights of fully visible Boltzmann Machine")
      
   p_tilda_all, losses, train_time, w_norms = \
    bm.train_model(num_epochs         = num_epochs, 
                   learning_rate      = learning_rate, 
                   momentum           = momentum, 
                   num_iters          = num_iters,
                   report_pseudo_cost = report_pseudo_cost,
                   save_every_epoch   = save_every_epoch,
                   report_step        = report_step,
                   report_p_tilda     = report_p_tilda,
                   report_w_norm      = report_w_norm,
                   exp_path           = exp_path,
                   test_gradients     = test_grad_mode)
                   
   if report_pseudo_cost:
    
      np.savetxt(os.path.join(exp_path, "TRAIN_PSEUDO_LOSSES.dat"), losses)
   
   if report_w_norm:
      print("Saving W norms")
      np.savetxt(os.path.join(exp_path, "W_NORMS.dat"), w_norms)

   np.savetxt(os.path.join(exp_path,"TRAINING_TIME.dat"), 
              np.array([train_time]))

   if report_p_tilda:
    
      np.savetxt(os.path.join(exp_path,"TRAIN_P_TILDA.dat"), p_tilda_all)


    
    
    
    
    
    
    

  









