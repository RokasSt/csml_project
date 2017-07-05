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


np_rand_gen = np.random.RandomState(1234)

num_is_samples = 10000

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

test_images      = np.round(mnist.test.images)

test_labels      = mnist.test.labels

train_images     = np.round(mnist.train.images)

train_labels     = mnist.train.labels

num_train_images = train_images.shape[0]

D                = train_images.shape[1]

assert D == 784

validate_images  = np.round(mnist.validation.images)

validate_labels  = mnist.validation.labels

arg_parser       = argparse.ArgumentParser()

arg_parser.add_argument('--path_to_params', type=str,required= True)

arg_parser.add_argument('--num_samples', type = str, required = True)

arg_parser.add_argument('--num_chains', type = str, required = True)

arg_parser.add_argument('--trained_subset', type = str, required = True)

arg_parser.add_argument('--num_steps', type = str, required = True)

arg_parser.add_argument('--sampler', type = str, required = True)

arg_parser.add_argument('--init_with_dataset', type = str, required = True)

arg_parser.add_argument('--num_burn_in', type = str, required = False)

FLAGS, _          = arg_parser.parse_known_args()

path_to_params    = FLAGS.path_to_params

num_samples       = int(FLAGS.num_samples)

num_chains        = int(FLAGS.num_chains)

trained_subset    = int(FLAGS.trained_subset)

num_steps         = int(FLAGS.num_steps)

sampler           = FLAGS.sampler

if (FLAGS.num_burn_in != None) and sampler =="GIBBS":
    
   num_burn_in = int(FLAGS.num_burn_in)
   
else:
    
   num_burn_in = 0
   
   
if "RH" in path_to_params:

   ind0 = path_to_params.find("RH")
   ind1 = path_to_params.find("LR")
   num_hidden = int(path_to_params[ind0+2:ind1])
   
else:
    
   num_hidden = 0

init_with_dataset = bool(int(FLAGS.init_with_dataset))

split_path        = os.path.split(path_to_params)

if bool(trained_subset):
    
   indices =np.loadtxt(os.path.join(split_path[0],"LEARNT_INSTANCES.dat"))
   
   indices = np.array(indices, dtype = np.int64)
   
   test_inputs = train_images[indices,:]
   
   if indices.size == 1 and (num_chains != 1):
       
      use_num_chains = 1
      
      test_inputs = np.reshape(test_inputs,[1,len(test_inputs)])
      
   elif indices.size != 1 and (num_chains ==1):
       
      select_inds = np.random.choice(len(indices), 
                                     num_chains, 
                                     replace=False)  
      
      test_inputs = test_inputs[select_inds,:]
      
      use_num_chains = num_chains
       
   else:
       
      len_inds = len(indices)
   
      if num_chains <= len_inds:
         
         use_num_chains = num_chains 
          
      else:
         
         use_num_chains  = len(indices)
         
   x_to_test_p = test_inputs
   
else:
    
   test_inputs = test_images
   
   use_num_chains = num_chains
   
   x_inds = np.random.choice(test_inputs.shape[0], 10, replace = False)

   x_to_test_p = test_inputs[x_inds, :]
   
filename = "SS%dCH%dST%d"%(num_samples, num_chains, num_steps)

if sampler == "MF":
    
   filename+="_MF"
   
elif sampler == "GIBBS":
    
   filename+="_GIBBS"
   
if "INIT" in split_path[1]:
    
   filename+="_init"
    
   save_to_path = os.path.join(split_path[0],filename+".jpeg")
   
else:
    
   save_to_path = os.path.join(split_path[0],filename+".jpeg")

bm = BoltzmannMachine(num_vars        = D, 
                      num_hidden      = num_hidden,
                      training        = False)
      
bm.load_model_params(full_path = path_to_params)

bm.test_relative_probability(inputs = x_to_test_p, trained= True)

rand_samples = bm.np_rand_gen.binomial(n=1,p=0.5, 
                                       size = (x_to_test_p.shape[0], 784))
              
rand_samples = np.asarray(rand_samples, dtype = theano.config.floatX)
   
bm.test_relative_probability(inputs = rand_samples, trained = False)
   
print("------------------------------------------------------------")
print("-------------- Computing p_tilda values --------------------")
print("------------------for training set--------------------------")
print("")
   
is_samples = np_rand_gen.binomial(n=1, p=0.5, size = (num_is_samples, D))
   
train_p_tilda, rand_p_tilda = bm.test_p_tilda(x_to_test_p, 
                                              is_samples,
                                              training = False)
   
print("p_tilda values for training inputs:")
print(train_p_tilda)
print("")
print("p_tilda values for 10 randomly chosen importance samples:")
print(rand_p_tilda)
print("")
   
print("-------------- Computing pseudo likelihood ------------------")
   
pseudo_cost = bm.test_pseudo_likelihood(x_to_test_p, num_steps= 784)
print("Stochastic approximation to pseudo likelihood ---- %f"%pseudo_cost)
   
start_time = timeit.default_timer()

## init_with_dataset overrides trained_subset option
if not init_with_dataset:
    
   test_inputs = None
#################### 
 
if sampler == "MF":

   bm.sample_from_mf_dist(num_chains   = use_num_chains, 
                          num_samples  = num_samples,
                          num_steps    = num_steps,
                          save_to_path = save_to_path,
                          test_inputs  = test_inputs,
                          save_mf_params = True)
    
    
elif sampler == "GIBBS":

   bm.sample_from_bm(num_chains   = use_num_chains, 
                     num_samples  = num_samples,
                     num_steps    = num_steps,
                     save_to_path = save_to_path,
                     num_burn_in  = num_burn_in,
                     test_inputs  = test_inputs)
        
                    
end_time = timeit.default_timer()
                   
print('Image generation took %f minutes'%((end_time - start_time)/60.))


                   

                   

                      
