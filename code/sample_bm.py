""" 
Author: Rokas Stanislovas
MSc Project: Likelihood Approximations
for Energy-Based Models
MSc Computational Statistics and 
Machine Learning

Script to generate samples from trained Boltzmann Machines.
"""

import numpy as np
import argparse
import shutil
import os
import sys
import json
import theano
import theano.tensor as T
import datetime
import utils
import argparse
import timeit
import os
from   model_classes import BoltzmannMachine

np_rand_gen = np.random.RandomState(1234)

num_is_samples = 100

N_train        = 60000

arg_parser     = argparse.ArgumentParser()

arg_parser.add_argument('--path_to_params', type=str,required= True)

arg_parser.add_argument('--num_samples', type = str, required = True)

arg_parser.add_argument('--num_chains', type = str, required = True)

arg_parser.add_argument('--trained_subset', type = str, required = True)

arg_parser.add_argument('--num_steps', type = str, required = True)

arg_parser.add_argument('--init_with_dataset', type = str, required = True)

arg_parser.add_argument('--num_burn_in', type = str, required = False)

FLAGS, _          = arg_parser.parse_known_args()

path_to_params    = FLAGS.path_to_params

num_samples       = int(FLAGS.num_samples)

num_chains        = int(FLAGS.num_chains)

trained_subset    = int(FLAGS.trained_subset)

num_steps         = int(FLAGS.num_steps)

print("Importing data ...")
all_images, all_labels = utils.get_data_arrays()

test_images            = all_images[N_train:,:]

test_labels            = all_labels[N_train:,:]

train_images           = all_images[0:N_train,:]

train_labels           = all_labels[0:N_train,:]

num_train_images       = train_images.shape[0]

D                      = train_images.shape[1]

assert D == 784

if (FLAGS.num_burn_in != None):
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
    
   path_to_indices = os.path.join(split_path[0],"LEARNT_INSTANCES.dat")
    
   if os.path.exists(path_to_indices): 
    
      indices =np.loadtxt(os.path.join(split_path[0],"LEARNT_INSTANCES.dat"))
   
      indices = np.array(indices, dtype = np.int64)
   
      test_inputs = train_images[indices,:]
      
      indices_size = indices.size
      
   else:
       
      print("LEARNT_INSTANCES.dat was not found in %s"%split_path[0])
      split_path1 = os.path.split(split_path[0])
      
      path_to_train_imgs = os.path.join(split_path1[0], "TRAIN_IMAGES.dat")
      
      test_inputs = np.loadtxt(path_to_train_imgs)
      
      indices_size= test_inputs.shape[0]
      
   if indices_size == 1 and (num_chains != 1):
       
      use_num_chains = 1
      
      test_inputs = np.reshape(test_inputs,[1,len(test_inputs)])
      
   elif indices_size ==1 and (num_chains ==1):
       
      use_num_chains = 1
      
      test_inputs = np.reshape(test_inputs,[1,len(test_inputs)])
      
   elif indices_size != 1 and (num_chains ==1):
       
      select_inds = np.random.choice(len(indices), 
                                     num_chains, 
                                     replace=False)  
      
      test_inputs = test_inputs[select_inds,:]
      
      use_num_chains = num_chains
       
   else:
       
      if num_chains <= indices_size:
         
         use_num_chains = num_chains 
          
      else:
         
         use_num_chains  = indices_size
         
   x_to_test_p = test_inputs
   
else:
    
   test_inputs = test_images
   
   use_num_chains = num_chains
   
   x_inds = np.random.choice(test_inputs.shape[0], 10, replace = False)

   x_to_test_p = test_inputs[x_inds, :]
   
filename = "SS%dCH%dST%dNB%d"%(num_samples, 
                               num_chains, 
                               num_steps,
                               num_burn_in)

filename+="_GIBBS"
   
if "END" in path_to_params:
   
   filename+="_END"
   
else:
    
   filename+="_INTER"
   
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
   
test_p_tilda, rand_p_tilda = bm.test_p_tilda(x_to_test_p, 
                                              is_samples,
                                              training = False)
   
if bool(trained_subset):
   print("p_tilda values for 10 training inputs:")
else:
   print("p_tilda values for 10 test inputs:")
print(test_p_tilda)
print("")
print("p_tilda values for 10 randomly chosen importance samples:")
print(rand_p_tilda)
print("")

np.savetxt(os.path.join(split_path[0], "TEST_P_TILDA.dat"), test_p_tilda)
np.savetxt(os.path.join(split_path[0], "RAND_P_TILDA.dat"), rand_p_tilda)
   
print("-------------- Computing pseudo likelihood ------------------")
   
pseudo_cost = bm.test_pseudo_likelihood(x_to_test_p, num_steps= 784)
print("Stochastic approximation to pseudo likelihood ---- %f"%pseudo_cost)
   
## init_with_dataset overrides trained_subset option
if not init_with_dataset:
    
   test_inputs = None
#################### 
start_time = timeit.default_timer()

bm.sample_from_bm(num_chains   = use_num_chains, 
                  num_samples  = num_samples,
                  num_steps    = num_steps,
                  save_to_path = save_to_path,
                  num_burn_in  = num_burn_in,
                  test_inputs  = test_inputs)
         
end_time = timeit.default_timer()
                   
print('Image generation took %f minutes'%((end_time - start_time)/60.))


                   

                   

                      
