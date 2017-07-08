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

test_images      = np.round(mnist.test.images)

test_labels      = mnist.test.labels

train_images     = np.round(mnist.train.images)

train_labels     = mnist.train.labels

validate_images  = np.round(mnist.validation.images)

validate_labels  = mnist.validation.labels

####### global parameters and settings ##

N_train          = train_images.shape[0]

D                = train_images.shape[1]

assert D == 784

use_gpu           = False

report_step       = 1

save_images       = True

save_inter_params = True

report_on_weights = True

save_init_weights = True

report_pseudo_cost= False

report_p_tilda    = True

learning_rate     = 0.05

batch_size        = 10

use_momentum      = False

momentum          = 0.0

num_hidden        = 0

num_to_learn      = 10

equal_per_classes = True

class_files = ["CLASS0.dat",
               "CLASS1.dat",
               "CLASS2.dat",
               "CLASS3.dat",
               "CLASS4.dat",
               "CLASS5.dat",
               "CLASS6.dat",
               "CLASS7.dat",
               "CLASS8.dat",
               "CLASS9.dat"]

if equal_per_classes:
    
   found_all_files = True
   
   for cl_file in class_files:
   
       if not os.path.exists(cl_file):
           
          found_all_files = False
          
          break
          l 
   if not found_all_files:
      print("Will save class-specific images to individual files")
      utils.save_images_per_class(images = train_images, 
                            labels = train_labels, 
                            root_dir = "./")
                            
if num_to_learn < N_train:
   
   print("Will train on only %d randomly selected images"%num_to_learn)
   
   if equal_per_classes:
   
      train_images = utils.select_subset(class_files, 
                                         n = num_to_learn//10,
                                         D = D)
                                   
   else:
       
      train_inds = np.random.choice(range(N_train), 
                                    num_to_learn, 
                                    replace=False)
                                    
      np.savetxt(os.path.join(exp_path,"LEARNT_INSTANCES.dat"), train_inds)
   
      train_images = train_images[train_inds,:]
   
   N_train      = num_to_learn
   
   assert batch_size <= num_to_learn
         
num_iters = N_train // batch_size

####### gobal parameters end
if use_gpu:    
   print("Will attempt to use GPU")
   os.environ['THEANO_FLAGS'] = 'device=cuda'
   
import theano
import theano.tensor as T
from   model_classes import BoltzmannMachine
      
   
if save_init_weights:
    
   save_model_params(os.path.join(exp_path,"INIT_PARAMS.model"))
   print("saved initial weights of fully visible Boltzmann Machine")
      
exp1 ={'num_epochs'    : 1500,
       'batch_size'    : batch_size,
       'learning_rate' : learning_rate,
       'algorithm'     : 'CSS',
       'algorithm_dict':
           {
            'data_samples'  : False,
            'num_samples'   : 100,
            'resample'      : False,  
            'is_uniform'    : True,
            'mf_steps'      : 0,
           
           },
       
       'use_momentum'  : use_momentum,
       'momentum'      : momentum,
       'num_hidden'    : num_hidden,
       'use_gpu'       : use_gpu,
       'report_p_tilda': True
       }
         
exp2 ={'num_epochs'    : 15,
       'batch_size'    : batch_size,
       'learning_rate' : learning_rate,
       'num_hidden'    : num_hidden,
       'algorithm'     : 'CD',
       'algorithm_dict':
           {
           'num_cd_steps':1,
           },
       'use_gpu'       : use_gpu
       }

experiments = {'exp1':exp1,'exp2':exp2}

with open('%s.json', 'w') as fp:
     json.dump(data, fp)

dir_name ="logs_%s_vs_%s"%experiments['exp1']['algorithm']

curr_time = datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" )

save_to_path = os.path.join(dir_name, curr_time)

os.makedirs(exp_path)

with open(os.path.join(save_to_path,'HYPERPARAMS.json'), 'w') as json_file:
    
     json.dump(experiments, json_file)
     
if save_images:
    
   print("Saving training images")
   np.savetxt(os.path.join(dir_name,"TRAIN_IMAGES.dat"), train_images)
   
# TODO: W0, b0, bhid0 init
  
for exp_tag in experiments.keys():
    
    param_dict = experiments[exp_tag]
    
    print("Algorithm : %s"%param_dict['algorithm'])
    
    bm = BoltzmannMachine(num_vars        = D, 
                          num_hidden      = num_hidden,
                          training_inputs = train_images,
                          algorithm       = param_dict['algorithm'],
                          algorithm_dict  = param_dict['algorithm_dict'],
                          batch_size      = batch_size,
                          use_momentum    = use_momentum,
                          W0              = W_init, 
                          b0              = b_init, 
                          bhid0           = bhid_init,
                          report_p_tilda  = report_p_tilda)
    
    bm.add_graph()
   
    exp_path = os.path.join(save_to_path, param_dict['algorithm'])
        
    os.makedirs(exp_path)
    
    p_tilda_all, losses, train_time = \
    bm.train_model(num_epochs = param_dict['num_epochs'], 
                   learning_rate = learning_rate, 
                   momentum = momentum, 
                   num_iters = num_iters,
                   report_pseudo_cost = report_pseudo_cost,
                   save_inter_params  = save_inter_params,
                   report_step = report_step,
                   report_p_tilda  = report_p_tilda,
                   exp_path  = exp_path)
    
    np.savetxt(os.path.join(exp_path,"TRAIN_PSEUDO_LOSSES.dat"), losses)

    np.savetxt(os.path.join(exp_path,"TRAINING_TIME.dat"), np.array([train_time]))

    if report_p_tilda:
    
       np.savetxt(os.path.join(exp_path,"TRAIN_P_TILDA.dat"), p_tilda_all)


    
    
