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

np_rand_gen = np.random.RandomState(1234)

hyperparams = {}

hyperparams['N_train'] = train_images.shape[0]

hyperparams['D']       = train_images.shape[1]

assert  hyperparams['D'] == 784

hyperparams['use_gpu']           = False

hyperparams['report_step']       = 1

assert hyperparams['report_step'] != None

hyperparams['save_images']       = True

hyperparams['save_every_epoch']  = False

hyperparams['report_w_norm']     = True

hyperparams['save_init_weights'] = True

hyperparams['report_pseudo_cost']= True

hyperparams['learning_rate']     = 0.01

hyperparams['batch_size']        = 2

hyperparams['use_momentum']      = True

hyperparams['momentum']          = 0.95

hyperparams['num_hidden']        = 0

hyperparams['num_to_learn']      = 2

hyperparams['equal_per_classes'] = False

hyperparams['normal_init']       = True

hyperparams['xavier_init']       = False

hyperparams['zero_wii']          = True

hyperparams['pflip']             = 0.1

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

if hyperparams['equal_per_classes']:
    
   found_all_files = True
   
   for cl_file in class_files:
   
       if not os.path.exists(cl_file):
           
          found_all_files = False
          
          break
          
   if not found_all_files:
      print("Will save class-specific images to individual files")
      utils.save_images_per_class(images = train_images, 
                            labels = train_labels, 
                            root_dir = "./")
                            
if hyperparams['num_to_learn'] < hyperparams['N_train']:
   
   print("Will train on only %d randomly selected images"%
   hyperparams['num_to_learn'])
   
   if hyperparams['equal_per_classes']:
   
      train_images = utils.select_subset(class_files, 
                                         n = hyperparams['num_to_learn']//10,
                                         D = hyperparams['D'])
                                   
   else:
       
      train_inds = np.random.choice(range(hyperparams['N_train']), 
                                    hyperparams['num_to_learn'], 
                                    replace=False)
                                    
      train_images = train_images[train_inds,:]
   
   hyperparams['N_train'] = hyperparams['num_to_learn']
   
   assert hyperparams['batch_size'] <= hyperparams['num_to_learn']

hyperparams['num_iters'] = hyperparams['N_train'] // hyperparams['batch_size']

alpha=0.995

if alpha != None:
   
   is_probs = (1-alpha)*0.5*np.ones([1,hyperparams['D']])+\
    alpha*np.mean(train_images,0)
    
else:
    
   is_probs  = []

####### gobal parameters end
if hyperparams['use_gpu']:    
   print("Will attempt to use GPU")
   os.environ['THEANO_FLAGS'] = 'device=cuda'
   
import theano
import theano.tensor as T
from   model_classes import BoltzmannMachine

if is_probs != []:

   is_probs = np.asarray(is_probs, dtype = theano.config.floatX)

exp1 ={'num_epochs'    : 200,
       'batch_size'    : hyperparams['batch_size'],
       'learning_rate' : hyperparams['learning_rate'],
       'algorithm'     : 'CSS',
       'algorithm_dict':
           {
            'data_samples'  : False,
            'num_samples'   : 100,
            'resample'      : False,  
            'use_is'        : True,
            'is_probs'      : is_probs,
            'mf_steps'      : 0,
           
           },
       
       'use_momentum'  : hyperparams['use_momentum'],
       'momentum'      : hyperparams['momentum'],
       'num_hidden'    : hyperparams['num_hidden'],
       'use_gpu'       : hyperparams['use_gpu'],
       'report_p_tilda': True
       }
         
exp2 ={'num_epochs'    : 200,
       'batch_size'    : hyperparams['batch_size'],
       'learning_rate' : hyperparams['learning_rate'],
       'num_hidden'    : hyperparams['num_hidden'],
       'algorithm'     : 'CD',
       'algorithm_dict':
           {
           'num_cd_steps':1,
           },
       'use_gpu'       : hyperparams['use_gpu'],
       'report_p_tilda': False
       }

experiments = {'exp1':exp1,'exp2':exp2 }

exp1_string = experiments['exp1']['algorithm']

exp2_string = experiments['exp2']['algorithm']

dir_name ="logs_%s_vs_%s"%(exp1_string, exp2_string)

curr_time = datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" )

save_to_path = os.path.join(dir_name, curr_time)

os.makedirs(save_to_path)

all_params = {'exp1':exp1,
              'exp2':exp2, 
              'GLOBAL': hyperparams}

with open(os.path.join(save_to_path,'PARAMETERS.json'), 'w') as json_file:
    
     json.dump(all_params, json_file)
     
if hyperparams['save_images']:
    
   print("Saving training images")
   np.savetxt(os.path.join(save_to_path,"TRAIN_IMAGES.dat"), train_images)
   if not hyperparams['equal_per_classes']:
      np.savetxt(os.path.join(save_to_path,"LEARNT_INSTANCES.dat"), train_inds)
   
# TODO: W0, b0, bhid0 init

if hyperparams['num_hidden'] > 0 and hyperparams['xavier_init']:

   W0 = np_rand_gen.uniform(
      -4*np.sqrt(6.0/(hyperparams['D'] + hyperparams['num_hidden'])),\
       4*np.sqrt(6.0 /(hyperparams['D'] + hyperparams['num_hidden'])), 
      size = (hyperparams['D'], hyperparams['num_hidden'])
     )
     
elif hyperparams['num_hidden'] ==0 and hyperparams['xavier_init']:

   W0 = np_rand_gen.uniform(
      -4*np.sqrt(6.0/(hyperparams['D'] + hyperparams['D'])),\
       4*np.sqrt(6.0 /(hyperparams['D'] + hyperparams['D'])), 
      size = (hyperparams['D'], hyperparams['D'])
     )
   
elif hyperparams['num_hidden'] > 0 and hyperparams['normal_init']:
    
   W0 = 0.00000001*\
     np_rand_gen.normal(size = (hyperparams['D'], hyperparams['num_hidden']))
   
elif hyperparams['num_hidden'] == 0 and hyperparams['normal_init']:
    
   W0 = 0.00000001*\
   np_rand_gen.normal(size = (hyperparams['D'], hyperparams['D']))
                    
W0 = np.asarray(W0, dtype = theano.config.floatX) 

if hyperparams['zero_wii']:
   
   W0 = W0 - np.diag(np.diag(W0)) 
   
W0_stored = np.copy(W0)
   
b0        = np.zeros(hyperparams['D'], dtype = theano.config.floatX)

b0_stored = np.copy(b0)

if hyperparams['num_hidden'] > 0:
    
   bhid0 =  np.zeros(hyperparams['num_hidden'], 
                     dtype = theano.config.floatX)
                     
   bhid0_stored = np.copy(bhid0)
   
else:
    
   bhid0 = None
   
if hyperparams['save_init_weights']:
    
   np.savetxt(os.path.join(save_to_path,"W0.dat"), W0)
   
   np.savetxt(os.path.join(save_to_path,"b.dat"), b0)
   
   if hyperparams['num_hidden'] > 0:
       
      np.savetxt(os.path.join(save_to_path,"bhid.dat"), bhid0)
      
      print("saved initial weights for Restricted Boltzmann Machine")
      
   elif hyperparams['num_hidden'] == 0:
       
      print("saved initial weights for Fully Visible Boltzmann Machine")
      
avg_errors = {}

which_noise_pixels = utils.select_noisy_pixels(pflip = hyperparams['pflip'], 
                                               D = hyperparams['D'], 
                                               N = hyperparams['num_to_learn'])
                                               
noisy_images = np.copy(train_images)

noisy_images[which_noise_pixels] = 1- noisy_images[which_noise_pixels]

which_missing_pixels = utils.select_missing_pixels(gamma = 0.5, 
                                                   D= hyperparams['D'], 
                                                   N= hyperparams['num_to_learn'])
                                                   
blocked_images    = np.copy(train_images)
blocked_images[which_missing_pixels]    = -1

reconst_noise_images          = {}

reconst_missing_pixels_images = {}

for exp_tag in experiments.keys():
    
    param_dict = experiments[exp_tag]
    
    print("Algorithm : %s"%param_dict['algorithm'])
    
    avg_errors[param_dict['algorithm']] = {}
    
    bm = BoltzmannMachine(num_vars        = hyperparams['D'], 
                          num_hidden      = hyperparams['num_hidden'],
                          training_inputs = train_images,
                          algorithm       = param_dict['algorithm'],
                          algorithm_dict  = param_dict['algorithm_dict'],
                          batch_size      = hyperparams['batch_size'],
                          use_momentum    = hyperparams['use_momentum'],
                          W0              = W0, 
                          b0              = b0, 
                          bhid0           = bhid0,
                          report_p_tilda  = param_dict['report_p_tilda'])
    
    bm.add_graph()
   
    exp_path = os.path.join(save_to_path, param_dict['algorithm'])
        
    os.makedirs(exp_path)
    
    p_tilda_all, losses, train_time, w_norms = \
    bm.train_model(num_epochs = param_dict['num_epochs'], 
                   learning_rate = hyperparams['learning_rate'], 
                   momentum = hyperparams['momentum'], 
                   num_iters = hyperparams['num_iters'],
                   report_pseudo_cost = hyperparams['report_pseudo_cost'],
                   save_every_epoch   = hyperparams['save_every_epoch'],
                   report_step = hyperparams['report_step'],
                   report_p_tilda  = param_dict['report_p_tilda'],
                   report_w_norm   = hyperparams['report_w_norm'],
                   exp_path  = exp_path)
                   
    if hyperparams['report_pseudo_cost']:
    
       np.savetxt(os.path.join(exp_path, "TRAIN_PSEUDO_LOSSES.dat"), losses)
       
    if hyperparams['report_w_norm']:
        
       np.savetxt(os.path.join(exp_path, "W_NORMS.dat"), w_norms)

    np.savetxt(os.path.join(exp_path, "TRAINING_TIME.dat"), 
               np.array([train_time]))

    if param_dict['report_p_tilda']:
    
       np.savetxt(os.path.join(exp_path,"TRAIN_P_TILDA.dat"), p_tilda_all)
       
    W0 = W0_stored
    b0 = b0_stored
    if hyperparams['num_hidden']:
       bhid0 = bhid0_stored
       
    #### check initial parameters:
    assert (W0_stored == W0).all()  == True
    assert (b0_stored == b0).all() == True
    if hyperparams['num_hidden']:
       assert (bhid0_stored == bhid0).all() == True
       
    ###### reconstruction of missing pixels
    filename = "RECONST_MISSING"

    save_plots_to = os.path.join(exp_path, filename+".jpeg")
    
    images_to_reconst = np.copy(train_images)
    
    images_to_reconst[which_missing_pixels] =  1
    
    images_to_reconst = \
    bm.reconstruct_missing_pixels(num_iters    = 1, 
                                  recon_images = images_to_reconst, 
                                  which_pixels = which_missing_pixels)
                                  
    reconst_missing_pixels_images[param_dict['algorithm']] = images_to_reconst
                                                                            
    recon_errors= utils.plot_reconstructions(train_images,
                                             blocked_images,
                                             images_to_reconst,
                                             save_plots_to)
                                         
    np.savetxt(os.path.join(exp_path,"%s_ERRORS.dat"%filename), recon_errors)

    avg_errors[param_dict['algorithm']]['MISSING'] = np.mean(recon_errors)
    ############### reconstruction of noisy data
    filename = "RECONST_NOISY"

    save_plots_to = os.path.join(exp_path, filename+".jpeg")
    
    images_to_reconst= np.copy(noisy_images)

    images_to_reconst = bm.reconstruct_noisy_pixels(num_iters   = 1, 
                                                    correct_images = train_images,
                                                    recon_images = images_to_reconst, 
                                                    noisy_images = noisy_images,
                                                    pflip = hyperparams['pflip'])
                                                    
    reconst_noise_images[param_dict['algorithm']] = images_to_reconst
                                                
    recon_errors= utils.plot_reconstructions(train_images,
                                             noisy_images,
                                             images_to_reconst,
                                             save_plots_to)
                                         
    np.savetxt(os.path.join(exp_path, "%s_ERRORS.dat"%filename),
               recon_errors)

    avg_errors[param_dict['algorithm']]['NOISY'] = np.mean(recon_errors)

print("Average reconstruction errors:")

print(avg_errors)

utils.compare_reconstructions(train_images,
                              noisy_images,
                              reconst_noise_images,
                              os.path.join(save_to_path,"COMPARE_NOISE.jpeg"))
                              
utils.compare_reconstructions(train_images,
                              blocked_images,
                              reconst_missing_pixels_images,
                              os.path.join(save_to_path,"COMPARE_MISSING.jpeg"))

with open(os.path.join(save_to_path,'MEAN_ERRORS.json'), 'w') as json_file:
    
     json.dump(avg_errors, json_file)

       
       
    


    
    
