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
from matplotlib import pyplot as plt

print("Importing data:")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

all_test_images      = np.round(mnist.test.images)

all_test_labels      = mnist.test.labels

all_train_images     = np.round(mnist.train.images)

all_train_labels     = mnist.train.labels

all_validate_images  = np.round(mnist.validation.images)

all_validate_labels  = mnist.validation.labels

####### global parameters and settings ##

np_rand_gen = np.random.RandomState(1234)

params = {}

params['num_runs'] = 20

params['N_train'] = all_train_images.shape[0]

params['D']       = all_train_images.shape[1]

assert params['D'] == 784

params['use_gpu']           = False

params['num_epochs']        = 1500

params['report_step']       = 1

assert params['report_step'] != None

params['save_every_epoch']  = False

params['report_w_norm']     = True

params['save_init_weights'] = True

params['report_pseudo_cost']= True

params['learning_rate']     = 0.01

params['batch_size']        = 10

params['use_momentum']      = True

params['momentum']          = 0.95

params['num_hidden']        = 0

params['num_to_learn']      = 10

params['equal_per_classes'] = False

params['normal_init']       = True

params['xavier_init']       = False

params['zero_wii']          = False

params['learn_biases']      = False

params['pflip']             = 0.1

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

if params['equal_per_classes']:
    
   found_all_files = True
   
   for cl_file in class_files:
   
       if not os.path.exists(cl_file):
           
          found_all_files = False
          
          break
          
   if not found_all_files:
      print("Will save class-specific images to individual files")
      utils.save_images_per_class(images = all_train_images, 
                                  labels = all_train_labels, 
                                  root_dir = "./")
                                  
select_images = False
                            
if params['num_to_learn'] < params['N_train']:
   
   print("Will train on only %d randomly selected images"%params['num_to_learn'])
   
   params['N_train'] = params['num_to_learn']
   
   select_images =  True
   
   assert params['batch_size'] <= params['num_to_learn']

params['num_iters'] = params['N_train'] // params['batch_size']

alpha= 0.995 # 0.5

####### gobal parameters end
if params['use_gpu']:    
   print("Will attempt to use GPU")
   os.environ['THEANO_FLAGS'] = 'device=cuda'
   
import theano
import theano.tensor as T
from   model_classes import BoltzmannMachine

exp1 ={'num_epochs'    : params['num_epochs'],
       'batch_size'    : params['batch_size'],
       'learning_rate' : params['learning_rate'],
       'algorithm'     : 'CSS',
       'algorithm_dict':
           {
            'data_samples'  : False,
            'num_samples'   : 100,
            'resample'      : False,  
            'use_is'        : True,
            'alpha'         : alpha,
            'mf_steps'      : 0,
           
           },
       
       'use_momentum'  : params['use_momentum'],
       'momentum'      : params['momentum'],
       'num_hidden'    : params['num_hidden'],
       'use_gpu'       : params['use_gpu'],
       'report_p_tilda': True
       }
         
exp2 ={'num_epochs'    : params['num_epochs'],
       'batch_size'    : params['batch_size'],
       'learning_rate' : params['learning_rate'],
       'num_hidden'    : params['num_hidden'],
       'algorithm'     : 'CD',
       'algorithm_dict':
           {
           'num_cd_steps':1,
           },
       'use_gpu'       : params['use_gpu'],
       'report_p_tilda': False
       }
       
exp3 ={'num_epochs'    : params['num_epochs'],
       'batch_size'    : params['batch_size'],
       'learning_rate' : params['learning_rate'],
       'num_hidden'    : params['num_hidden'],
       'algorithm'     : 'PCD',
       'algorithm_dict':
           {
           'num_cd_steps':1,
           },
       'use_gpu'       : params['use_gpu'],
       'report_p_tilda': False
       }

experiments = {'exp1':exp1,'exp2':exp2, 'exp3': exp3}

exp1_string = experiments['exp1']['algorithm']

exp2_string = experiments['exp2']['algorithm']

exp3_string = experiments['exp3']['algorithm']

dir_name ="logs_%s_%s_%s"%(exp1_string, exp2_string, exp3_string)

curr_time = datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" )

spec_tag = "NR%dNEP%dLR%sM%s_%s"%(params['num_runs'],
                                  params['num_epochs'],
                                  str(params['learning_rate']),
                                  str(params['momentum']),
                                  curr_time)
  
root_path = os.path.join(dir_name, spec_tag)

os.makedirs(root_path)

all_params = {'exp1':exp1,
              'exp2':exp2, 
              'exp3':exp3, 
              'GLOBAL': params}
              
with open(os.path.join(root_path,'PARAMETERS.json'), 'w') as json_file:
    
     json.dump(all_params, json_file)
     
# TODO: W0, b0, bhid0 init

if params['num_hidden'] > 0 and params['xavier_init']:

   W0 = np_rand_gen.uniform(
      -4*np.sqrt(6.0/(params['D']  + params['num_hidden'])),\
       4*np.sqrt(6.0 /(params['D'] + params['num_hidden'])), 
      size = (params['D'], params['num_hidden'])
     )
     
elif params['num_hidden'] ==0 and params['xavier_init']:

   W0 = np_rand_gen.uniform(
      -4*np.sqrt(6.0/(params['D']  + params['D'])),\
       4*np.sqrt(6.0 /(params['D'] + params['D'])), 
      size = (params['D'], params['D'])
     )
   
elif params['num_hidden'] > 0 and params['normal_init']:
    
   W0 = 0.0*0.00000001*\
     np_rand_gen.normal(size = (params['D'], params['num_hidden']))
   
elif params['num_hidden'] == 0 and params['normal_init']:
    
   W0 = 0.0*0.00000001*\
   np_rand_gen.normal(size = (params['D'], params['D']))
                    
W0 = np.asarray(W0, dtype = theano.config.floatX) 

if params['zero_wii']:
   
   W0 = W0 - np.diag(np.diag(W0)) 
   
W0_stored = np.copy(W0)
   
b0        = np.zeros(params['D'], dtype = theano.config.floatX)

b0_stored = np.copy(b0)

if params['num_hidden'] > 0:
    
   bhid0 =  np.zeros(params['num_hidden'], dtype = theano.config.floatX)
                     
   bhid0_stored = np.copy(bhid0)
   
else:
    
   bhid0 = None
   
if params['save_init_weights']:
    
   np.savetxt(os.path.join(root_path,"W0.dat"), W0)
   
   np.savetxt(os.path.join(root_path,"b0.dat"), b0)
   
   if params['num_hidden'] > 0:
       
      np.savetxt(os.path.join(root_path,"bhid0.dat"), bhid0)
      
      print("saved initial weights for Restricted Boltzmann Machine")
      
   elif params['num_hidden'] == 0:
       
      print("saved initial weights for Fully Visible Boltzmann Machine")
      
avg_errors = {}

for tag in experiments.keys():
    
    avg_errors[experiments[tag]['algorithm']] = {}
    
    avg_errors[experiments[tag]['algorithm']]['MISSING'] = \
    np.zeros(params['num_runs'])
    
    avg_errors[experiments[tag]['algorithm']]['NOISY'] = \
    np.zeros(params['num_runs'])

for run_index in range(params['num_runs']):
    
    run_path = os.path.join(root_path, "run%d"%run_index)

    os.makedirs(run_path)
    
    if select_images:
   
       if params['equal_per_classes']:
   
          train_images = utils.select_subset(class_files, 
                                             n = params['num_to_learn']//10,
                                             D = params['D'])
                                   
       else:
       
          train_inds = np.random.choice(range(params['N_train']), 
                                        params['num_to_learn'], 
                                        replace=False)
                                        
          np.savetxt(os.path.join(run_path,"LEARNT_INSTANCES.dat"), train_inds)
                                    
          train_images = all_train_images[train_inds,:]
          
       print("Saving selected training images")
       np.savetxt(os.path.join(run_path,"TRAIN_IMAGES.dat"), train_images)
    
    which_noise_pixels = utils.get_noisy_pixels(pflip = params['pflip'], 
                                                D = params['D'], 
                                                N = params['num_to_learn'])
                                               
    noisy_images = np.copy(train_images)

    noisy_images[which_noise_pixels] = 1- noisy_images[which_noise_pixels]

    which_missing_pixels = utils.get_missing_pixels(gamma = 0.5, 
                                                    D= params['D'], 
                                                    N= params['num_to_learn'])
                                                   
    blocked_images    = np.copy(train_images)
    blocked_images[which_missing_pixels]    = -1
    
    reconst_noise_images= {}

    reconst_missing_all = {}

    w_norms_all = {}
    
    for tag in experiments.keys():
    
        loc_params = experiments[tag]
    
        print("Algorithm : %s"%loc_params['algorithm'])
        
        bm = BoltzmannMachine(num_vars        = params['D'], 
                              num_hidden      = params['num_hidden'],
                              training_inputs = train_images,
                              algorithm       = loc_params['algorithm'],
                              algorithm_dict  = loc_params['algorithm_dict'],
                              batch_size      = params['batch_size'],
                              use_momentum    = params['use_momentum'],
                              W0              = W0, 
                              b0              = b0, 
                              bhid0           = bhid0,
                              report_p_tilda  = loc_params['report_p_tilda'],
                              learn_biases    = params['learn_biases'])
    
        bm.add_graph()
    
        exp_path = os.path.join(run_path, loc_params['algorithm'])
        
        os.makedirs(exp_path)
    
        p_tilda_all, losses, train_time, w_norms = \
        bm.train_model(num_epochs = loc_params['num_epochs'], 
                       learning_rate = params['learning_rate'], 
                       momentum = params['momentum'], 
                       num_iters = params['num_iters'],
                       report_pseudo_cost = params['report_pseudo_cost'],
                       save_every_epoch   = params['save_every_epoch'],
                       report_step = params['report_step'],
                       report_p_tilda  = loc_params['report_p_tilda'],
                       report_w_norm   = params['report_w_norm'],
                       exp_path  = exp_path)
                   
        w_norms_all[loc_params['algorithm']] = w_norms
                   
        if params['report_pseudo_cost']:
    
           np.savetxt(os.path.join(exp_path, "TRAIN_PSEUDO_LOSSES.dat"), losses)
       
        if params['report_w_norm']:
        
           np.savetxt(os.path.join(exp_path, "W_NORMS.dat"), w_norms)

        np.savetxt(os.path.join(exp_path, "TRAINING_TIME.dat"), 
                   np.array([train_time]))

        if loc_params['report_p_tilda']:
    
           np.savetxt(os.path.join(exp_path,"TRAIN_P_TILDA.dat"), 
                      p_tilda_all)
       
        W0 = np.copy(W0_stored)
        b0 = np.copy(b0_stored)
        if params['num_hidden']:
           bhid0 = np.copy(bhid0_stored)
       
        #### check initial parameters:
        assert (W0_stored == W0).all()  == True
        assert (b0_stored == b0).all()  == True
    
        if params['num_hidden']:
           assert (bhid0_stored == bhid0).all() == True
       
        if not params['learn_biases']:
       
           assert (bm.b.get_value() == b0_stored).all() == True
       
        ###### reconstruction of missing pixels
        print("Reconstructing images from images with missing pixels")
    
        filename = "RECONST_MISSING"

        save_plots_to = os.path.join(exp_path, filename+".jpeg")
    
        images_to_reconst = np.copy(train_images)
    
        images_to_reconst[which_missing_pixels] =  1
    
        images_to_reconst = \
        bm.reconstruct_missing(num_iters    = 1, 
                              recon_images = images_to_reconst, 
                              which_pixels = which_missing_pixels)
                                  
        reconst_missing_all[loc_params['algorithm']] = images_to_reconst
                                                                            
        recon_errors= utils.plot_reconstructions(train_images,
                                                 blocked_images,
                                                 images_to_reconst,
                                                 save_plots_to)
                                         
        np.savetxt(os.path.join(exp_path,"%s_ERRORS.dat"%filename), 
                   recon_errors)

        avg_errors[loc_params['algorithm']]['MISSING'][run_index] =\
        np.mean(recon_errors)
        ############### reconstruction of noisy data
        print("Reconstructing images from noisy images")
    
        filename = "RECONST_NOISY"

        save_plots_to = os.path.join(exp_path, filename+".jpeg")
    
        images_to_reconst= np.copy(noisy_images)

        images_to_reconst = bm.reconstruct_noisy(num_iters   = 1, 
                                                 correct_images = train_images,
                                                 recon_images = images_to_reconst, 
                                                 noisy_images = noisy_images,
                                                 pflip = params['pflip'])
                                                    
        reconst_noise_images[loc_params['algorithm']] = images_to_reconst
                                                
        recon_errors= utils.plot_reconstructions(train_images,
                                                 noisy_images,
                                                 images_to_reconst,
                                                 save_plots_to)
                                         
        np.savetxt(os.path.join(exp_path, "%s_ERRORS.dat"%filename),
                   recon_errors)

        avg_errors[loc_params['algorithm']]['NOISY'][run_index] = \
        np.mean(recon_errors)
        
    utils.plot_w_norms(w_norms_all, os.path.join(run_path, "W_NORMS.jpeg"))
    
    utils.compare_reconstructions(train_images,
                                  noisy_images,
                                  reconst_noise_images,
                                  os.path.join(run_path,"COMPARE_NOISE.jpeg"))
                              
    utils.compare_reconstructions(train_images,
                                  blocked_images,
                                  reconst_missing_all,
                                  os.path.join(run_path,"COMPARE_MISSING.jpeg"))
                                  
std_errors = {}
                                  
for tag in experiments.keys():
    
    s  = experiments[tag]['algorithm']
    
    std_errors[s] = {}
    
ordered_labels     = []

avg_errors_missing = []

std_errors_missing = []

avg_errors_noisy   = [] 

std_errors_noisy   = []
    
for tag in experiments.keys():
    
    s = experiments[tag]['algorithm']
    print(s)
    ordered_labels.append(s)
    
    std_errors[s]['MISSING'] = np.std(avg_errors[s]['MISSING'])
    
    avg_errors[s]['MISSING'] = np.mean(avg_errors[s]['MISSING'])
    
    avg_errors_missing.append(avg_errors[s]['MISSING'])
    
    std_errors_missing.append(std_errors[s]['MISSING'])
    
    std_errors[s]['NOISY'] = np.std(avg_errors[s]['NOISY'])
    
    avg_errors[s]['NOISY'] = np.mean(avg_errors[s]['NOISY'])
    
    avg_errors_noisy.append(avg_errors[s]['NOISY'])
    
    std_errors_noisy.append(std_errors[s]['NOISY'])

print("Average reconstruction errors:")
print(avg_errors)

with open(os.path.join(root_path,'MEAN_RECON_ERRORS.json'), 'w') as json_file:
    
     json.dump(avg_errors, json_file)
     
with open(os.path.join(root_path,'STD_RECON_ERRORS.json'), 'w') as json_file:
    
     json.dump(std_errors, json_file)
####################################################     
fig, ax = plt.subplots(1, 2, sharex=False)

ax = ax.ravel()

width = 0.7
num_exps = len(experiments.keys())
x_axis   = np.arange(num_exps)
####################################################
ax[0].bar(x_axis, 
          avg_errors_missing, 
          width = width, 
          color='b', 
          yerr= std_errors_missing)

ax[0].set_ylabel('Mean Reconstruction Errors')
ax[0].set_xticks(x_axis + width / 2)
ax[0].set_xticklabels(ordered_labels)
ax[0].set_title('Reconstruction of missing pixels')
##################################################
ax[1].bar(x_axis, 
          avg_errors_noisy, 
          width = width, 
          color='b', 
          yerr=std_errors_noisy)

ax[1].set_ylabel('Mean Reconstruction Errors')
ax[1].set_xticks(x_axis + width / 2)
ax[1].set_xticklabels(ordered_labels)
ax[1].set_title('Reconstruction of noisy pixels')
########################################################
plt.tight_layout()
plt.savefig(os.path.join(root_path,"BAR_ERRORS.jpeg"))
plt.clf()
       
       
    


    
    
