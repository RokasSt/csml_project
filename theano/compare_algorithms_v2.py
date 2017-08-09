""" 
Author: Rokas Stanislovas
MSc Project: Complementary Sum Sampling 
for Learning in Boltzmann Machines
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
import datetime
import utils
import argparse
import timeit
import os
from matplotlib import pyplot as plt
import plot_utils
import copy

print("Importing data ...")
all_images, all_labels = utils.get_data_arrays()

all_train_images = all_images[range(0,60000),:]

all_train_labels = all_labels[range(0,60000),:]
    
def compare_algorithms(params ={'num_runs': 20,
                                'N_train' : all_train_images.shape[0],
                                'D': all_train_images.shape[1],
                                'use_gpu': False,
                                'num_epochs': 1500,
                                'report_step':1,
                                'save_every_epoch': False,
                                'report_w_norm': True,
                                'save_init_weights':True,
                                'report_pseudo_cost':True,
                                'learning_rate':0.01,
                                'batch_size':10,
                                'use_momentum':True,
                                'momentum':0.95,
                                'num_hidden':0,
                                'num_to_learn':10,
                                'equal_per_classes':True,
                                'init_type'   :'ZEROS', # 'XAV' or 'NORM'
                                'zero_diag'   : False,
                                'learn_biases': False,
                                'num_reconst_iters' :5,
                                'num_to_reconstruct':10,
                                'pflip': 0.1,
                                'pmiss': 0.5},
                       class_files = ["CLASS0.dat",
                                      "CLASS1.dat",
                                      "CLASS2.dat",
                                      "CLASS3.dat",
                                      "CLASS4.dat",
                                      "CLASS5.dat",
                                      "CLASS6.dat",
                                      "CLASS7.dat",
                                      "CLASS8.dat",
                                      "CLASS9.dat"
                                      ],
                       exps ={
                       'exp1':{'algorithm' : 'CSS',
                               'algorithm_dict':
                                   {
                                        'num_samples'   : 100,
                                        'resample'      : False,  
                                        'alpha' : 0.5, # 0.7 #0.3 # 0.0 # 0.995 
                                        'mf_steps'      : 0,
                                        },
                               'report_p_tilda': True,
                               'regressor':None},
                       'exp2':{'algorithm'     : 'CD1',
                               'algorithm_dict':
                                   {
                                        'gibbs_steps':1,
                                    },
                               'report_p_tilda': False,
                               'regressor': None},
                       'exp3':{'algorithm'     : 'PCD1',
                               'algorithm_dict':
                                   {
                                        'gibbs_steps':1,
                                    },
                               'report_p_tilda': False,
                               'regressor':None}
                       },
                       experiment_id = "DEFAULT",
                       train_data_inputs = all_train_images,
                       train_data_labels = all_train_labels,
                       np_rand_gen = np.random.RandomState(1234)):
    
    """ master function to compare different algorithms """
    
    assert params['D'] == 784

    assert params['report_step'] != None
    
    if params['equal_per_classes']:
    
       found_all_files = True
   
       for cl_file in class_files:
   
           if not os.path.exists(cl_file):
           
              found_all_files = False
          
              break
          
    if not found_all_files:
       print("Will save class-specific images to individual files")
       utils.save_images_per_class(images = train_data_inputs, 
                                   labels = train_data_labels, 
                                   root_dir = "./")
                                  
    select_images = False
                            
    if params['num_to_learn'] != params['N_train']:
   
       print("Will train on only %d randomly selected images"%params['num_to_learn'])
   
       params['N_train'] = params['num_to_learn']
   
       select_images =  True
   
       assert params['batch_size'] <= params['num_to_learn']

    params['num_iters'] = params['N_train'] // params['batch_size'] 
    
    if params['use_gpu']:    
       print("Will attempt to use GPU")
       os.environ['THEANO_FLAGS'] = 'device=cuda'
   
    import theano
    import theano.tensor as T
    from   train_utils   import run_experiment
    ###### specify saving directory ####################################
    dir_name = "logs"
    for exp_tag in exps.keys():
        
        exp_string = exps[exp_tag]['algorithm']
        
        dir_name += "_%s"%exp_string

    curr_time = datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" )

    spec_tag = "%s_%s"%(experiment_id, curr_time)
                                          
    root_path = os.path.join(dir_name, spec_tag)

    os.makedirs(root_path)

    all_params = copy.deepcopy(exps)
    
    for exp_tag in all_params.keys():
        
        if 'mix_params' in all_params[exp_tag]['algorithm_dict']:
            
           coeffs = all_params[exp_tag]['algorithm_dict']['mix_params']
            
           np.savetxt(os.path.join(root_path,'MIX_PARAMS.dat'), coeffs) 
           
           del all_params[exp_tag]['algorithm_dict']['mix_params']
        
    all_params['GLOBAL'] =  params
    
    with open(os.path.join(root_path,'PARAMETERS.json'), 'w') as json_file:
    
         json.dump(all_params, json_file)
    ################ initialization block #############################
    if params['num_hidden'] > 0 and params['init_type'] == "XAV":

       W0 = np_rand_gen.uniform(
           -4*np.sqrt(6.0/(params['D']  + params['num_hidden'])),\
            4*np.sqrt(6.0 /(params['D'] + params['num_hidden'])), 
            size = (params['D'], params['num_hidden']))
     
    elif params['num_hidden'] ==0 and params['init_type'] == "XAV":

       W0 = np_rand_gen.uniform(
           -4*np.sqrt(6.0/(params['D']  + params['D'])),\
            4*np.sqrt(6.0 /(params['D'] + params['D'])), 
            size = (params['D'], params['D']))
   
    elif params['num_hidden'] > 0 and params['init_type'] == "NORM":
    
       W0 = 0.00000001*\
       np_rand_gen.normal(size = (params['D'], params['num_hidden']))
   
    elif params['num_hidden'] == 0 and params['init_type'] == "NORM":
            
       W0 = 0.00000001*\
       np_rand_gen.normal(size = (params['D'], params['D']))
   
    elif params['num_hidden'] > 0 and params['init_type'] == "ZEROS":
    
       W0 = np.zeros([params['D'], params['num_hidden']])
   
    elif params['num_hidden'] == 0 and params['init_type'] == "ZEROS":
    
       W0 = np.zeros([params['D'], params['D']])
       
    if params['num_hidden']:
       W0 = (W0 + np.transpose(W0))/2.0
                    
    W0 = np.asarray(W0, dtype = theano.config.floatX) 

    if params['zero_diag']:
   
       W0 = W0 - np.diag(np.diag(W0)) 
   
    W0_stored = np.copy(W0)
   
    b0   = np.zeros(params['D'], dtype = theano.config.floatX)

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

    num_bars = 0
    ################ prepare dictionary for average reconstruction errors
    for tag in exps.keys():
    
        avg_errors[exps[tag]['algorithm']] = {}
        
        var_name = exps[tag]['regressor']
        
        if var_name != None:
            
           assert exps[tag]['regressor'] \
           in exps[tag]['algorithm_dict'].keys()
           
           get_vals = exps[tag]['algorithm_dict'][var_name]
           
           assert isinstance(get_vals,list) == True
           
           avg_errors[exps[tag]['algorithm']]['MISSING'] = {}
           
           avg_errors[exps[tag]['algorithm']]['NOISY']   = {}
           
           for val in get_vals:
               
               field = "%s %s"%(var_name, str(val))
               
               avg_errors[exps[tag]['algorithm']]['MISSING'][field] = \
               np.zeros(params['num_runs'])
           
               avg_errors[exps[tag]['algorithm']]['NOISY'][field]  = \
               np.zeros(params['num_runs'])
               
               num_bars +=1
        
        else:
    
           avg_errors[exps[tag]['algorithm']]['MISSING'] = \
           np.zeros(params['num_runs'])
    
           avg_errors[exps[tag]['algorithm']]['NOISY'] = \
           np.zeros(params['num_runs'])
           
           num_bars +=1
    
    ####################################################################
    ### run experiments ################################################
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
                                        
              np.savetxt(os.path.join(run_path,"LEARNT_INSTANCES.dat"), 
                         train_inds)
                                    
              train_images = train_data_inputs[train_inds,:]
          
           print("Saving selected training images")
           np.savetxt(os.path.join(run_path,"TRAIN_IMAGES.dat"), train_images)
           
           imgs_to_reconstruct = train_images
           
        else:
            
           ####### training uses all of the data
           train_images = train_data_inputs
           
           ##### then reconstruct only a specified number
           if params['equal_per_classes']:
   
              imgs_to_reconstruct = \
              utils.select_subset(class_files, 
                                  n = params['num_to_reconstruct']//10,
                                  D = params['D'])
                                  
           else:
       
              reconst_inds = np.random.choice(range(params['N_train']), 
                                              params['num_to_reconstruct'], 
                                              replace=False)
                                            
              imgs_to_reconstruct = train_data_inputs[reconst_inds,:]
           
              np.savetxt(os.path.join(run_path,"RECONST_INSTANCES.dat"), 
                         reconst_inds)
              
           np.savetxt(os.path.join(run_path,"IMAGES_TO_RECONST.dat"),
                      imgs_to_reconstruct) 
    
        which_noise_pixels = utils.get_noisy_pixels(pflip = params['pflip'], 
                                                D = params['D'], 
                                                N = params['num_to_reconstruct'])
                                               
        noisy_images = np.copy(imgs_to_reconstruct)

        noisy_images[which_noise_pixels] = 1- noisy_images[which_noise_pixels]
        
        which_missing_pixels = utils.get_missing_pixels(gamma = params['pmiss'], 
                                                        D= params['D'], 
                                                        N= params['num_to_reconstruct'])
                                                   
        blocked_images    = np.copy(imgs_to_reconstruct)
        blocked_images[which_missing_pixels]  = 0.5  # = -1
        
        reconst_dict= {}
        w_norms_all = {}
        
        reconst_dict['NOISY']   = {}
        reconst_dict['MISSING'] = {}
        
        for tag in exps.keys():
    
            loc_params = copy.deepcopy(exps[tag])
            
            m_name = loc_params['algorithm']
    
            print("Algorithm : %s"%m_name)
        
            if loc_params['regressor'] != None:
           
               get_vals = exps[tag]['algorithm_dict'][loc_params['regressor']]
               
               for val in get_vals:
                   
                   reconst_dict = None
               
                   w_norms_all = None
               
                   loc_params['algorithm_dict'][loc_params['regressor']] = val
               
                   spec_str = "%s_%s%s"%(m_name,
                                         loc_params['regressor'],
                                         str(val))
               
                   exp_path = os.path.join(run_path, spec_str)
           
                   os.makedirs(exp_path)
               
                   reconst_missing = np.copy(imgs_to_reconstruct)
               
                   reconst_missing[which_missing_pixels] =  1
               
                   reconst_noisy = np.copy(noisy_images)
               
                   bm_obj, w_norms_all, reconst_dict, reconst_errors = \
                   run_experiment(glob_params     = params, 
                                  method_params   = loc_params,
                                  init_params     = {'W0':W0,
                                                     'b0':b0,
                                                     'bhid0':bhid0
                                                     },
                                  training_inputs = train_images,
                                  images_to_reconst = imgs_to_reconstruct,
                                  reconst_arrays  = {'MISSING':reconst_missing,
                                                     'NOISY'  :reconst_noisy},
                                  missing_pixels  = which_missing_pixels,
                                  blocked_images  = blocked_images,
                                  noisy_images    = noisy_images,
                                  exp_path        = exp_path,
                                  version         = "v2",
                                  collect_w_norms = w_norms_all,
                                  collect_reconst = reconst_dict)
                                  
                   ##################### reset to the same init values #####             
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
                      assert (bm_obj.b.get_value() == b0_stored).all() == True
                   ######################################################### 
                                 
                   field = "%s %s"%(var_name, str(val))
                   
                   avg_errors[m_name]['MISSING'][field][run_index] =\
                   np.mean(reconst_errors['MISSING'])
                   
                   avg_errors[m_name]['NOISY'][field][run_index]= \
                   np.mean(reconst_errors['NOISY'])                   
               
            else:
            
               exp_path = os.path.join(run_path, m_name)
        
               os.makedirs(exp_path)
               
               reconst_missing = np.copy(imgs_to_reconstruct)
               
               reconst_missing[which_missing_pixels] =  1
               
               reconst_noisy = np.copy(noisy_images)
           
               bm_obj, w_norms_all, reconst_dict, reconst_errors = \
               run_experiment(glob_params     = params, 
                              method_params   = loc_params,
                              init_params     = {'W0':W0,
                                                 'b0':b0,
                                                 'bhid0':bhid0
                                                 },
                              training_inputs = train_images,
                              images_to_reconst = imgs_to_reconstruct,
                              reconst_arrays  = {'MISSING':reconst_missing,
                                                 'NOISY'  :reconst_noisy},
                              missing_pixels  = which_missing_pixels,
                              blocked_images  = blocked_images,
                              noisy_images    = noisy_images,
                              exp_path        = exp_path,
                              version         = "v2",
                              collect_w_norms = w_norms_all,
                              collect_reconst  = reconst_dict)
                              
               ##################### reset to the same init values #####             
               W0 = np.copy(W0_stored)
               b0 = np.copy(b0_stored)
               if params['num_hidden']:
                  bhid0 = np.copy(bhid0_stored)
               #### check initial parameters:
               assert (W0_stored == W0).all() == True
               assert (b0_stored == b0).all() == True
               if params['num_hidden']:
                  assert (bhid0_stored == bhid0).all() == True
               if not params['learn_biases']:
                  assert (bm_obj.b.get_value() == b0_stored).all()== True
               ######################################################### 
               avg_errors[m_name]['MISSING'][run_index] =\
               np.mean(reconst_errors['MISSING'])
               avg_errors[m_name]['NOISY'][run_index] = \
               np.mean(reconst_errors['NOISY'])
    
        if w_norms_all != None and isinstance(w_norms_all, dict): 
           w_path = os.path.join(run_path, "W_NORMS.jpeg")
           xlabel_dict = {}
           ylabel_dict = {}
           for tag in exps.keys():
               alg = exps[tag]['algorithm']
               xlabel_dict[alg] = "Iteration number"
               ylabel_dict[alg] = "L2-norm on W"
           plot_utils.plot_sequences(means_dict   = w_norms_all, 
                                     xlabel_dict  = xlabel_dict,
                                     ylabel_dict  = ylabel_dict,
                                     save_to_path = w_path)
           
        if reconst_dict != None and isinstance(reconst_dict, dict):
           save_plot_to = os.path.join(run_path,"COMPARE_NOISE.jpeg")
           plot_utils.compare_reconstructions(train_images,
                                  noisy_images,
                                  reconst_dict['NOISY'],
                                  save_plot_to
                                  )
           save_plot_to =os.path.join(run_path,"COMPARE_MISSING.jpeg")
           plot_utils.compare_reconstructions(train_images,
                                              blocked_images,
                                              reconst_dict['MISSING'],
                                              save_plot_to)
                                  
    avg_errors, std_errors = utils.get_means_and_stds(target_dict = avg_errors)
    
    file_to_open = os.path.join(root_path, 'MEAN_RECON_ERRORS.json')
                                  
    with open(file_to_open, 'w') as json_file:
    
         json.dump(avg_errors, json_file)
         
    file_to_open = os.path.join(root_path, 'STD_RECON_ERRORS.json') 
      
    with open(file_to_open, 'w') as json_file:
    
         json.dump(std_errors, json_file)
    
    look_at = avg_errors.keys()
     
    dict_of_lists = plot_utils.process_err_dict(means_dict = avg_errors,
                                                target_fields = look_at,
                                                std_dict = std_errors,
                                                bar_plots = True)
    
    save_bar_plots_to =  os.path.join(root_path,"BAR_ERRORS.jpeg")
    
    plot_std = False
    
    if params['num_runs'] > 1:
        
       plot_std  = True

    plot_utils.display_recon_errors(array_dict   = dict_of_lists,
                                    num_exps     = num_bars,            
                                    save_to_path = save_bar_plots_to,
                                    plot_std     = plot_std)

########################################################################       
if __name__ == "__main__":
   
   exps ={'exp1':{'algorithm'     : 'CSS_GIBBS',
                  'algorithm_dict':
                          {#500, #[10, 50, 100],#[10, 50, 100, 300, 500],
                           'num_samples'   : [250,125], # 40
                           'num_u_gibbs'   : 125,
                           'resample'      : False,  
                           'alpha'         : None, #0.01, # 0.05;
                           'uniform_to_mf' : False,
                           'mixture'       : False,
                           'mix_params'    : np.array([[0.001, 0.998],
                                                       [0.2,   0.0],
                                                       [0.1,   0.0],
                                                       [0.01,  0.0],
                                                       [0.1,   0.8],
                                                       [0.01,  0.98]]),
                           'mf_steps'      : 0, #50,
                           'gibbs_steps'   : 1,
                           },
                  'report_p_tilda': True,
                  'regressor': 'num_samples',
                  },
          'exp2':{'algorithm'     : 'CD1',
                  'algorithm_dict':
                    {
                      'gibbs_steps':1,
                     },
                  'report_p_tilda': False,
                  'regressor': None},
          'exp3':{'algorithm': 'PCD1',
                  'algorithm_dict':
                    {
                        'gibbs_steps':1,
                     },
                  'report_p_tilda': False,
                  'regressor':None},
          'exp4':{'algorithm' : 'CSS',
                  'algorithm_dict':
                          {
                           'num_samples'   : [50],
                           'resample'      : False,  
                           'alpha'         : None,
                           'uniform_to_mf' : False,
                           'mixture'       : False,
                           'mf_steps'      : 0, 
                           'gibbs_steps'   : 0,
                           },
                  'report_p_tilda': True,
                  'regressor': 'num_samples',
                  },            }
                       
   #del exps['exp2'] #  uncomment for testing specific algorithm
   del exps['exp3']
   del exps['exp4']
   
   params ={'num_runs': 1,
            'N_train' : all_train_images.shape[0],
            'D': all_train_images.shape[1],
            'use_gpu': False,
            'num_epochs': 300, #15000,   #1500, 
            'report_step':1,
            'save_every_epoch': False,
            'report_w_norm': True,
            'save_init_weights':True,
            'report_pseudo_cost':True,
            'learning_rate': 0.05, #0.1, #0.01,
            'batch_size':125,
            'use_momentum':True,
            'momentum' : 0.93,  #0.90, for multi-run experiments
            'num_hidden':0,
            'num_to_learn': 60000, #10, for multi-run experiments
            'equal_per_classes':True,
            'init_type':'ZEROS', # options: 'ZEROS', 'XAV, 'NORM'
            'zero_diag': False,
            'learn_biases': False,
            'num_reconst_iters' :10,
            'num_to_reconstruct':20, # 10
            'pflip': 0.2,    #0.1,
            'pmiss': 0.9}    #0.5}
   
   compare_algorithms(params = params,
                      exps = exps,
                      experiment_id = "ALL_NS250_U125_BS125")
   
   


    
    
