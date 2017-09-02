""" 
Author: Rokas Stanislovas
MSc Project: Complementary Sum Sampling 
for Learning in Boltzmann Machines
MSc Computational Statistics and 
Machine Learning

Training functions.
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
from matplotlib import pyplot as plt
import plot_utils
import copy
from plot_utils    import plot_reconstructions
from model_classes import BoltzmannMachine

def run_experiment(glob_params, 
                   method_params, 
                   init_params,
                   training_inputs,
                   images_to_reconst,
                   reconst_arrays,
                   missing_pixels,
                   blocked_images,
                   noisy_images,
                   exp_path,
                   collect_w_norms = None,
                   collect_reconst = None):
                            
    """ function for running a single training experiment and testing 
    trained Boltzmann Machine. 
    
    glob_params       - dictionary of global parameters
    method_params     - dictionary of algorithm-specific parameters
    init_params       - dictionary of inital values for parameters W and b
                        init_params[W0], init_params[b0], init_params[bhid0] 
    training_inputs   - N x D matrix of D-dimensional training inputs
    
    images_to_reconst - matrix of training inputs for which perform
                        reconstruction tasks
    reconst_arrays    - dictionary of matrices of corrupted images:
                        reconst_arrays['MISSING'] and 
                        reconst_arrays['NOISY'];
                        each array must have the same dimensions as
                        images_to_reconst.
    missing_pixels    - matrix to indicate which pixels are missing;
                        must have the same dimensions as images_to_reconst
    blocked_images    - matrix of incomplete versions of training examples 
                        provided in images_to_reconst. Used for 
                        plotting. Must have the same dimensions as 
                        images_to_reconst
    noisy_images      - matrix of noisy versions of training examples
                        provided in images_to_reconst. Must have the same 
                        dimensions as images_to_reconst 
    exp_path          - full path for saving data and reconstructions
    
    collect_w_norms   - (default None) optional dictionary to which the
                        array of L2 norms on W matrix can be added;
                        the key corresponding to the added array is the
                        name of the training algorithm (e.g. CD1)
    
    collect_reconst   - (default None) dictionary of reconstructed images
                        at the top level there must be two keys: 
                        "MISSING" and "NOISY", corresponding to
                        reconstruction tasks; then, each of these two
                        entries is a dictionary over individual algorithms.
                        
    return:
                        bm - instance of Boltzmann Machine
                        collect_w_norms - (default None) 
                        collect_reconst - (default None)
                        reconstruction_errors - dictionary of reconstruction
                        errors with keys ['MISSING', 'NOISY'], corresponding
                        to the individual reconstruction tasks. 
    """
            
    bm = BoltzmannMachine(num_vars       = glob_params['D'], 
                          num_hidden     = glob_params['num_hidden'],
                          training_inputs= training_inputs,
                          algorithm       = method_params['algorithm'],
                          algorithm_dict  = method_params['algorithm_dict'],
                          batch_size      = glob_params['batch_size'],
                          use_momentum    = glob_params['use_momentum'],
                          W0              = init_params['W0'], 
                          b0              = init_params['b0'], 
                          bhid0           = init_params['bhid0'],
                          zero_diag       = glob_params['zero_diag'],
                          report_p_tilda  = method_params['report_p_tilda'],
                          learn_biases    = glob_params['learn_biases'])
                                  
    bm.add_graph()
    
    p_tilda_all, losses, train_time, w_norms = \
    bm.train_model(num_epochs         = glob_params['num_epochs'], 
                   learning_rate      = glob_params['learning_rate'], 
                   momentum           = glob_params['momentum'], 
                   num_iters          = glob_params['num_iters'],
                   report_pseudo_cost = glob_params['report_pseudo_cost'],
                   save_every_epoch   = glob_params['save_every_epoch'],
                   report_step        = glob_params['report_step'],
                   report_p_tilda     = method_params['report_p_tilda'],
                   report_w_norm      = glob_params['report_w_norm'],
                   exp_path           = exp_path)
    
    if collect_w_norms != None:            
       collect_w_norms[method_params['algorithm']] = w_norms
                   
    if glob_params['report_pseudo_cost']:
       np.savetxt(os.path.join(exp_path, "TRAIN_PSEUDO_LOSSES.dat"), losses)
       
    if glob_params['report_w_norm']:
       np.savetxt(os.path.join(exp_path, "W_NORMS.dat"), w_norms)

    np.savetxt(os.path.join(exp_path, "TRAINING_TIME.dat"),
               np.array([train_time]))

    if method_params['report_p_tilda']:
    
       np.savetxt(os.path.join(exp_path,"TRAIN_P_TILDA.dat"), 
                  p_tilda_all)
       
    reconstruction_errors = {}
    ###### reconstruction of missing pixels
    print("Reconstructing images from images with missing pixels")
    
    filename = "RECONST_MISSING"

    save_plots_to = os.path.join(exp_path, filename+".jpeg")
    
    bm.reconstruct_missing(num_iters    = glob_params['num_reconst_iters'], 
                           recon_images = reconst_arrays['MISSING'], 
                           which_pixels = missing_pixels)
                              
    if collect_reconst != None:
            
      collect_reconst['MISSING'][method_params['algorithm']] =\
      reconst_arrays['MISSING']
                                  
    np.savetxt(os.path.join(exp_path, "RECONST_MISSING.dat"), 
               reconst_arrays['MISSING'])
                                                                            
    recon_errors = plot_reconstructions(images_to_reconst,
                                        blocked_images,
                                        reconst_arrays['MISSING'],
                                        save_plots_to)
                                         
    np.savetxt(os.path.join(exp_path,"%s_ERRORS.dat"%filename), recon_errors)

    reconstruction_errors['MISSING'] = np.mean(recon_errors)
    ############### reconstruction of noisy data
    print("Reconstructing images from noisy images")
    
    filename = "RECONST_NOISY"

    save_plots_to = os.path.join(exp_path, filename+".jpeg")
    
    bm.reconstruct_noisy(num_iters = glob_params['num_reconst_iters'], 
                         correct_images= images_to_reconst,
                         recon_images  = reconst_arrays['NOISY'], 
                         noisy_images  = noisy_images,
                         pflip         = glob_params['pflip'])
                                                 
    np.savetxt(os.path.join(exp_path, "RECONST_NOISY.dat"), 
               reconst_arrays['NOISY'])
    
    if collect_reconst != None:
                                                    
       collect_reconst['NOISY'][method_params['algorithm']] =\
       reconst_arrays['NOISY']
                                                
    recon_errors= plot_reconstructions(images_to_reconst,
                                       noisy_images,
                                       reconst_arrays['NOISY'],
                                       save_plots_to)
                                         
    np.savetxt(os.path.join(exp_path, "%s_ERRORS.dat"%filename), recon_errors)

    reconstruction_errors['NOISY'] = np.mean(recon_errors)
    
    return bm, collect_w_norms, collect_reconst, reconstruction_errors
    
############################################################################
def compare_algorithms(train_data_inputs,
                       train_data_labels,
                       params ={'num_runs': 1,
                                'N_train' : 60000,
                                'D': 784,
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
                       np_rand_gen = np.random.RandomState(1234)):
    
    """ main function to compare different algorithms.
    
    train_data_inputs - number_examples x D matrix of training inputs
    
    train_data_labels - number_examples x number_classes matrix of class
                        labels for training images
    
    params            - dictionary of global training settings (see above)
    
    class_files       - list of file names for class-specific images
    
    exps              - dictionary of settings for individual training
                        algorithms (see above)
                        
    experiment_id     - (default "DEFAULT") string to identify the path
                        where training data and reconstruction results
                        are saved, e.g.
                        logs_CD1_CSS/experiment_id_current_date/run0/ ...
                                                               /run1/ ...
                                                               /run2/ ...
                                                               ...
                                                               ...
    np_rand_gen       - instance of seeded numpy random generator .
    
    """
    
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
       
    if params['num_hidden'] == 0:
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
           
           for alg_var in exps[tag]['algorithm_dict'].keys():
               
               test_var = exps[tag]['algorithm_dict'][alg_var]
               
               if alg_var == var_name:
                  if isinstance(test_var,list) != True:
                     print("Error: variable '%s' in algorithm_dict was"%alg_var+\
                     " specified as a regressor but its type is not 'list'")
                     sys.exit()
                  
               else:
                  if isinstance(test_var, list) == True:
                     print("Error: variable '%s' in algorithm_dict was not"%alg_var+\
                     " specified as a regressor but its type is 'list'")
                     sys.exit()
                  
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
    
        which_noise_pixels =\
        utils.get_noisy_pixels(pflip = params['pflip'], 
                               D = params['D'], 
                               N = params['num_to_reconstruct'])
                                               
        noisy_images = np.copy(imgs_to_reconstruct)

        noisy_images[which_noise_pixels] = 1-noisy_images[which_noise_pixels]
        
        which_missing_pixels =\
        utils.get_missing_pixels(gamma = params['pmiss'], 
                                 D     = params['D'], 
                                 N     = params['num_to_reconstruct'])
                                                   
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
    
    
    
    
    
    
