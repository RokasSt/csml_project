""" 
Author: Rokas Stanislovas
MSc Project: Likelihood Approximations
for Energy-Based Models
MSc Computational Statistics and 
Machine Learning
"""
import numpy as np
from plot_utils import plot_reconstructions
import model_classes as clv1
import model_classes_v2  as clv2
import os

def run_experiment(glob_params, 
                   method_params, 
                   init_params,
                   training_inputs,
                   reconst_arrays,
                   missing_pixels,
                   blocked_images,
                   noisy_images,
                   exp_path,
                   version = "v1",
                   collect_w_norms = None,
                   collect_reconst = None):
                            
    """ function to run a single training experiment """
    
    if version == "v1":
        
       cl =clv1
       
    elif version == "v2":
    
       cl = clv2
                           
    bm = cl.BoltzmannMachine(num_vars       = glob_params['D'], 
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
    
    images_to_reconst = np.copy(training_inputs)
    
    images_to_reconst[missing_pixels] =  1
    
    reconst_images = \
    bm.reconstruct_missing(num_iters    = glob_params['num_reconst_iters'], 
                           recon_images = reconst_arrays['MISSING'], 
                           which_pixels = missing_pixels)
                              
    if collect_reconst != None:
            
      collect_reconst['MISSING'][method_params['algorithm']] = reconst_images
                                  
    np.savetxt(os.path.join(exp_path, "RECONST_MISSING.dat"), reconst_images)
                                                                            
    recon_errors = plot_reconstructions(training_inputs,
                                        blocked_images,
                                        reconst_images,
                                        save_plots_to)
                                         
    np.savetxt(os.path.join(exp_path,"%s_ERRORS.dat"%filename), recon_errors)

    reconstruction_errors['MISSING'] = np.mean(recon_errors)
    ############### reconstruction of noisy data
    print("Reconstructing images from noisy images")
    
    filename = "RECONST_NOISY"

    save_plots_to = os.path.join(exp_path, filename+".jpeg")
    
    reconst_images = \
    bm.reconstruct_noisy(num_iters = glob_params['num_reconst_iters'], 
                         correct_images= training_inputs,
                         recon_images  = reconst_arrays['NOISY'], 
                         noisy_images  = noisy_images,
                         pflip         = glob_params['pflip'])
                                                 
    np.savetxt(os.path.join(exp_path, "RECONST_NOISY.dat"), reconst_images)
    
    if collect_reconst != None:
                                                    
       collect_reconst['NOISY'][method_params['algorithm']] = reconst_images
                                                
    recon_errors= plot_reconstructions(training_inputs,
                                       noisy_images,
                                       reconst_images,
                                       save_plots_to)
                                         
    np.savetxt(os.path.join(exp_path, "%s_ERRORS.dat"%filename), recon_errors)

    reconstruction_errors['NOISY'] = np.mean(recon_errors)
    
    return bm, collect_w_norms, collect_reconst, reconstruction_errors
