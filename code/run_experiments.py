""" 
Author: Rokas Stanislovas
MSc Project: Complementary Sum Sampling 
for Learning in Boltzmann Machines
MSc Computational Statistics and 
Machine Learning

Script to run experiments using train_utils.py .
"""

import numpy as np
from train_utils import compare_algorithms
from utils import get_data_arrays

print("Importing data ...")
all_images, all_labels =  get_data_arrays()

all_train_images = all_images[range(0,60000),:]

all_train_labels = all_labels[range(0,60000),:]

exps ={'exp1':{'algorithm'     : 'CSS_GIBBS',
               'algorithm_dict':{#[10, 50, 100],#[10, 50, 100, 300, 500],
                                 'num_samples'   : [40],
                                 'num_u_gibbs'   : 0,
                                 'resample'      : False,  
                                 'alpha'         : None,
                                 'uniform'       : False,
                                 'mixture'       : False,
                                 'mix_params'    : np.array([[0.001, 0.998],
                                                             [0.2,   0.0],
                                                             [0.1,   0.0],
                                                             [0.01,  0.0],
                                                             [0.1,   0.8],
                                                             [0.01,  0.98]]),
                                 'mf_steps'      : 0, #50,
                                 'gibbs_steps'   : 1 },
               'report_p_tilda': True,
               'regressor': 'num_samples',
               },
       'exp2':{'algorithm'     : 'CD1',
               'algorithm_dict':{'gibbs_steps':1},
               'report_p_tilda': False,
               'regressor': None},
       'exp3':{'algorithm': 'PCD1',
               'algorithm_dict':{'gibbs_steps':1 },
               'report_p_tilda': False,
               'regressor':None},
       'exp4':{'algorithm' : 'CSS',
               'algorithm_dict':
                          {
                           'num_samples': 50,
                           'resample'   : False,  
                           'alpha'      : [0.5, 0.7, 0.3, 0.0, 0.1, 0.9, 0.995],
                           'uniform'    : True,
                           'mixture'    : False,
                           'mf_steps'   : 0, 
                           'gibbs_steps': 0,
                           },
               'report_p_tilda': True,
               'regressor': 'alpha',
               },            }
                       
#del exps['exp1'] #  uncomment for testing specific algorithm
del exps['exp2']  
del exps['exp3']
del exps['exp4']
   
params ={'num_runs': 1,
         'N_train' : all_train_images.shape[0],
         'D': all_train_images.shape[1],
         'use_gpu': False,
         'num_epochs': 100,  #300, #15000,   #1500, 
         'report_step':1,
         'save_every_epoch': False,
         'report_w_norm': True,
         'save_init_weights':True,
         'report_pseudo_cost':True,
         'learning_rate': 0.1, # 0.05, #0.1, #0.01,
         'batch_size': 20, #125,
         'use_momentum':True,
         'momentum' : 0.90,
         'num_hidden':500,
         'num_to_learn': 60000, #10 a whole training set
         'equal_per_classes':True,
         'init_type':'ZEROS', # options: 'ZEROS', 'XAV, 'NORM'
         'zero_diag': False,
         'learn_biases': False,
         'num_reconst_iters' :10,
         'num_to_reconstruct': 20,  #10,  #20
         'pflip': 0.2,    #0.1,
         'pmiss': 0.9}    #0.5}
   
compare_algorithms(train_data_inputs = all_train_images,
                   train_data_labels = all_train_labels,
                   params = params,
                   exps = exps,
                   experiment_id = "CSS_GIBBS_H500")
