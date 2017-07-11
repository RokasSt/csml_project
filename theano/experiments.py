""" 

Author: Rokas Stanislovas 

"""

class Experiments(object):
    
  """ Add parameters of the experiments here """
    
  exp1 ={'num_epochs'   :   '1500',
         'batch_size'   :   '10',
         'learning_rate':   '0.05',
         'experiment'   :   'exp1_CSS',
         'data_samples' :   '0', # 0 means that only minibatch points
                #will contribute to the data term of Z approximation.
         'num_samples'  :   '100',
         'resample'     :   '0',  # '0' - do not resample for each minibatch point.
         'momentum'     :   '0',
         'use_is'       :   '1',
         'algorithm'    :   'CSS',
         'num_hidden'   :   '0',
         'use_gpu'      :   '0', #'1'
         'learn_subset' :   '10'}
          
  exp2 ={'num_epochs'   :   '15',
         'batch_size'   :   '50',
         'learning_rate':   '0.05',
         'experiment'   :   'exp2_CSS',
         'data_samples' :   '55000',# a whole training set is used to approx. Z
         'num_samples'  :   '0',
         'algorithm'    :   'CSS',
         'use_gpu'      :   '1'}
         
  exp3 ={'num_epochs'   :   '50',
         'batch_size'   :   '50',
         'learning_rate':   '0.05',
         'experiment'   :   'exp1_CSS_MF',
         'data_samples' :   '0',
         'num_samples'  :   '500',
         'mf_steps'     :   '200',
         'resample'     :   '0',
         'algorithm'    :   'CSS',
         'num_hidden'   :   '500',
         'use_gpu'      :   '0',
         'learn_subset' :   '0'}
         
  exp4 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment'   :   'exp2_CSS_MF',
         'data_samples' :   '5000',
         'num_samples'  :   '5000',
         'resample'     :   '1',
         'algorithm'    :   'CSS',
         'use_gpu'    :     '1'}
  
  exp5 ={'num_epochs':      '1500',
         'batch_size':      '10',
         'learning_rate':   '0.05',
         'experiment':      'exp1_CD1',
         'num_steps':       '1',
         'algorithm'  :     'CD',
         'use_gpu'    :     '0',
         'learn_subset' :   '10'}
         
  exp6 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment':      'exp2_CD1',
         'num_steps':       '1',
         'algorithm'  :     'CD',
         'num_hidden' :     '500',
         'use_gpu'    :     '0'}
         
  exp7 ={'num_epochs':      '15',
         'batch_size':      '20',
         'learning_rate':   '0.1',
         'experiment':      'exp1_PCD',
         'num_steps':       '15',
         'algorithm'  :     'PCD',
         'num_hidden' :     '500',
         'use_gpu'    :     '0'}       
         
  experiments= {exp1['experiment']:exp1,
                exp2['experiment']:exp2,
                exp3['experiment']:exp3,
                exp4['experiment']:exp4,
                exp5['experiment']:exp5,
                exp6['experiment']:exp6,
                exp7['experiment']:exp7}
