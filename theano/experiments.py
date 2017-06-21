""" 

Author: Rokas Stanislovas 

"""

class Experiments(object):
    
  """ Add parameters of the experiments here """
    
  exp1 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment'   :   'exp1_CSS',
         'num_data'     :   '500', # 500 samples form training set
         'num_samples'  :   '0',
         'algorithm'    :   'CSS',
         'use_gpu'      :   '1'}
         
  exp2 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment':      'exp2_CSS',
         'num_data'   :     '55000',# a whole training set is used to approx. Z
         'num_samples':     '0',
         'algorithm'  :     'CSS',
         'use_gpu'    :     '1'}
         
  exp3 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment'   :   'exp1_CSS_MF',
         'num_data'     :   '55000',
         'num_samples'  :   '1000',
         'mf_steps'     :   '0',
         'resample'     :   '0',# whether to resample for each minibatch instance if num_samples != 0
         'algorithm'    :   'CSS',
         'use_gpu'    :     '1'}
         
  exp4 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment'   :   'exp2_CSS_MF',
         'num_data'     :   '5000',
         'num_samples'  :   '5000',
         'resample'     :   '1',
         'algorithm'    :   'CSS',
         'use_gpu'    :     '1'}
  
  exp5 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment':      'exp1_CD1',
         'num_steps':       '1',
         'algorithm'  :     'CD1',
         'use_gpu'    :     '1'}
         
  experiments= {exp1['experiment']:exp1,
                exp2['experiment']:exp2,
                exp3['experiment']:exp3,
                exp4['experiment']:exp4,
                exp5['experiment']:exp5}
