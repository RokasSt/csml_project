""" 

Author: Rokas Stanislovas 

"""

class Experiments(object):
    
  """ Add parameters of the experiments here """
    
  exp1 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment':      'exp1_CSS',
         'num_samples':     '100',
         'algorithm'  :     'CSS'}
         
  exp2 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment':      'exp2_CSS',
         'num_samples':     '55000',# a whole training set is used to approx. Z
         'algorithm'  :     'CSS',
         'use_gpu'    :     '1'}
         
  exp3 ={'num_epochs':      '20', #15
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment':      'exp1_CD1',
         'num_steps':       '1',
         'algorithm'  :     'CD1'}
         
  exp4 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment'   :   'exp1_CSS_MF',
         'num_steps'    :   '4',
         'include_all'  :   '1',
         'num_samples'  :   '200',
         'algorithm'    :   'CSS_MF'}
         
  exp5 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment'   :   'exp2_CSS_MF',
         'num_steps'    :   '4',
         'include_all'  :   '0',
         'num_samples'  :   '200',
         'algorithm'    :   'CSS_MF'}
  
  experiments= {exp1['experiment']:exp1,
                exp2['experiment']:exp2,
                exp3['experiment']:exp3,
                exp4['experiment']:exp4,
                exp5['experiment']:exp5}
