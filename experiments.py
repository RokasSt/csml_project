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
         'algorithm'  :     'CSS'}
         
  exp3 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment':      'exp1_CD1',
         'num_steps':       '1',
         'algorithm'  :     'CD1'}       
  
  experiments= {exp1['experiment']:exp1,
                exp2['experiment']:exp2,
                exp3['experiment']:exp3 }
