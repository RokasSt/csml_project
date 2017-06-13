""" 

Author: Rokas Stanislovas 

"""

class Experiments(object):
    
  """ Add parameters of the experiments here """
    
  exp1 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment':      'experiment1',
         'num_samples':     '50',
         'algorithm'  :     'CSS'}
         
  exp2 ={'num_epochs':      '15',
         'batch_size':      '50',
         'learning_rate':   '0.05',
         'experiment':      'experiment2',
         'num_samples':     '0',
         'algorithm'  :     'CD1'}       
  
  experiments= {exp1['experiment']:exp1,
                exp2['experiment']:exp2 }
