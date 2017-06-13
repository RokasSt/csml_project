"""
Author : Rokas Stanislovas
"""

import subprocess

from experiments import Experiments

### list of experiments to run  ###
list_experiments = ['experiment1']

list_experiments = ['experiment2']

###################################
               
if __name__=="__main__":
    
   all_experiments = Experiments()
    
   all_experiments = all_experiments.experiments
     
   for exp_tag in list_experiments:
        
       command_string = "python train_bm.py "
        
       for parameter in all_experiments[exp_tag].keys():
            
           command_string+="--%s %s "%(parameter,all_experiments[exp_tag][parameter])    
    
       subprocess.call(command_string,shell=True)
    
   