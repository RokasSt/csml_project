""" 
Author : Rokas Stanislovas 
simple class for testing 
Theano functionality
"""
import theano
import theano.tensor as T
import numpy as np
from   collections import OrderedDict

class Test(object):
    
    def __init__(self, num_vals):
        
        self.x_gibbs  = theano.shared(value = 0, name = "x_gibbs")
        
        self.x_dummy  = theano.shared(value = 0, name = "x_gibbs")
        
        self.num_vals = num_vals
        
        self.indices  = theano.shared(np.arange(self.num_vals))
        
        self.test_shared_x = theano.shared(value = 10, name ="test_shared_x")
        
    def update_node(self, target_node):
        
        p_xi_given_x_ =   self.x_gibbs + self.indices[target_node]
        
        samples       =   2*self.x_gibbs + 1 
        
        updates = OrderedDict([(self.x_gibbs, self.x_gibbs +1)])
        
        return (p_xi_given_x_, samples), updates
        
    def full_step(self):
        
        (get_p, get_samples), updates  =\
         theano.scan(self.update_node, 
                     sequences =[T.arange(self.num_vals)])
                     
        return (get_p, get_samples), updates
                     
    def test_functions(self, use_givens):
       
        (p_xi_given_x_, x_samples), updates =\
        theano.scan(self.full_step, n_steps = 1) 
        
        if use_givens:
        
           get_samples = theano.function(inputs  = [],
                                         outputs = [p_xi_given_x_,x_samples], 
                                         givens  = {self.x_gibbs: self.test_shared_x},
                                         updates = updates)
                                      
        else:
            
           get_samples = theano.function(inputs  = [],
                                         outputs = [p_xi_given_x_,x_samples], 
                                         updates = updates)
                                      
        p, samples = get_samples()
        print(self.x_gibbs.get_value())
        print(p[0])
        print(samples[0])
        
    def test_givens_shared_to_shared(self):
        
        get_x_value = theano.function(inputs = [], 
                                      outputs= [self.x_gibbs, self.x_dummy],
                                      givens = {self.x_gibbs: self.test_shared_x,
                                                self.x_dummy: self.test_shared_x})
                                      
        val, val_dummy = get_x_value()
        
        print(val)
        print(val_dummy)
                                      
if __name__ == "__main__":
    
   to_test = Test(num_vals =4)
    
   to_test.test_functions(use_givens = False)
   
   to_test.test_functions(use_givens = True)
   
   to_test.test_givens_shared_to_shared()
   

