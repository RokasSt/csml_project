""" 
Author : Rokas Stanislovas 
simple class for testing 
Theano functionality
"""
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg
import numpy as np
from   collections import OrderedDict
import timeit

class Test(object):
    
    def __init__(self, num_vals):
        
        self.x_gibbs  = theano.shared(value = 0, name = "x_gibbs")
        
        self.x_dummy  = theano.shared(value = 0, name = "x_gibbs")
        
        self.num_vals = num_vals
        
        self.indices  = theano.shared(np.arange(self.num_vals))
        
        self.test_shared_x = theano.shared(value = 10, name ="test_shared_x")
        
        np_rand_gen = np.random.RandomState(1234)
        
        self.theano_rand_gen =\
         theano.sandbox.rng_mrg.MRG_RandomStreams(np_rand_gen.randint(2**30))
        
        self.num_vars        = 4
        
        self.num_samples     = 1
        
        self.minibatch_size  = 3
        
        self.mf_params = theano.shared(0.5*np.ones([self.num_vars,self.num_samples]))
        
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
        
    def get_mf_samples(self):
        
        """ function to sample from mean-field distribution """
        
        samples = self.theano_rand_gen.binomial(size= (self.num_vars, self.num_samples),
                                                n   = 1, 
                                                p   = self.mf_params,
                                                dtype=theano.config.floatX)
                                                
        return samples
        
    def get_bin_samples(self):
        
        """ function to sample binary units """
        
        samples = self.theano_rand_gen.binomial(size= (784, 100000),
                                                n   = 1, 
                                                p   = 0.5,
                                                dtype=theano.config.floatX)
                                                
        return samples
        
    def test_get_bin_samples(self):
        
        """ function to test get_bin_samples() """
        
        list_samples, updates  = theano.scan(self.get_bin_samples, n_steps =1)
        
        test_function = theano.function(inputs  = [],
                                        outputs = list_samples,
                                        updates = updates)
        
        t0 = timeit.default_timer()
        
        test_function()
        
        t1 = timeit.default_timer()
        
        print("Sampling took --- %f"%((t1-t0)/60.0))
            
    def test_get_mf_samples(self):
        
        list_samples, updates = theano.scan(self.get_mf_samples, n_steps =2)
        
        #importance_weights =  self.get_mf_evaluations(samples)
        
        samples_reshaped = T.reshape(list_samples,[2,self.num_vars*self.num_samples])
        
        test_funct = theano.function(inputs  = [],
                                     outputs = [list_samples,samples_reshaped],
                                     updates = updates)
        
        samples_out, samples_collected = test_funct()
        
        print(len(samples_out))
        print(samples_out[0].shape)
        print((samples_out[0] == samples_out[1]).all())
        print("")
        print(samples_out)
        print("")
        print(samples_out[0])
        print("")
        print(samples_out[1])
        print("")
        print(samples_collected)
        
if __name__ == "__main__":
    
   to_test = Test(num_vals =4)
    
   to_test.test_functions(use_givens = False)
   
   to_test.test_functions(use_givens = True)
   
   to_test.test_givens_shared_to_shared()
   
   to_test.test_get_mf_samples()
   
   to_test.test_get_bin_samples()
   

