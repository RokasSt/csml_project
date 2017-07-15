""" 
Author: Rokas Stanislovas
MSc Project: Likelihood Approximations
for Energy-Based Models
MSc Computational Statistics and 
Machine Learning
"""

import theano
import theano.tensor as T
import theano.sandbox.rng_mrg
import numpy as np
import cPickle
import Image
import sys
from   plot_utils import make_raster_plots
from   collections import OrderedDict
import timeit
import os

class BoltzmannMachine(object):
    
    """ class to implement fully visible Boltzmann Machine or
    Restricted Boltzmann Machine"""
    
    def __init__(self, 
                 num_vars, 
                 num_hidden,
                 training_inputs = None,
                 algorithm = None,
                 algorithm_dict = None,
                 batch_size = None,
                 use_momentum = None,
                 W0= None, 
                 b0= None, 
                 bhid0 = None,
                 report_p_tilda =False,
                 learn_biases = True,
                 test_mode= False,
                 training = True):
        
        """ Constructor for Fully Visible Boltzmann Machine
        
        num_vars - a number of visible nodes/variables
        
        num_hidden - a number of hidden variables; if greater than zero
        Restricted Boltzmann Machine is implemented.
        
        training_inputs - N x D matrix of training inputs
        
        TODO
        
        """
        
        self.num_vars       = num_vars
        
        self.num_hidden     = num_hidden
        
        self.batch_size     = batch_size
        
        self.algorithm      = algorithm
        
        self.num_samples    = None
        
        self.num_cd_steps   = None
        
        self.data_samples   = None
        
        self.resample       = None
        
        self.mf_steps       = None
        
        self.use_is         = None
        
        self.is_probs       = []
        
        self.learn_biases   = learn_biases
        
        if isinstance(algorithm_dict, dict):
        
           for param in algorithm_dict.keys():
            
               if param == 'resample':
                
                  self.resample = algorithm_dict[param]
               
               if param == 'mf_steps':
                
                  self.mf_steps = algorithm_dict[param]
               
               if param == "num_cd_steps":
                
                  self.num_cd_steps = algorithm_dict[param]
               
               if param == "use_is":
                
                  self.use_is = algorithm_dict[param]
               
               if param == "num_samples":
                
                  self.num_samples = algorithm_dict[param]
               
               if param == "data_samples":
                
                  self.data_samples = algorithm_dict[param]
                  
               if param == "alpha":
                   
                  alpha = algorithm_dict[param]  
                
                  if alpha != None:
                     
                     self.is_probs= (1-alpha)*0.5*np.ones([1,self.num_vars])+\
                                  alpha*np.mean(training_inputs,0)
                                  
                     self.is_probs = \
                     np.asarray(self.is_probs, dtype = theano.config.floatX)
                                                 
        if self.is_probs != []:
           
           if self.resample:
               
              num_times = self.batch_size*self.num_samples
           
              self.is_probs = np.tile(self.is_probs, (num_times, 1))
              
           else:
              
              self.is_probs = np.tile(self.is_probs, (self.num_samples, 1))
              
        ##########
        
        self.use_momentum   = use_momentum
        
        self.report_p_tilda = report_p_tilda
        
        self.side = int(np.sqrt(self.num_vars))
        
        self.np_rand_gen = np.random.RandomState(1234)
        
        self.theano_rand_gen =\
         theano.sandbox.rng_mrg.MRG_RandomStreams(self.np_rand_gen.randint(2**30))
         
        #self.theano_rand_gen =\
         #T.shared_randomstreams.RandomStreams(self.np_rand_gen.randint(2**30))
        
        theano.config.exception_verbosity = 'high'
        
        self.node_indices  = \
        theano.shared(np.arange(self.num_vars), name="node_indices")
        
        self.x               = T.matrix('x')
           
        self.x_tilda         = T.matrix('x_tilda')
        
        if training:
            
           if self.num_hidden ==0:
               
              self.num_x2 = self.num_vars
              
           elif self.num_hidden > 0 :
               
              self.num_x2 = self.num_hidden
           
           self.updates = OrderedDict()
           
           self.N_train = training_inputs.shape[0]
           
           self.train_inputs = theano.shared(np.asarray(training_inputs,
                                          dtype=theano.config.floatX),
                                          borrow= True)
                                          
           self.learning_rate  = T.dscalar('learning_rate')
           
           if use_momentum:
               
              print("Will add momentum term to gradient computations")
              
              self.momentum  = T.dscalar('learning_rate')
              
              self.grad_vec = {}
              
              self.grad_vec['W'] = theano.shared(np.zeros([self.num_vars, self.num_x2],
              dtype = theano.config.floatX), name = 'W_momentum', borrow = True)
                 
              if self.num_hidden > 0:
                  
                 self.grad_vec['bhid'] = theano.shared(np.zeros([self.num_x2],
                 dtype = theano.config.floatX), name = 'b_momentum', borrow = True)
              
              self.grad_vec['b'] = theano.shared(np.zeros([self.num_vars],
              dtype = theano.config.floatX), name = 'b_momentum', borrow = True)
              
           if test_mode:
              
              b_init =self.np_rand_gen.uniform(0,1, num_vars)
    
              W_init =self.np_rand_gen.uniform(0,1, size = (num_vars, num_vars))
              
              # also tested ones
              # b_init = np.ones(num_vars)
    
              # W_init = np.ones([num_vars, num_vars])
              
              self.b_init= np.asarray(b_init, dtype = theano.config.floatX)
        
              self.W_init= np.asarray(W_init, dtype = theano.config.floatX)
              
              self.b = theano.shared(self.b_init, name='b', borrow = False)
        
              self.W = theano.shared(self.W_init, name='W', borrow = False)
              
              print("Initialized with test mode")
              
           else:
                                          
              if W0 is None:
                  
                 if self.num_hidden > 0:
                    
                    W0_init =\
                    self.np_rand_gen.uniform(
                   -4*np.sqrt(6.0/(self.num_vars+self.num_hidden)),\
                    4*np.sqrt(6.0 /(self.num_vars + self.num_hidden)), 
                    size = (num_vars, self.num_hidden)
                    )
                    
                    W0 = np.asarray(W0_init, dtype = theano.config.floatX) 
                    
                 if self.num_hidden == 0:
                     
                    # different W initializations: 
        
                    # W0_init =\
                    # self.np_rand_gen.uniform(-np.sqrt(3.0/(num_vars)),\
                    # np.sqrt(3.0 / (num_vars)), size = (num_vars, num_vars))
                  
                    # W0_init =\
                    # self.np_rand_gen.uniform(-0.00000001,\
                    # 0.00000001, size = (num_vars, num_vars))
                 
                    W0_init = 0.00000001*\
                    self.np_rand_gen.normal(size = (num_vars, self.num_x2)) 
        
                    W0 = np.asarray(W0_init, dtype = theano.config.floatX)
              
                    W0 = (W0 + np.transpose(W0))/2.0
              
                    W0 = W0 - np.diag(np.diag(W0))
        
                 self.W = theano.shared(value= W0, name='W', borrow=True)
                
                 if self.num_hidden == 0:
              
                    test_W = self.W.get_value() 
              
                    assert sum(np.diag(test_W)) == 0.0
              
                    assert (test_W == np.transpose(test_W)).all() == True
              
              else:
                 print("W is initialized with provided array")
                 self.W = theano.shared(value= W0, name='W', borrow=True)
           
              if b0 is None:
        
                 bias_init = np.zeros(num_vars, dtype = theano.config.floatX)
        
                 self.b = theano.shared(value= bias_init, name='b', borrow=True)
           
              else:
                 print("b vector is initialized with provided vector")
                 self.b = theano.shared(value= b0, name='b', borrow=True)
                 
              if bhid0 is None and self.num_hidden > 0:
                 
                 hbias_init = np.zeros(self.num_hidden, dtype = theano.config.floatX)
        
                 self.bhid = theano.shared(value= hbias_init, name='bhid', borrow=True)
                 
              elif (bhid0 != None) and (self.num_hidden > 0):
                 print("bhid vector is initialized with provided vector") 
                 self.bhid = theano.shared(value= bhid0, name='bhid', borrow=True)
           
           self.theta    = [self.W, self.b]
           
           if self.num_hidden > 0 :
               
              self.theta.append(self.bhid)
           
           self.train_set       = set(range(self.N_train))
        
           self.minibatch_set   = T.ivector('minibatch_set')
        
           self.sample_set      = T.ivector('sample_set')
           
           if "CD" in self.algorithm and self.num_hidden ==0:
           
              self.x_gibbs= theano.shared(np.ones([self.batch_size,self.num_vars],
                                          dtype=theano.config.floatX),
                                          borrow = True, name= "x_gibbs")
                                          
           if "CD" in self.algorithm and self.num_hidden > 0:
              
              self.persistent_gibbs =\
              theano.shared(np.ones([self.batch_size,self.num_hidden],
                            dtype=theano.config.floatX),
                            borrow = True, 
                            name= "persistent_gibbs")
              
           if "CSS" in self.algorithm and self.use_is != True:
              
              init_mf_vis = self.np_rand_gen.uniform(0, 1, size =(self.num_vars,1))
              
              init_mf_vis = np.asarray(init_mf_vis, dtype = theano.config.floatX)
              
              self.mf_vis_params = theano.shared(init_mf_vis, 
                                                 name= "mf_vis_params", 
                                                 borrow= True)
                                                
              if self.num_hidden > 0:
                
                 init_mf_hid = \
                 self.np_rand_gen.uniform(0, 1, size =(self.num_hidden,1))
              
                 init_mf_hid = np.asarray(init_mf_hid, 
                                          dtype = theano.config.floatX)
              
                 self.mf_hid_params = theano.shared(init_mf_hid, 
                                                    name= "mf_hid_params", 
                                                    borrow= True)
                  
                                             
    def energy_function(self, x):
    
        """ to compute energy function for fully visible Boltzmann machine
    
        W - D x D matrix of weight variables in theano
    
        b - D-dimensional column vector of bias terms in theano
    
        x - D-dimensional input vector
    
        return:
    
        - xTWx - bT x 
    
        as a symbolic variable to add to the theano computational graph.
    
        """
  
        return -T.dot(T.transpose(x), T.dot(self.W, x)) -\
         T.dot(T.transpose(self.b), x)
         
    def get_h_given_v_samples(self, x):
        
        """ This function infers state of hidden units given visible units
        Original implementation in http://deeplearning.net/tutorial/rbm.html"""
        
        sig_input = T.dot(x, self.W) + self.bhid
         
        sig_output= T.nnet.sigmoid(sig_input)
         
        sample = self.theano_rand_gen.binomial(size= sig_output.shape,
                                               n=1, 
                                               p= sig_output,
                                               dtype=theano.config.floatX)
                                          
        return [sig_input, sig_output, sample]
        
    def get_v_given_h_samples(self, h):
        
        """ This function infers state of visible units given hidden units
        Original implementation in http://deeplearning.net/tutorial/rbm.html"""
        
        sig_input = T.dot(h, T.transpose(self.W)) + self.b
         
        sig_output= T.nnet.sigmoid(sig_input)
         
        sample = self.theano_rand_gen.binomial(size= sig_output.shape,
                                               n=1, 
                                               p= sig_output,
                                               dtype=theano.config.floatX)
                                          
        return [sig_input, sig_output, sample]
        
    def gibbs_step_rbm_hid(self, x):
        '''This function implements one step of Gibbs sampling
           for Restricted Boltzmann Machine starting with hidden units.'''
            
        sig_input1, sig_output1, sample1 = self.get_v_given_h_samples(x)
            
        sig_input2, sig_output2, sample2 = self.get_h_given_v_samples(sample1)
            
        return [sig_input1, sig_output1, sample1,
                sig_input2, sig_output2, sample2]
        
    def gibbs_step_rbm_vis(self, x):
        ''' This function implements one step of Gibbs sampling
            for Restricted Boltzmann Machine starting with visible units.'''
            
        sig_input1, sig_output1, sample1 = self.get_h_given_v_samples(x)
            
        sig_input2, sig_output2, sample2 = self.get_v_given_h_samples(sample1)    
        
        return [sig_input1, sig_output1, sample1,
                sig_input2, sig_output2, sample2]
                
    def free_energy_function(self, x):
        
        """ to compute free energy function for restricted Boltzmann Machine
        of binary stochastic units.
        
        x - N x D matrix of binary inputs """
        
        wx_b = T.dot(x, self.W) + self.bhid
        
        return -T.sum(T.log(1 + T.exp(wx_b)), axis=1) -T.dot(x, self.b)
        
    def add_mf_approximation(self):
        
        """ function to add mf approximation of energy terms in Z"""
        
        if self.resample:
               
           print("Samples will be drawn for each instance in a minibatch")
           [list_mf_samples, weight_term], updates =\
           theano.scan(self.get_mf_samples(data=False), n_steps = self.batch_size)
         
           self.updates.update(updates)
         
           list_mf_samples = theano.gradient.disconnected_grad(list_mf_samples)
        
           weight_term    = theano.gradient.disconnected_grad(weight_term)
        
           weight_term    = T.log(self.num_samples) + weight_term
           
           if self.num_hidden ==0:
        
              approx_Z_mf, updates = \
              theano.scan(lambda i : -self.compute_energy(list_mf_samples[i],\
              self.num_samples),sequences = [T.arange(self.batch_size)])
        
           if self.num_hidden > 0:
               
              approx_Z_mf, updates = \
              theano.scan(lambda i : -self.compute_free_energy(list_mf_samples[i]),\
              sequences = [T.arange(self.batch_size)])
        
           self.updates.update(updates)
        
           weight_term   = T.reshape(weight_term,
                                    [self.batch_size, self.num_samples])
        
           approx_Z_mf = T.reshape(approx_Z_mf,
                                   [self.batch_size, self.num_samples])
                                     
           approx_Z_mf = approx_Z_mf - weight_term
           
        else:
               
           print("Samples will be shared between instances in a minibatch")
              
           mf_samples, weight_term = self.get_mf_samples(data= False)
              
           mf_samples  = theano.gradient.disconnected_grad(mf_samples)
        
           weight_term = theano.gradient.disconnected_grad(weight_term)
        
           weight_term = T.log(self.num_samples) + weight_term
           
           if self.num_hidden == 0:
              
              approx_Z_mf = -self.compute_energy(mf_samples, 
                                                 self.num_samples)
                                                 
           if self.num_hidden > 0:
              
              approx_Z_mf = -self.compute_free_energy(mf_samples)
              
           approx_Z_mf = approx_Z_mf - weight_term
           
        return approx_Z_mf
        
    def add_is_approximation(self):
        
        """ function to add computations of energies and their weights
        in the approximating term in normalizer Z. Samples are obtained
        using uniform importances sampling."""
        
        if self.is_probs == []:
           print("Given importance distribution is uniform")
           weight_term = T.log(self.num_samples) + self.num_vars*T.log(0.5)
           
        else:
           print("Given importance distribution is not uniform")
           weight_term = T.log(self.num_samples)+\
             self.get_importance_evals(T.transpose(self.x_tilda), 
                                       np.transpose(self.is_probs))
                                       
           if self.resample:
               
              weight_term =  T.reshape(weight_term, 
                                       [self.batch_size, self.num_samples])
        
        if self.resample and self.num_hidden ==0:
           
           approx_Z = -self.compute_energy(self.x_tilda, 
                                           self.num_samples*self.batch_size)
           
        elif (not self.resample) and self.num_hidden ==0:
           
           approx_Z = -self.compute_energy(self.x_tilda, self.num_samples)
           
        elif self.resample and self.num_hidden > 0:
            
           approx_Z = -self.compute_free_energy(self.x_tilda)
           
        elif (not self.resample) and self.num_hidden >0:
           
           approx_Z = -self.compute_free_energy(self.x_tilda)
        
        if self.resample:
           
           approx_Z = T.reshape(approx_Z, [self.batch_size, self.num_samples])
           
        approx_Z = approx_Z - weight_term
        
        return approx_Z
        
    def add_complementary_term(self):
        
        """ function to add computations on approximating term of log Z
        This term does not involve training points explicitly."""
        
        approx_Z_mf = None
        
        if self.num_samples > 0:
            
           non_data_samples = True
           
           if self.use_is:
               
              print("Will use importance sampling for Z approximation")
              
              approx_Z = self.add_is_approximation()
              
           else:
           
              print("Will use mean-field sampling for Z approximation")
              
              approx_Z = self.add_mf_approximation()
           
        return approx_Z, non_data_samples
        
    def compute_approx_log_Z(self, data_term, non_data_term, axis= None):
        
        """ function to combine data-specific and non-data-specific 
        energy terms for computating the approximation of log Z. 
        """
        
        if (axis == 1) and (non_data_term != None):
           
           approx_Z = T.concatenate([non_data_term, data_term], axis=1)
              
           max_vals = T.max(approx_Z, axis=1)

           max_vals = T.reshape(max_vals,[self.batch_size,1])
           
           if self.use_is:
              
              max_vals_tiled = T.tile(max_vals,
                                      (1,self.num_samples+self.batch_size))
               
           else:
           
              max_vals_tiled= T.tile(max_vals,(1,self.num_samples+1))
           
           approx_Z = approx_Z - max_vals_tiled
        
           approx_Z = max_vals + T.log(T.sum(T.exp(approx_Z), axis=1))
           
           approx_Z = T.mean(approx_Z)
           
        if (axis == None) and (non_data_term != None):
           
           approx_Z = T.concatenate([non_data_term, data_term])
           
           max_val  = T.max(approx_Z)
           
           ### self. added to enable reporting p_tilda instead of
           ### pseudo likelihood cost
           self.approx_Z = approx_Z - max_val
        
           approx_Z = max_val + T.log(T.sum(T.exp(self.approx_Z)))
           
        if (non_data_term == None) and (axis == None):
           
           approx_Z = data_term
           
           max_val  = T.max(approx_Z)
           
           approx_Z = max_val + T.log(T.sum(T.exp(approx_Z -max_val)))
           
        if (non_data_term == None) and (axis == 1):
            
           max_vals  = T.max(data_term, axis=1)

           max_vals  = T.reshape(max_vals,[self.batch_size,1])
           
           max_vals_tiled  = T.tile(max_vals,(1,self.data_samples+1))
           
           approx_Z = data_term - max_vals_tiled
        
           approx_Z = max_vals + T.log(T.sum(T.exp(approx_Z), axis=1))
           
           approx_Z = T.mean(approx_Z)
           
        return approx_Z
        
    def add_css_approximation(self, minibatch_evals):
        
        """ function to define complementary sum sampling (css) 
        approximation of log Z.
        
        minibatch_evals - minibatch energy evaluations computed with
        self.compute_energy().
        
        """
        
        approx_Z, non_data_samples = self.add_complementary_term()
        
        if self.data_samples == 0:
            
           print("Will use minibatch set only for data term in Z approximation")
           
           approx_Z_data = -minibatch_evals
            
        if self.data_samples == self.N_train:
            
           print("Will explicitly include all training points in Z approximation")
           
           if self.num_hidden == 0:
           
              approx_Z_data = -self.compute_energy(self.x_tilda, self.N_train)
              
           else:
               
              approx_Z_data = -self.compute_free_energy(self.x_tilda)
           
        if (self.data_samples < self.N_train) and (self.data_samples != 0):
            
           print("Will uniformly sample from training set for Z approximation")
           
           if self.num_hidden == 0:
           
              approx_Z_data = -self.compute_energy(self.x_tilda, 
                                          self.batch_size*self.data_samples)
                                          
           else:
               
              approx_Z_data = -self.compute_free_energy(self.x_tilda)
                                        
           approx_Z_data = T.reshape(approx_Z_data, [self.batch_size, self.data_samples])
           
           approx_Z_data = approx_Z_data - T.log(self.data_samples)
           
           minibatch_evals = -T.reshape(minibatch_evals, [self.batch_size,1])
           
           approx_Z_data = T.concatenate([approx_Z_data, minibatch_evals],axis = 1)
           
        if non_data_samples:
           
           if self.resample:
               
              if (self.data_samples == self.N_train) or (self.data_samples ==0):
            
                 approx_Z_data = T.tile(approx_Z_data,(self.batch_size,1))
                 
              approx_Z = self.compute_approx_log_Z(approx_Z_data, approx_Z, axis=1)
              
           else:
              
              if (self.data_samples  < self.N_train) and (self.data_samples != 0):
                 
                 approx_Z = T.tile(approx_Z,(self.batch_size,1))
                 
                 approx_Z = self.compute_approx_log_Z(approx_Z_data, approx_Z, axis=1)
               
              if (self.data_samples == self.N_train) or (self.data_samples == 0):
                 
                 approx_Z = self.compute_approx_log_Z(approx_Z_data, approx_Z)
              
        else:
            
           if self.data_samples == self.N_train:
               
              approx_Z = self.compute_approx_log_Z(approx_Z_data, 
                                                   non_data_term =None)
              
              
           if (self.data_samples < self.N_train) and (self.data_samples !=0):
               
              approx_Z = self.compute_approx_log_Z(approx_Z_data,
                                                   non_data_term = None,
                                                   axis=1)
           
        return approx_Z
        
    def get_importance_evals(self, samples, params):
        
        """ function to get evaluations of log importance distribution"""
        
        evals = T.log(params)*samples + T.log(1.0 - params)*(1- samples)
        
        evals = T.sum(evals, axis=0) # axis= 0: node index, axis=1: nth datum
        
        return evals
        
    def get_mf_samples(self, data = True):
        
        """ function to sample visible units from mean-field distribution"""
        
        if not data:
           
           mf_vals =  T.tile(self.mf_vis_params,(1,self.num_samples))
           
        else:
            
           mf_vals = self.mf_vis_params
        
        samples = self.theano_rand_gen.binomial(size= (self.num_vars,
                                                       self.num_samples),
                                                n   = 1, 
                                                p   = mf_vals,
                                                dtype=theano.config.floatX)
        
        log_q_vals =  self.get_importance_evals(samples, mf_vals)
        
        return T.transpose(samples), log_q_vals
        
    def add_mf_updates(self):
        
        """ function to add mean-field updates"""
        
        if self.num_hidden == 0:
        
           self.mf_updates, _ =\
            theano.scan(lambda i: self.sigmoid_update(self.mf_vis_params,i),
                                         sequences = [T.arange(self.num_vars)])
                                         
        elif self.num_hidden > 0:
            
           self.mf_vis_updates = self.sigmoid_update_vis(self.mf_hid_params)
           
           self.mf_hid_updates = self.sigmoid_update_hid(self.mf_vis_params)
           
           # damp high oscillations:
           self.mf_vis_updates = 0.02*self.mf_vis_params + 0.98*self.mf_vis_updates
           
           self.mf_hid_updates = 0.02*self.mf_hid_params + 0.98*self.mf_hid_updates
           #####
           
    def do_mf_updates(self, num_steps, report = False):
        
        """ function to implement mean-field updates for approximation
        of model (equilibrium) distribution"""
        
        output_vars =[]
        
        if self.num_hidden == 0:
        
           if report:
              
              output_vars.append(T.mean(T.log(self.mf_vis_params)))
           
           update_funct = theano.function(inputs  =[],
                                          outputs = output_vars,
                                          updates = [(self.mf_vis_params,\
                                          self.mf_updates)])
                                       
           for step in range(num_steps):
               if report:
                  avg_log_mf = update_funct() 
                  print("Step %d: average value of MF parameter --- %f"%
                  (step, avg_log_mf[0]))
               else:
                  update_funct()
        
        elif self.num_hidden > 0: 
            
           if report:
               
              output_vars.append(T.mean(T.log(self.mf_vis_params)))
              
              output_vars.append(T.mean(T.log(self.mf_hid_params)))
              
           updates = OrderedDict([(self.mf_vis_params, self.mf_vis_updates),
                                  (self.mf_hid_params, self.mf_hid_updates)])
           
           update_funct = theano.function(inputs  = [],
                                          outputs = output_vars,
                                          updates = updates)
                                          
           for step in range(num_steps):
               if report:
                  avg_log_vis, avg_log_hid = update_funct() 
                  print("Step %d: average value of visible MF parameter --- %f"%
                  (step, avg_log_vis))
                  print("Step %d: average value of hidden MF parameter --- %f"%
                  (step, avg_log_hid))
               else:
                  update_funct()
              
    def compute_energy(self, x, num_terms):
        
        """ function to evaluate energies over a given set of inputs
        for fully visible Boltzmann Machine."""
        
        evals, _ = \
        theano.scan(lambda i: self.energy_function(T.transpose(x[i,:])), \
        sequences = [T.arange(num_terms)] )
        
        return evals
        
    def compute_free_energy(self, x):
        
        """ function to evaluate free energies over a given set of inputs
        for Restricted Boltzmann Machine."""
        
        return self.free_energy_function(x)
        
    def test_compute_energy(self):
        
        """ test compute_energy() """
        
        if self.data_samples < self.N_train:
        
           approx_Z_data = -self.compute_energy(self.x_tilda, 
                                                self.batch_size*self.data_samples)
                                             
        if self.data_samples == self.N_train:
            
           approx_Z_data = -self.compute_energy(self.x_tilda, self.data_samples)
                                             
        input_dict = {self.x_tilda: self.train_inputs[self.sample_set,:]}
                                             
        test_function = theano.function(inputs  = [self.sample_set],
                                        outputs = approx_Z_data,
                                        givens  = input_dict)
                                        
        return test_function
        
    def test_add_complementary_term(self):
        
        """ test add_complementary_term() """
        
        approx_Z, _ = self.add_complementary_term()
        
        test_function = theano.function(inputs=[],
                                        outputs=[approx_Z])
                                        
        return test_function
                                        
    def add_objective(self):
        
        """ function to add model objective for model optimization """ 
        
        if "CSS" in self.algorithm:
            
           if self.num_hidden == 0:
            
              data_term = self.compute_energy(self.x, self.batch_size)
              
           else:
               
              data_term = self.compute_free_energy(self.x)
              
           normalizer_term = self.add_css_approximation(data_term)
               
        if "CD" in self.algorithm and self.num_hidden ==0:
            
           data_term = self.compute_energy(self.x, self.batch_size)
           
           normalizer_term = self.compute_energy(self.x_gibbs, 
                                                 self.batch_size)
           
           normalizer_term = -T.mean(normalizer_term)
           
        if "CD" in self.algorithm and self.num_hidden > 0:
           
           data_term = self.compute_free_energy(self.x)
            
           normalizer_term = self.compute_free_energy(self.rbm_cd_samples)
           
           normalizer_term = -T.mean(normalizer_term)
           
        #  cost is negative log likelihood   
        self.cost = T.mean(data_term) + normalizer_term
        
    def add_cd_samples(self):
        
        """ function to add sampling procedure for CD approximation """ 
        
        if self.num_hidden == 0:
        
           (self.p_xi_given_x_, self.gibbs_samples), self.gibbs_updates =\
           theano.scan(self.gibbs_step_fully_visible, n_steps = self.num_cd_steps)
           
        if self.num_hidden > 0:
           
           if "PCD" in self.algorithm:
              
              init_chain  = self.persistent_gibbs
               
           else:
            
              # positive phase
              _, _, hid_sample = self.get_h_given_v_samples(self.x)

              init_chain = hid_sample
          
           (
            [
                vis_inputs,
                vis_outputs,
                vis_samples,
                hid_inputs,
                hid_means,
                hid_samples
            ],
            updates
           ) = theano.scan(
            self.gibbs_step_rbm_hid,
            outputs_info=[None, None, None, None, None, init_chain],
            n_steps= self.num_cd_steps)
            
           self.updates.update(updates)
            
           self.hid_samples    = hid_samples[-1]
            
           self.rbm_cd_samples = theano.gradient.disconnected_grad(vis_samples[-1])
        
    def get_cd_samples(self): 
        
        """ function to obtain samples for CD or PCD approxmation
        for training fully visible Boltzmann Machine"""
        
        if "PCD" in self.algorithm:
            
           input_vars = []
           
           given_vars = []
           
        else:
           
           input_vars = [self.minibatch_set]
           
           given_vars = {self.x_gibbs: self.train_inputs[self.minibatch_set,:]} 
           
        get_samples = theano.function(inputs  = input_vars,
                                      outputs = [self.p_xi_given_x_[-1], 
                                                 self.gibbs_samples[-1]
                                                 ], 
                                      givens  = given_vars,
                                      #start the chain at the data distribution
                                      updates = self.gibbs_updates)
                                         
        return get_samples
                                         
    def add_grad_updates(self):
        
        """ Compute and collect gradient updates to dictionary """
        
        gradients = T.grad(self.cost, self.theta)
        
        for target_param, grad in zip(self.theta, gradients):
            
            if target_param.name =="W" and self.num_hidden ==0:
                
               grad = grad - T.diag(T.diag(grad)) # no x i - xi connections
               # for all i = 1, ..., D
               
            if target_param.name =="b" and self.learn_biases == False:
               print("Will not learn bias terms")
               pass
            
            elif target_param.name =="bhid" and self.learn_biases == False:
               print("Will not learn bias terms")
               pass
            
            else:
               
               if self.use_momentum:
                  
                  # alternative definition (mostly seen):
                  #g_tilda = self.momentum*self.grad_vec[target_param.name] - \
                  #T.cast(self.learning_rate, dtype = theano.config.floatX)*grad
                  #self.updates[target_param] = target_param + g_tilda
               
                  g_tilda = self.momentum*self.grad_vec[target_param.name] - \
                  (1-self.momentum)*grad
                
                  self.updates[target_param] = target_param +\
                  T.cast(self.learning_rate, dtype = theano.config.floatX)*g_tilda
               
                  # store g_tilda for next iteration:
                  self.updates[self.grad_vec[target_param.name]] = g_tilda
               
               else:
               
                  self.updates[target_param] = target_param -\
                  T.cast(self.learning_rate, dtype = theano.config.floatX)*grad
               
        if ("PCD" in self.algorithm) and self.num_hidden > 0:
           
           self.updates[self.persistent_gibbs] = self.hid_samples
           
    def select_data(self, minibatch_set):
        
        """ function to select samples for css approximation
        with uniform sampling
         """
        
        minibatch_size = len(minibatch_set)
        
        assert minibatch_size == self.batch_size
        
        ##  compl_inds :: complementary set of training points
        ##  for now, complementary set is computed jointly;
        ##  uses naive approach, uniform sampling
        
        if self.data_samples < self.N_train:
            
           data_samples = []
           
           for i in range(minibatch_size):
               
               compl_inds = list(self.train_set - set([minibatch_set[i]]))
            
               s = np.random.choice(compl_inds,
                                    self.data_samples, 
                                    replace=False)
        
               data_samples.extend(list(s))
        
           assert len(data_samples) == self.data_samples*minibatch_size
           
        if self.data_samples == self.N_train:
            
           data_samples = list(self.train_set)
        
        return data_samples  
        
    def add_approximate_likelihoods(self):
        
        """ function to define approximate likelihoods (p_tilda) 
        for progress tracking"""
        
        p_tilda = T.exp(self.approx_Z)
        
        self.p_tilda = p_tilda/ T.sum(p_tilda)
        
    def add_pseudo_cost_measure(self):
        
        """adds stochastic approximation to the cost objective;
        the functional form is similar to the Gibbs sampling update
        for the fully visible Boltzmann Machine. See
        http://deeplearning.net/tutorial/rbm.html#rbm for
        original implementation.
        """ 
        
        node_index = theano.shared(value=0, name='node_index')

        x_bin = T.round(self.x)
        
        # flip node x_i of matrix x_bin and preserve all other bits x_{\i}
        # Equivalent to x_bin[:, node_index] = 1-x_bin[:, node_index], but assigns
        # the result to x_bin_flip_i, instead of working in place on x_bin
        x_bin_flip_i = T.set_subtensor(x_bin[:, node_index], 1- x_bin[:, node_index])
        
        if self.num_hidden ==0:
        
           fe_x_bin  = self.compute_energy(x_bin, self.batch_size)
        
           fe_x_bin_flip_i = self.compute_energy(x_bin_flip_i, self.batch_size)
           
        else:
            
           fe_x_bin  = self.compute_free_energy(x_bin)
        
           fe_x_bin_flip_i = self.compute_free_energy(x_bin_flip_i)
        
        #  mean(input dimension * log P ( xi | {x -i} ):
        self.pseudo_cost = T.mean(self.num_vars *\
         T.log(T.nnet.sigmoid(fe_x_bin_flip_i- fe_x_bin)))

        # increment bit_i_idx % number as part of updates
        self.updates[node_index] = (node_index + 1) % self.num_vars
        
    def optimization_step(self):
        
        """ function to define a theano function which implements
        a single learning step
        """
        
        if self.algorithm =="CSS":
            
           if self.data_samples > 0:
            
              input_dict = {
               self.x      : self.train_inputs[self.minibatch_set,:],
               self.x_tilda: self.train_inputs[self.sample_set,:]
              }
           
              var_list = [self.sample_set, self.minibatch_set]
              
           elif (self.data_samples == 0) and self.use_is:
               
              input_dict = {
               self.x      : self.train_inputs[self.minibatch_set,:]
              }
           
              var_list = [self.x_tilda, self.minibatch_set]
              
           elif (self.data_samples ==0) and (not self.use_is):
              
              input_dict = {
               self.x      : self.train_inputs[self.minibatch_set,:]
              }
           
              var_list = [self.minibatch_set] 
              
        if "CD" in self.algorithm:
            
           input_dict = {
            self.x  : self.train_inputs[self.minibatch_set,:],
            } 
            
           var_list = [self.minibatch_set]
           
        var_list.append(self.learning_rate)
        
        if self.use_momentum:
            
           var_list.append(self.momentum)
           
        output_vars = [self.pseudo_cost]
        
        if self.report_p_tilda:
            
           output_vars.append(self.p_tilda)
           
        else:
            
           output_vars.append(theano.shared(0))
        
        opt_step = theano.function(inputs  = var_list,
                                   outputs = output_vars,
                                   updates = self.updates,
                                   givens  = input_dict,
                                   on_unused_input='warn')
        
        return opt_step
         
    def sigmoid_update(self, x, var_index):
        
        """ function to compute the fixed point update for the Gibbs step
        for fully visible Boltzmann Machine."""
        
        sigmoid_activation = self.b[self.node_indices[var_index]] +\
         2*(T.dot(self.W[self.node_indices[var_index],:],x) - 
         self.W[self.node_indices[var_index],var_index]*x[var_index,:])
         
        return T.nnet.sigmoid(sigmoid_activation)
        
    def sigmoid_update_vis(self, x):
        
        """ function to compute the fixed point updates for the mean-field
        parameters of visible units of Restricted Boltzmann Machine """
        
        sigmoid_activation = T.reshape(self.b,[self.num_vars,1]) + T.dot(self.W, x)
        
        return T.nnet.sigmoid(sigmoid_activation)
        
    def sigmoid_update_hid(self,x):
        
        """ function to compute the fixed point updates for the mean-field
        parameters of hidden units of Restricted Boltzmann Machine """
        
        sigmoid_activation = T.reshape(self.bhid, [self.num_hidden,1]) +\
        T.dot(T.transpose(self.W),x)
        
        return T.nnet.sigmoid(sigmoid_activation)
        
    def gibbs_update_node(self, target_node):
        
        """ Gibbs sampling update for a target node for fully
        visible Boltzmann Machine
        """
        
        p_xi_given_x_ = self.sigmoid_update(T.transpose(self.x_gibbs),target_node)
        
        samples = self.theano_rand_gen.binomial(size = tuple(p_xi_given_x_.shape),
                                                n    = 1, 
                                                p    = p_xi_given_x_,
                                                dtype=theano.config.floatX)
        
        x_gibbs_update= T.set_subtensor(self.x_gibbs[:, target_node], samples)
        
        updates = OrderedDict([(self.x_gibbs, x_gibbs_update)])
        
        return (p_xi_given_x_, samples), updates
         
    def gibbs_step_fully_visible(self, ordered = True):
        
        """   
        Function to inplement a Gibbs step for fully
        visible Boltzmann Machine. 
        """
        
        if ordered:
            
           seq_var = T.arange(self.num_vars)
           
        else:
            
           seq_var = self.theano_rand_gen.permutation(n=self.num_vars)
        
        (get_p, get_samples), updates  =\
         theano.scan(self.gibbs_update_node, sequences = [seq_var])
        
        return (get_p, get_samples), updates
        
    def add_graph(self):
        
        """ function to build a Theano computational graph """
        
        self.cd_sampling = None
        
        if "CD" in self.algorithm:

           self.add_cd_samples()
           
           if self.num_hidden ==0:
              
              self.cd_sampling = self.get_cd_samples()
           
        if "CSS" in self.algorithm and self.mf_steps > 0: 
        
           self.add_mf_updates()
           
        self.add_objective()

        self.add_grad_updates()  
        
        if self.report_p_tilda:
            
           self.add_p_tilda()
    
        self.add_pseudo_cost_measure()

        self.optimize = self.optimization_step()
        
    def test_pseudo_likelihood(self, test_inputs, num_steps):
        
        """ function to test pseudo_likelihood with trained Boltzmann Machine"""
        
        self.updates = OrderedDict()
        
        self.add_pseudo_cost_measure()
        
        cost_estimate = []
        
        get_measure = theano.function(inputs = [self.x],
                                      outputs= self.pseudo_cost,
                                      updates= self.updates)
                                      
        for step in range(num_steps):
            
             cost_estimate.append(get_measure(test_inputs))
             
        cost_estimate = sum(cost_estimate)/float(num_steps)
        
        return cost_estimate
                                      
    def test_p_tilda(self, test_inputs, random_inputs, training):
        
        """ function to test p_tilda values with trained Boltzmann Machine"""
        
        self.batch_size      = test_inputs.shape[0]
        
        self.num_samples     = random_inputs.shape[0]
        
        self.add_p_tilda(training = training)
        
        var_list = [self.x, self.x_tilda]
        
        get_p_tilda = theano.function(inputs = var_list,
                                      outputs= self.p_tilda)
                                     
        probs = get_p_tilda(test_inputs, random_inputs)
        
        si = self.batch_size+self.np_rand_gen.choice(self.num_samples, 10, False)
        
        return probs[0:self.batch_size], probs[si]
        
    def relative_likelihood(self):
    
        """ function to compute relative, unnormalized likelihood 
        of given examples"""
        
        if self.num_hidden == 0:
        
           return T.exp(-self.compute_energy(self.x, self.batch_size))
           
        if self.num_hidden > 0:
            
           return T.exp(-self.compute_free_energy(self.x))
        
    def test_relative_probability(self, inputs, trained= True):
        
        """ function to test relative, unnormalized likelihood 
        of given examples"""
        
        self.batch_size = inputs.shape[0]
        
        prob_op = self.relative_likelihood()
        
        test_function = theano.function(inputs=[self.x],
                                        outputs=[prob_op])
                                        
        probs = test_function(inputs)
        
        if trained:
           print("Relative likelihoods of training examples:")
        else:
           print("Relative likelihoods of random examples:")
           
        print(probs)
        
    def add_p_tilda(self, training = True):
        
        """ function to compute p_tilda for explicit gradient computations"""
         
        if self.num_hidden == 0:
        
           minibatch_energies = -self.compute_energy(self.x, self.batch_size)
           
        if self.num_hidden > 0:
            
           minibatch_energies = -self.compute_free_energy(self.x)
           
        if self.use_is and training:
        
           sample_energies = self.add_is_approximation()
           
        elif (not self.use_is) and training:
            
           sample_energies = self.add_mf_approximation()
           
        elif (not training):
            
           sample_energies = self.add_is_approximation()
        
        all_energies = T.concatenate([minibatch_energies, sample_energies])
        
        max_val  = T.max(all_energies)
              
        approx_Z = all_energies - max_val
        
        p_tilda = T.exp(approx_Z)
        
        self.p_tilda = p_tilda/ T.sum(p_tilda)
        
    def xn_xn_prod(self,x_n):
        
        """ function to add computation of x(n)^T x(n);
        x_n - is 1 x D vector """
        
        x_n_tiled =T.tile(x_n,(self.num_vars,1))
        
        return T.transpose(x_n_tiled)*x_n_tiled
    
    def test_grad_computations(self, samples, training_points):
        
        """ function to test gradient computations explicitly
        (implementation 2) """
        
        self.add_p_tilda(training = False)
        
        do_updates = OrderedDict()
        
        self.b.set_value(self.b_init)
        
        self.W.set_value(self.W_init)
        
        gradW = theano.shared(np.zeros([self.num_vars,self.num_vars]))
        
        gradb = theano.shared(np.zeros([self.num_vars]))
        
        [gradW, gradb], updates =\
         theano.scan(lambda i, gradW, gradb: [gradW+ \
        (1.0-self.batch_size*self.p_tilda[i])\
        *self.xn_xn_prod(self.x[i,:]),
        gradb+ \
        (1.0-self.batch_size*self.p_tilda[i])\
        *self.x[i,:]],
        outputs_info =[gradW, gradb],
        sequences =[T.arange(self.batch_size)])
        
        gradW = gradW[-1]
        
        gradb = gradb[-1]
        
        do_updates.update(updates)
        
        [gradW, gradb], updates = \
        theano.scan(lambda i, gradW, gradb: [gradW - \
        self.batch_size*self.p_tilda[self.batch_size+i]*\
        self.xn_xn_prod(self.x_tilda[i,:]),
        gradb-self.batch_size*self.p_tilda[self.batch_size+i]*\
        self.x_tilda[i,:]],
        outputs_info =[gradW, gradb],
        sequences =[T.arange(self.num_samples)])
        
        gradW = gradW[-1] /self.batch_size
        
        gradb = gradb[-1] /self.batch_size
        
        gradW = gradW - T.diag(T.diag(gradW)) # no recurrent connections
        
        do_updates.update(updates)
        
        ## ML objective log likelihood (ascend gradient)
        ## the first, more efficient implementation uses the cost
        ## objective which is negative of the log likelihood.
        
        do_updates.update([(self.W, self.W + self.learning_rate*gradW)])
        
        do_updates.update([(self.b, self.b + self.learning_rate*gradb)])
        
        input_dict = {self.x: self.train_inputs[self.minibatch_set,:]}
           
        var_list = [self.x_tilda, self.minibatch_set, self.learning_rate]
        
        test_grads = theano.function(inputs = var_list,
                                     outputs= [],
                                     updates= do_updates,
                                     givens = input_dict,
                                     on_unused_input='warn')
                                     
        test_grads(samples, training_points)
        
    def log_noise_model(self, x, y, pflip):
        
        """ function to compute the log likelihood of the noise
        model for the task of reconstructing noisy images """
        
        return np.log(pflip)*np.sum(x != y) + np.log(1.0-pflip)*np.sum(x == y)
        
    def reconstruct_noisy(self,
                          num_iters,
                          correct_images,
                          recon_images,
                          noisy_images,
                          pflip):
                                
        """ function to reconstruct images from noisy images """
        
        N = recon_images.shape[0]
    
        if self.num_hidden == 0:
        
           tE = -self.compute_energy(self.x, 1)
           
        elif self.num_hidden > 0:
            
           tE = -self.compute_free_energy(self.x)
    
        do_computation = theano.function(inputs  =[self.x], outputs = tE)
        
        for xi  in range(N):
            
            print("Reconstructing test image --- %d"%xi)
            
            E_model = do_computation([noisy_images[xi,:]])
            
            E_noise = self.log_noise_model(noisy_images[xi,:],
                                           noisy_images[xi,:],
                                           pflip)
            
            Emax = E_model + E_noise
            
            for iter_ind in range(num_iters):
                
                permuted_nodes = list(np.random.permutation(self.num_vars))
        
                for d in permuted_nodes:
                    
                    image_delta    = np.copy(recon_images[xi,:])
                    
                    image_delta[d] = 1 - image_delta[d]
                    
                    E_model = do_computation([image_delta])
                    
                    E_noise = self.log_noise_model(noisy_images[xi,:],
                                                   image_delta,
                                                   pflip)
                    
                    E_delta = E_model + E_noise
                    
                    if E_delta > Emax:
                       
                       recon_images[xi, d] = image_delta[d]
                        
                       Emax = E_delta
                       
        return recon_images
        
    def reconstruct_missing(self, 
                            num_iters, 
                            recon_images, 
                            which_pixels,
                            test_mode = False):
    
        """ function to reconstruct images with missing pixels """
    
        N = recon_images.shape[0]
    
        if self.num_hidden == 0:
        
           tE = -self.compute_energy(self.x, 1)
           
        elif self.num_hidden > 0:
            
           tE = -self.compute_free_energy(self.x)
    
        do_computation = theano.function(inputs  =[self.x], outputs = tE)
        
        for xi  in range(N):
            
            print("Reconstructing test image --- %d"%xi)
            
            if test_mode:
               num_checked =0
               check_counter = int(0.5*np.sum(which_pixels[xi,:] == True))
        
            for iter_ind in range(num_iters):
        
                permuted_nodes = list(np.random.permutation(self.num_vars))
        
                for d in permuted_nodes:
            
                    if which_pixels[xi,d]:
                    
                       recon_images[xi, d] = 1
                   
                       E1 = do_computation([recon_images[xi, :]])
                   
                       recon_images[xi, d] = 0
                   
                       E0 = do_computation([recon_images[xi, :]])
                   
                       if E1>E0:
                          recon_images[xi, d] = 1
                       else:
                          recon_images[xi, d] = 0
                          
                       if test_mode:
                          num_checked +=1
                       if test_mode and num_checked <= check_counter:
                          break
        return recon_images
        
    def train_model(self, 
                    num_epochs, 
                    learning_rate, 
                    momentum, 
                    num_iters,
                    report_pseudo_cost,
                    save_every_epoch,
                    report_step,
                    report_p_tilda,
                    report_w_norm,
                    exp_path,
                    test_gradients = False):
                    
        """ function to carry out training of Botlzmann Machine"""
    
        if report_p_tilda:
        
           p_t_i = 0
       
           p_tilda_all = np.zeros(
           [num_epochs*num_iters//report_step,self.batch_size]
           )
       
        else:
        
           p_tilda_all = []
       
        pseudo_losses = []
        
        if report_w_norm:
            
           w_norms = np.zeros(num_epochs*num_iters//report_step)
           
           w_i = 0
           
        else:
            
           w_norms = []
       
        start_train_time = timeit.default_timer()
    
        for epoch_index in range(num_epochs):
        
            epoch_time0 = timeit.default_timer()
    
            perm_inds = self.np_rand_gen.permutation(self.N_train)
    
            #put different learning_rate rules (per epoch) for now here:
        
            #lrate_epoch = learning_rate
    
            lrate_epoch = (1.0/(1+epoch_index/100.))*learning_rate/self.batch_size
      
            # lrate_epoch = (0.9**epoch_index)*learning_rate  # 0.99
    
            # lrate_epoch  = learning_rate
    
            if self.use_momentum:
    
               momentum_epoch = momentum
    
            print("Learning rate for epoch %d --- %f"%(epoch_index,lrate_epoch))
        
            if report_pseudo_cost:
        
               avg_pseudo_cost_val = []
    
            for i in range(num_iters):
        
                iter_start_time = timeit.default_timer()
    
                minibatch_inds = perm_inds[self.batch_size*i:self.batch_size*(i+1)]
            
                if self.algorithm =="CSS":
                    
                   if (self.num_samples > 0) and (self.use_is != True):
               
                      if self.mf_steps > 0:
                         print("Updating MF parameters ...")
                         mf_t0 = timeit.default_timer()
              
                         self.do_mf_updates(num_steps = self.mf_steps)
              
                         mf_t1 = timeit.default_timer()
                         print("%d steps of MF updates took --- %f minutes"%
                         (self.mf_steps,(mf_t1 -mf_t0)/60.0))
               
                   if self.data_samples > 0:
               
                      sampled_indices = bm.select_data(minibatch_inds)
              
                      sampling_var = sampled_indices
                                               
                   if self.data_samples ==0 and self.use_is:
               
                      if self.resample and self.is_probs ==[]:
                 
                         is_samples = self.np_rand_gen.binomial(n=1,p=0.5, 
                         size = (self.num_samples*self.batch_size, self.num_vars))
                 
                      elif (not self.resample) and self.is_probs == []:
                 
                         is_samples = self.np_rand_gen.binomial(n=1, p=0.5, 
                         size = (self.num_samples, self.num_vars))
                         
                      elif self.resample and self.is_probs != []:
                         
                         is_samples = self.np_rand_gen.binomial(n=1, p= self.is_probs, 
                         size = (self.num_samples*self.batch_size, self.num_vars)) 
                         
                      elif (not self.resample) and self.is_probs != []:
                          
                         is_samples = self.np_rand_gen.binomial(n=1, p= self.is_probs, 
                         size = (self.num_samples, self.num_vars)) 
                         
                      sampling_var = np.asarray(is_samples, 
                                                dtype = theano.config.floatX)
              
                   if test_gradients:
                      t0 = timeit.default_timer()
                      approx_cost, p_tilda = self.optimize(sampling_var, 
                                                     list(minibatch_inds),
                                                     lrate_epoch)
                                                     
                      t1 = timeit.default_timer()  
                      print("Gradient computation with implementation 1 took"+\
                      " --- %f minutes"%((t1 - t0)/60.0))
                      W_implicit = np.asarray(self.W.get_value())
                      b_implicit = np.asarray(self.b.get_value())
                      t0 = timeit.default_timer()
                      self.test_grad_computations(is_samples, list(minibatch_inds))
                      t1 = timeit.default_timer()
                      print("Gradient computation with implementation 2 took "+\
                      "--- %f minutes"%((t1 - t0)/60.0))
                      W_explicit = np.asarray(self.W.get_value())
                      b_explicit = np.asarray(self.b.get_value())
                      print("Equivalence of W updates in two implementations:")
                      print((np.round(W_implicit,12) == np.round(W_explicit,12)).all())
                      print("Equivalence of b updates in two implementations:")
                      print((np.round(b_implicit,12) == np.round(b_explicit,12)).all())
                      sys.exit()
           
                   if self.use_momentum and self.use_is:
              
                      approx_cost, p_tilda = self.optimize(sampling_var, 
                                                  list(minibatch_inds),
                                                  lrate_epoch,
                                                  momentum_epoch)
              
                   elif (not self.use_momentum) and self.use_is:
              
                      approx_cost, p_tilda = self.optimize(sampling_var, 
                                                  list(minibatch_inds),
                                                  lrate_epoch) 
                                              
                   elif self.use_momentum and (not self.use_is):
               
                      approx_cost, p_tilda = self.optimize(list(minibatch_inds),
                                                           lrate_epoch,
                                                           momentum_epoch)
                                              
                   elif (not self.use_momentum) and (not self.use_is):
              
                      approx_cost, p_tilda = self.optimize(list(minibatch_inds),
                                                           lrate_epoch) 
            
                if "CD" in self.algorithm:
           
                   if self.num_hidden ==0:
                        
                      if "PCD" in self.algorithm:
                          
                         mf_sample, cd_sample = self.cd_sampling()
           
                         self.x_gibbs.set_value(np.transpose(cd_sample))
                         
                      else:
                         ### "CD" 
                         mf_sample, cd_sample =\
                          self.cd_sampling(list(minibatch_inds))
           
                         self.x_gibbs.set_value(np.transpose(cd_sample))
                         
                   if self.use_momentum:
                         
                      approx_cost, p_tilda = self.optimize(list(minibatch_inds),
                                                           lrate_epoch,
                                                           momentum_epoch)
                                                           
                   else:
                       
                      approx_cost, p_tilda = self.optimize(list(minibatch_inds),
                                                           lrate_epoch)
        
                avg_pseudo_cost_val.append(approx_cost)
                
                if report_pseudo_cost:
        
                   if i % report_step ==0:
            
                      print('Training epoch %d ---- Iter %d ----'%(epoch_index, i)+\
                      ' pseudo cost value: %f'%approx_cost)
           
                if report_p_tilda:
                   
                   if i % report_step == 0:
               
                      print("p_tilda values for training examples:")
                      print(p_tilda[0:self.batch_size])
                      print("sum of these values:")
                      print(np.sum(p_tilda[0:self.batch_size]))
              
                      p_tilda_all[p_t_i,:] = p_tilda[0:self.batch_size]
              
                      p_t_i +=1
                         
                if report_w_norm: 
                       
                   curr_w = self.W.get_value()  
                       
                   w_norms[w_i] = np.sum(np.multiply(curr_w, curr_w))
                   
                   w_i +=1
                         
                iter_end_time = timeit.default_timer()
        
                print('Training iteration took %f minutes'%
                ((iter_end_time - iter_start_time) / 60.))
            
            if report_pseudo_cost:
        
               avg_pseudo_cost_val = np.mean(avg_pseudo_cost_val)
    
               pseudo_losses.append(avg_pseudo_cost_val)
        
            print('Training epoch %d ---- average pseudo cost value: %f'
            %(epoch_index, avg_pseudo_cost_val))
    
            epoch_time1 = timeit.default_timer()
    
            print ('Training epoch took %f minutes'%((epoch_time1 - epoch_time0)/60.))
            
            if save_every_epoch:
    
               self.save_model_params(os.path.join(exp_path,"TRAINED_PARAMS_END.model"))
        
        end_train_time = timeit.default_timer()
    
        training_time = (end_train_time - start_train_time)/60.0

        print('Training process took %f minutes'%training_time)
        
        self.save_model_params(os.path.join(exp_path,"TRAINED_PARAMS_END.model"))
    
        return p_tilda_all, pseudo_losses, training_time, w_norms
        
    def sample_from_mf_dist(self, 
                            num_chains, 
                            num_samples,
                            num_steps,
                            save_to_path,
                            test_inputs    = None,
                            save_mf_params = True):
        
        """ function to sample from mean-field approximation of trained
        fully visible Boltzmann Machine."""
        
        self.num_samples = num_chains
        
        if type(test_inputs) is  np.ndarray:
            
           print("Will initialize MF parameters for visible units"+\
           " with input images")
           get_points = self.np_rand_gen.choice(test_inputs.shape[0], 
                                                num_chains, 
                                                False)
           
           init_mf_vis = test_inputs[get_points,:]
           
           init_with_images = True
           
           images = np.zeros([num_chains*num_samples+num_chains, self.num_vars])
        
           images[0:num_chains,:] = init_mf_vis
           
           init_mf_vis = np.transpose(init_mf_vis)
            
        else:
            
           images = np.zeros([num_chains*num_samples, self.num_vars])
           
           print("Will initialize MF parameters for visible units"+\
            " with uniform distribution")
           init_mf_vis = self.np_rand_gen.uniform(0, 
                                                  1, 
                                                  size =(self.num_vars, num_chains))
                                                  
           init_with_images = False
              
        init_mf_vis = np.asarray(init_mf_vis, dtype = theano.config.floatX)
              
        self.mf_vis_params = theano.shared(init_mf_vis, 
                                           name= "mf_vis_params", 
                                           borrow= True)
        
        if self.num_hidden > 0:
           
           print("Will initialize MF parameters for RBM hidden units"+\
           " with uniform distribution")
           
           init_mf_hid = self.np_rand_gen.uniform(0,
                                                  1, 
                                                  size = (self.num_hidden, num_chains))
              
           init_mf_hid = np.asarray(init_mf_hid, dtype = theano.config.floatX)
              
           self.mf_hid_params = theano.shared(init_mf_hid, 
                                              name= "mf_hid_params", 
                                              borrow= True)
           
        self.add_mf_updates()
        
        self.do_mf_updates(num_steps = num_steps, report = True)
        
        if save_mf_params:
            
           split_path   = os.path.split(save_to_path)
           
           mf_file = os.path.join(split_path[0], "MF_PARAMS.model")
           
           print("Saving MF parameters to %s"%mf_file)
           
           mf_file = file(mf_file, 'wb')
           
           cPickle.dump(self.mf_vis_params, 
                        mf_file, 
                        protocol=cPickle.HIGHEST_PROTOCOL)
        
           mf_file.close()
        
        mf_samples, sample_probs = self.get_mf_samples(data = True)
        
        get_samples = theano.function(inputs  = [],
                                      outputs = [sample_probs, 
                                                 mf_samples,
                                                 self.mf_vis_params])
                                      
        print("Sampling...")
        for ind in range(num_samples):
            
            p_out, samples_out, mf_vals = get_samples()
            
            images[num_chains*ind:num_chains*(ind+1),:] = samples_out

        make_raster_plots(images, 
                          num_samples, 
                          num_chains, 
                          reshape_to = [self.side, self.side], 
                          save_to_path = save_to_path,
                          init_with_images = init_with_images)    
    
    def sample_from_bm(self,
                       num_chains, 
                       num_samples,
                       num_steps,
                       save_to_path,
                       num_burn_in,
                       test_inputs   = None,
                       print_p_tilda = False,
                       print_gibbs   = False):
        
        """ function to generate images from trained 
        Boltzmann Machine (fully visible).
        """
        
        if type(test_inputs) is  np.ndarray:
            
           print("Will initialize gibbs chains with dataset images\n")
           
           num_test_examples = test_inputs.shape[0]
           
           self.test_inputs = theano.shared(np.asarray(test_inputs,
                                         dtype=theano.config.floatX),
                                         borrow= True) 
                                         
           select_examples = np.random.choice(num_test_examples, 
                                              num_chains, 
                                              replace=False)
        
           init_chains =  np.asarray(
              self.test_inputs.get_value(borrow=True)[select_examples,:],
              dtype=theano.config.floatX)
        
        else:
        
           print("Will initialize gibbs chains with random images\n")
           init_chains = self.np_rand_gen.binomial(n=1,p=0.5, 
           size = (num_chains, self.num_vars))
        
        images = np.zeros([num_chains*num_samples+num_chains, self.num_vars])
        
        images[0:num_chains,:] = init_chains
        
        theano.config.exception_verbosity = 'high'
        
        self.x_gibbs = theano.shared(init_chains, name= "x_gibbs")
        
        if self.num_hidden > 0:
           print("Running gibbs chains for RBM ...\n")
           
           (
            [ _,
              _,
              _,
              x_inputs,
              p_xi_given_x_,
              x_samples
            ],
            updates
           ) = theano.scan(
           self.gibbs_step_rbm_vis,
           outputs_info=[None, None, None, None, None, self.x_gibbs],
           n_steps= num_steps)
           
           output_vars = [p_xi_given_x_[-1], x_samples[-1]]
           
           updates.update({self.x_gibbs: x_samples[-1]})
           
        else:
            
           print("Running gibbs chains for BM ...\n")
           
           (p_xi_given_x_, x_samples), updates =\
           theano.scan(self.gibbs_step_fully_visible, n_steps = num_steps)
        
           output_vars = [p_xi_given_x_[num_burn_in:],
                          x_samples[num_burn_in:]]
                          
           take_step = (num_steps - num_burn_in) // self.num_vars 
           
           if take_step == 0:
               
              take_step = 1
              
        get_samples = theano.function(inputs  = [],
                                      outputs = output_vars, 
                                      updates = updates)
        
        for ind in range(num_samples):
            
            p_all, samples_all = get_samples()
            
            if num_steps != 1 and self.num_hidden == 0:
               
               p_out, samples_out = self.assemble_image(p_all, 
                                                        samples_all,
                                                        num_chains,
                                                        step = take_step)
                                                        
            elif num_steps ==1 and self.num_hidden == 0:
                
               p_out       = p_all[-1]
               
               samples_out = samples_all[-1]
               
            elif self.num_hidden > 0:
               
               p_out       = p_all
               
               samples_out = samples_all
               
            if self.num_hidden == 0:
                
               p_out = np.transpose(p_out) 
            
            # without resetting the chains are persistent for
            # fully visible Boltzmann Machines
            # (self.x_gibbs are modified continuously)
            # self.x_gibbs.set_value(init_chains)
            
            print("Sample %d -- max pixel activations for %d gibbs chains:"%
            (ind, num_chains))
            print(np.max(p_out, axis= 1))
            print("")
            
            if print_gibbs:
               self.print_gibbs_conditionals(p_vals = p_all)
               
            if print_p_tilda:   
               is_samples = self.np_rand_gen.binomial(n=1, 
                                                      p=0.5, 
                                                      size =(10000, self.num_vars))
   
               gibbs_p_tilda, rand_p_tilda = \
               self.test_p_tilda(np.transpose(samples_out), 
                                 is_samples,
                                 training = False)
            
               print("p_tilda values for gibbs samples:")
               print(gibbs_p_tilda)
               print("")
               print("p_tilda values for randomly chosen importance samples:")
               print(rand_p_tilda)
               print("")
               
            images[num_chains*(ind+1):num_chains*(ind+2),:] = np.round(p_out)
        
        make_raster_plots(images, 
                          num_samples, 
                          num_chains, 
                          reshape_to = [self.side, self.side], 
                          save_to_path = save_to_path)    
                          
    def assemble_image(self, mf_vals, sample_vals, num_chains, step):
    
        """ function to assemble sample image from consecutive
        gibbs samples which are highly correlated."""
        
        img = np.zeros([self.num_vars, num_chains])
        
        samples = np.zeros([self.num_vars, num_chains])
        
        num_full_steps = len(mf_vals)
        
        node = 0
        
        for gibbs_step in range(num_full_steps):
            
            if node == self.num_vars:
                
               break
            
            if gibbs_step % step == 0:
                
               if node <= self.num_vars:
                
                  img[node,:] = mf_vals[gibbs_step][node]
                  
                  samples[node,:] = sample_vals[gibbs_step][node]
                  
                  node +=1
                  
        return img,  samples   
                          
    def print_gibbs_conditionals(self, p_vals):
        
        """ function to print values of gibbs conditionals """
        
        num_steps = len(p_vals);
        
        for step_ind in range(num_steps):
                
            print("Gibbs step %d ------------------------"%step_ind)
                
            p_t = p_vals[step_ind]
                
            for x in range(self.num_vars):
                
                print("Node update %d ------ gibbs conditionals:"%x)
                print(p_t[x])
                
    def load_model_params(self, full_path):
        
        """ function to load saved model parameters """
        
        print("Loading model parameters from %s"%full_path)
        with open (full_path, 'rb') as f:
             
             self.theta = cPickle.load(f)
             
        if self.num_hidden == True or (self.num_hidden > 0):
        
           self.W, self.b, self.bhid = self.theta
           
        else:
            
           self.W, self.b = self.theta
        
    def save_model_params(self, full_path):
        
        """ function to save model parameters (self.theta)
        
        file_to_save - file object for saving
        
         """
        
        file_to_save = file(full_path, 'wb')
        
        print("Saving model parameters to %s"%full_path)
        
        cPickle.dump(self.theta, 
                     file_to_save, 
                     protocol=cPickle.HIGHEST_PROTOCOL)
        
        file_to_save.close()
