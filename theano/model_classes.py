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
from   utils import make_raster_plots
from   collections import OrderedDict
import timeit
import os

class BoltzmannMachine(object):
    
    """ class to implement fully visible Boltzmann Machine """
    
    def __init__(self, 
                 num_vars, 
                 training_inputs = None,
                 algorithm = None,
                 batch_size = None,
                 num_samples = None,
                 num_cd_steps = None,
                 data_samples = None,
                 unique_samples = None,
                 is_uniform   = None,
                 mf_steps = None,
                 use_momentum = None,
                 W= None, 
                 b= None, 
                 report_p_tilda =False,
                 test_mode= False,
                 training = True):
        
        """ Constructor for Fully Visible Boltzmann Machine
        
        num_vars - a number of visible nodes/variables
        
        training_inputs - N x D matrix of training inputs
        
        """
        
        self.num_vars       = num_vars
        
        self.batch_size     = batch_size
        
        self.num_samples    = num_samples
        
        self.num_cd_steps   = num_cd_steps
        
        self.data_samples   = data_samples
        
        self.unique_samples = unique_samples
        
        self.mf_steps       = mf_steps
        
        self.is_uniform     = is_uniform
        
        self.use_momentum   = use_momentum
        
        self.report_p_tilda = report_p_tilda
        
        self.side = int(np.sqrt(self.num_vars))
        
        self.np_rand_gen = np.random.RandomState(1234)
        
        #self.theano_rand_gen =\
         #theano.sandbox.rng_mrg.MRG_RandomStreams(self.np_rand_gen.randint(2**30))
         
        self.theano_rand_gen =\
         T.shared_randomstreams.RandomStreams(self.np_rand_gen.randint(2**30))
                                         
        self.algorithm = algorithm
        
        theano.config.exception_verbosity = 'high'
        
        self.node_indices  = \
        theano.shared(np.arange(self.num_vars), name="node_indices")
        
        self.x               = T.matrix('x')
           
        self.x_tilda         = T.matrix('x_tilda')
        
        if training:
           
           self.updates = OrderedDict()
           
           self.N_train = training_inputs.shape[0]
           
           self.train_inputs = theano.shared(np.asarray(training_inputs,
                                          dtype=theano.config.floatX),
                                          borrow= True)
                                          
           self.learning_rate  = T.dscalar('learning_rate')
           
           if use_momentum == True:
               
              print("Will add momentum term to gradient computations")
              
              self.momentum  = T.dscalar('learning_rate')
              
              self.grad_vec = {}
              
              self.grad_vec['W'] = theano.shared(np.zeros([self.num_vars, self.num_vars],
              dtype = theano.config.floatX), name = 'W_momentum', borrow = True)
              
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
                                          
              if W is None:
                  
                 # different W initializations: 
        
                 # uniform_init =\
                 # self.np_rand_gen.uniform(-np.sqrt(3.0/(num_vars)),\
                 # np.sqrt(3.0 / (num_vars)), size = (num_vars, num_vars))
                 
                 # uniform_init =\
                 # self.np_rand_gen.uniform(-0.00000001,\
                 # 0.00000001, size = (num_vars, num_vars)) 
                 
                 W0_init = 0.00000001*\
                 self.np_rand_gen.normal(size = (num_vars, num_vars)) 
        
                 W0 = np.asarray(W0_init, dtype = theano.config.floatX)
              
                 W0 = (W0 + np.transpose(W0))/2.0
              
                 W0 = W0 - np.diag(np.diag(W0))
        
                 self.W = theano.shared(value= W0, name='W', borrow=True)
              
                 test_W = self.W.get_value() 
              
                 assert sum(np.diag(test_W)) == 0.0
              
                 assert (test_W == np.transpose(test_W)).all() == True
              
              else:
            
                 self.W = W
           
              if b is None:
        
                 bias_init = np.zeros(num_vars, dtype = theano.config.floatX)
        
                 self.b = theano.shared(value= bias_init, name='b', borrow=True)
           
              else:
            
                 self.b = b
           
           self.theta           = [self.W, self.b]
           
           self.train_set       = set(range(self.N_train))
        
           self.minibatch_set   = T.ivector('minibatch_set')
        
           self.sample_set      = T.ivector('sample_set')
           
           self.var_num_samples = T.iscalar('var_num_samples')
           
           if self.algorithm == "CD1":
           
              self.x_gibbs= theano.shared(np.zeros([self.batch_size,self.num_vars],
                                          dtype=theano.config.floatX),
                                          borrow = True, name= "x_gibbs")
              
           if (self.algorithm == "CSS") and (self.is_uniform  != True):
              
              init_mf = self.np_rand_gen.uniform(0,1, 
              size = (self.num_vars, self.num_samples))
              
              init_mf = np.asarray(init_mf, dtype = theano.config.floatX)
              
              self.mf_params = theano.shared(init_mf, 
                                             name= "mf_params", 
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
        
    def add_mf_approximation(self):
        
        """ function to add mf approximation of energy terms in Z"""
        
        if self.unique_samples:
               
           print("Samples will be drawn for each instance in a minibatch")
           [list_mf_samples, weight_term], updates =\
           theano.scan(self.get_mf_samples, n_steps = self.batch_size)
         
           self.updates.update(updates)
         
           list_mf_samples = theano.gradient.disconnected_grad(list_mf_samples)
        
           weight_term    = theano.gradient.disconnected_grad(weight_term)
        
           weight_term    = T.log(self.var_num_samples*weight_term)
        
           approx_Z_mf, updates = \
           theano.scan(lambda i : -self.compute_energy(list_mf_samples[i],\
           self.var_num_samples),sequences = [T.arange(self.batch_size)])
        
           self.updates.update(updates)
        
           weight_term   = T.reshape(weight_term,
                                    [self.batch_size, self.var_num_samples])
        
           approx_Z_mf = T.reshape(approx_Z_mf,
                                   [self.batch_size, self.var_num_samples])
                                     
           approx_Z_mf = approx_Z_mf - weight_term
           
        else:
               
           print("Samples will be shared between instances in a minibatch")
              
           mf_samples, weight_term = self.get_mf_samples()
              
           mf_samples  = theano.gradient.disconnected_grad(mf_samples)
        
           weight_term = theano.gradient.disconnected_grad(weight_term)
        
           weight_term = T.log(self.var_num_samples*weight_term)
              
           approx_Z_mf = -self.compute_energy(mf_samples, self.var_num_samples)
              
           approx_Z_mf = approx_Z_mf - weight_term
           
        return approx_Z_mf
        
    def add_is_uni_approximation(self):
        
        """ function to add computations of energies and their weights
        in the approximating term in normalizer Z. Samples are obtained
        using uniform importances sampling."""
        
        weight_term = T.log(self.var_num_samples) + self.num_vars*T.log(0.5)
        
        if self.unique_samples:
           
           approx_Z = -self.compute_energy(self.x_tilda, 
                                           self.var_num_samples*self.batch_size)
           
        else:
           
           approx_Z = -self.compute_energy(self.x_tilda, self.var_num_samples)
              
        approx_Z = approx_Z - weight_term
        
        if self.unique_samples:
           
           approx_Z = T.reshape(approx_Z, [self.batch_size, self.var_num_samples])
        
        return approx_Z
        
    def add_complementary_term(self):
        
        """ function to add computations on approximating term of log Z
        This term does not involve training points explicitly."""
        
        approx_Z_mf = None
        
        if self.num_samples > 0:
            
           non_data_samples = True
           
           if self.is_uniform:
               
              print("Will use uniform importance sampling for Z approximation")
              
              approx_Z = self.add_is_uni_approximation()
              
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
           
           if self.is_uniform:
              
              max_vals_tiled = T.tile(max_vals,
                                      (1,self.var_num_samples+self.batch_size))
               
           else:
           
              max_vals_tiled= T.tile(max_vals,(1,self.var_num_samples+1))
           
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
           
           approx_Z_data = -self.compute_energy(self.x_tilda, self.N_train)
           
        if (self.data_samples < self.N_train) and (self.data_samples != 0):
            
           print("Will uniformly sample from training set for Z approximation")
           
           approx_Z_data = -self.compute_energy(self.x_tilda, 
                                          self.batch_size*self.data_samples)
                                        
           approx_Z_data = T.reshape(approx_Z_data, [self.batch_size, self.data_samples])
           
           approx_Z_data = approx_Z_data - T.log(self.data_samples)
           
           minibatch_evals = -T.reshape(minibatch_evals, [self.batch_size,1])
           
           approx_Z_data = T.concatenate([approx_Z_data, minibatch_evals],axis = 1)
           
        if non_data_samples:
           
           if self.unique_samples:
               
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
        
    def get_mf_evaluations(self, samples):
        
        """ function to get evaluations of mean field distribution"""
        
        evals = (self.mf_params**samples)*((1.0 - self.mf_params)**(1- samples))
        
        evals = T.prod(evals, axis=0) # axis= 0 : node index, axis=1 : nth datum
        
        #evals = 1.0/(0.000000001 + evals)
        
        evals = 1.0 / evals
        
        return evals
        
    def get_mf_samples(self):
        
        """ function to sample from mean-field distribution """
        
        samples = self.theano_rand_gen.binomial(size= (self.num_vars,
                                                       self.var_num_samples),
                                                n   = 1, 
                                                p   = self.mf_params,
                                                dtype=theano.config.floatX)
        
        importance_weights =  self.get_mf_evaluations(samples)
        
        return T.transpose(samples), importance_weights
        
    def add_mf_updates(self):
        
        """ function to add mean-field updates"""
        
        self.mf_updates, _ = theano.scan(lambda i: self.sigmoid_output(self.mf_params,i),
                                         sequences = [T.arange(self.num_vars)])
                                   
    def do_mf_updates(self, num_steps, report = False):
        
        """ function to implement mean-field updates for approximation
        of data distribution"""
        
        output_vars =[]
        
        if report:
            
           mean_mf_params = T.mean(T.log(self.mf_params))
           
           output_vars.append(mean_mf_params)
           
        update_funct = theano.function(inputs  =[],
                                       outputs = output_vars,
                                       updates = [(self.mf_params,\
                                       self.mf_updates)])
                                       
        for step in range(num_steps):
            
            if report:
               
               avg_log_mf = update_funct() 
               
               print("Step %d: average value of MF parameter --- %f"%
               (step, avg_log_mf[0]))
               
            else:
            
               update_funct()
            
    def compute_energy(self, x, num_terms):
        
        """ function to evaluate energies over a given set of inputs """
        
        evals, _ = \
        theano.scan(lambda i: self.energy_function(T.transpose(x[i,:])), \
        sequences = [T.arange(num_terms)] )
        
        return evals
        
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
        
        minibatch_energy_evals = self.compute_energy(self.x, self.batch_size)
        
        if self.algorithm =="CSS":
            
           normalizer_term = self.add_css_approximation(minibatch_energy_evals)
           
        if self.algorithm =="CD1":
           
           normalizer_term = self.compute_energy(self.x_gibbs, self.batch_size)
           
           normalizer_term = -T.mean(normalizer_term)
           
        #  cost is negative log likelihood   
        self.cost = T.mean(minibatch_energy_evals) + normalizer_term
        
    def add_cd_samples(self):
        
        """ function to add sampling procedure for CD approximation """ 
        
        (self.p_xi_given_x_, self.gibbs_samples), self.gibbs_updates =\
        theano.scan(self.gibbs_step_fully_visible, n_steps = self.num_cd_steps)
        
    def get_cd_samples(self): 
        
        """ function to obtain samples for CD approxmation """
        
        get_samples = theano.function(inputs  = [self.minibatch_set],
                                      outputs = [self.p_xi_given_x_[-1], 
                                                 self.gibbs_samples[-1]], 
                                      givens  = {self.x_gibbs: 
                                      self.train_inputs[self.minibatch_set,:]},
                                      #start the chain at the data distribution
                                      updates = self.gibbs_updates)
                                      
        return get_samples
                                    
    def add_grad_updates(self):
        
        """ Compute and collect gradient updates to dictionary """
        
        gradients = T.grad(self.cost, self.theta)
        
        for target_param, grad in zip(self.theta, gradients):
            
            if target_param.name =="W":
                
               grad = grad - T.diag(T.diag(grad)) # no x i - xi connections
               # for all i = 1, ..., D
               
            if self.use_momentum:
                
               g_tilda = self.momentum*self.grad_vec[target_param.name] - \
               self.learning_rate*grad
                
               self.updates[target_param] = target_param + g_tilda
               
               # store g_tilda for next iteration:
               
               self.updates[self.grad_vec[target_param.name]] = g_tilda
               
            else:
               
               self.updates[target_param] = target_param -\
               self.learning_rate*grad
            
            ## or T.cast(lrate, dtype = theano.config.floatX) to 
            ## guarantee compatibility with GPU
            
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
        
        fe_x_bin  = self.compute_energy(x_bin, self.batch_size)
        
        fe_x_bin_flip_i = self.compute_energy(x_bin_flip_i, self.batch_size)
        
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
              
           if (self.data_samples == 0) and self.is_uniform:
               
              input_dict = {
               self.x      : self.train_inputs[self.minibatch_set,:]
              }
           
              var_list = [self.x_tilda, self.minibatch_set]
           
        if self.algorithm =="CD1":
            
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
            
           output_vars.append(T.shared(0))
        
        var_list.append(self.var_num_samples)
           
        opt_step = theano.function(inputs  = var_list,
                                   outputs = output_vars,
                                   updates = self.updates,
                                   givens  = input_dict,
                                   on_unused_input='warn')
        
        return opt_step
         
    def sigmoid_output(self, x, var_index):
        
        """ function to compute the sigmoid output for the Gibbs step
        for fully visible Boltzmann Machine.
        """
        
        sigmoid_activation = self.b[self.node_indices[var_index]] +\
         2*(T.dot(self.W[self.node_indices[var_index],:],x) - 
         self.W[self.node_indices[var_index],var_index]*x[var_index,:])
         
        return T.nnet.sigmoid(sigmoid_activation)
        
    def gibbs_update_node(self, target_node):
        
        """ Gibbs sampling update for a target node for fully
        visible Boltzmann Machine
        """
        
        p_xi_given_x_ = self.sigmoid_output(T.transpose(self.x_gibbs),target_node)
        
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
        
        cd_sampling = None
        
        if self.algorithm == "CD1":

           self.add_cd_samples()

           cd_sampling = self.get_cd_samples()
           
        if self.algorithm == "CSS" and self.mf_steps > 0: 
        
           self.add_mf_updates()
           
        self.add_objective()

        self.add_grad_updates()  
        
        if self.report_p_tilda:
            
           self.add_p_tilda()
    
        self.add_pseudo_cost_measure()

        optimize = self.optimization_step()
 
        return cd_sampling, optimize
        
    def test_p_tilda(self, test_inputs, random_inputs):
        
        """ function to test p_tilda values with trained Boltzmann Machine"""
        
        self.batch_size      = test_inputs.shape[0]
        
        self.num_samples     = random_inputs.shape[0]
        
        self.var_num_samples = self.num_samples
        
        self.add_p_tilda()
        
        var_list = [self.x, self.x_tilda]
        
        get_p_tilda = theano.function(inputs = var_list,
                                      outputs= self.p_tilda)
                                     
        probs = get_p_tilda(test_inputs, random_inputs)
        
        si = self.batch_size+self.np_rand_gen.choice(self.num_samples, 10, False)
        
        return probs[0:self.batch_size], probs[si]
        
    def relative_likelihood(self):
    
        """ function to compute relative, unnormalized likelihood 
        of given examples"""
        
        return T.exp(-self.compute_energy(self.x, self.batch_size))
        
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
        
    def add_p_tilda(self):
        
        """ function to compute p_tilda for explicit
         gradient computations"""
        
        minibatch_energies = -self.compute_energy(self.x, self.batch_size)
        
        sample_energies = self.add_is_uni_approximation()
        
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
        
        self.add_p_tilda()
        
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
        sequences =[T.arange(self.var_num_samples)])
        
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
        
    def sample_from_mf_approx(self, 
                              num_chains, 
                              num_samples,
                              num_steps,
                              save_to_path,
                              test_inputs    = None,
                              save_mf_params = True):
        
        """ function to sample from mean-field approximation of trained
        fully visible Boltzmann Machine."""
        
        images = np.zeros([num_chains*num_samples, self.num_vars])
        
        self.num_samples = num_chains
        
        self.var_num_samples = self.num_samples
        
        if type(test_inputs) is  np.ndarray:
           print("Will initialize MF parameters with input images") 
           init_mf = np.transpose(test_inputs)
            
        else:
           
           print("Will initialize MF parameters with uniform distribution")
           init_mf = self.np_rand_gen.uniform(0,1, 
           size = (self.num_vars, self.num_samples))
              
        init_mf = np.asarray(init_mf, dtype = theano.config.floatX)
              
        self.mf_params = theano.shared(init_mf, 
                                       name= "mf_params", 
                                       borrow= True)
                                       
        self.add_mf_updates()
        
        self.do_mf_updates(num_steps = num_steps, report = True)
        
        if save_mf_params:
            
           split_path   = os.path.split(save_to_path)
           
           mf_file = os.path.join(split_path[0], "MF_PARAMS.model")
           
           print("Saving MF parameters to %s"%mf_file)
           
           mf_file = file(mf_file, 'wb')
           
           cPickle.dump(self.mf_params, 
                        mf_file, 
                        protocol=cPickle.HIGHEST_PROTOCOL)
        
           mf_file.close()
        
        mf_samples, sample_probs = self.get_mf_samples()
        
        get_samples = theano.function(inputs  = [],
                                      outputs = [sample_probs, 
                                                 mf_samples,
                                                 self.mf_params])
                                      
        print("Sampling")      
                                
        for ind in range(num_samples):
            
            p_out, samples_out, mf_vals = get_samples()
            
            images[num_chains*ind:num_chains*(ind+1),:] = samples_out
            #np.round(np.transpose(mf_vals))
        
        make_raster_plots(images, 
                          num_samples, 
                          num_chains, 
                          reshape_to = [self.side, self.side], 
                          save_to_path = save_to_path,
                          test_images = False)    
    
    def sample_from_bm(self,
                       num_chains, 
                       num_samples,
                       num_steps,
                       save_to_path,
                       test_inputs = None,
                       print_gibbs = False):
        
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
        
        self.x_gibbs = theano.shared(init_chains)
        
        theano.config.exception_verbosity = 'high'
        
        print("Running gibbs chains ...\n")
        (p_xi_given_x_, x_samples), updates =\
        theano.scan(self.gibbs_step_fully_visible, n_steps = num_steps)
        
        output_vars = [p_xi_given_x_, x_samples]
        
        get_samples = theano.function(inputs  = [],
                                      outputs = output_vars, 
                                      updates = updates)
                                      
        for ind in range(num_samples):
            
            p_all, samples_all = get_samples()
            
            if print_gibbs:
            
               self.print_gibbs_conditionals(p_vals = p_all)
               
            p_out, samples_out = self.assemble_image(p_all, 
                                                     samples_all,
                                                     num_chains,
                                                     step = 100)
            
            #p_out = p_all
               
            #samples_out = samples_all[-1]
            
            # without resetting the chains are persistent
            #self.x_gibbs.set_value(init_chains)
            
            print("Sample %d -- max pixel activations for %d gibbs chains:"%
            (ind, num_chains))
            print(np.max(np.transpose(p_out), axis= 1))
            print("")
            
            is_samples = self.np_rand_gen.binomial(n=1, 
                                                   p=0.5, 
                                                   size = (10000, self.num_vars))
   
            gibbs_p_tilda, rand_p_tilda = \
            self.test_p_tilda(np.transpose(samples_out), is_samples)
            
            print("p_tilda values for gibbs samples:")
            print(gibbs_p_tilda)
            print("")
            print("p_tilda values for randomly chosen importance samples:")
            print(rand_p_tilda)
            print("")
            
            images[num_chains*(ind+1):num_chains*(ind+2),:] = \
            np.round(np.transpose(p_out))
        
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
        
        with open (full_path, 'rb') as f:
        
             self.theta = cPickle.load(f)
        
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
        
        
        
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
