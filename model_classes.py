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

class BoltzmannMachine(object):
    
    """ class to implement fully visible Boltzmann Machine """
    
    def __init__(self, 
                 num_vars, 
                 training_inputs,
                 algorithm,
                 batch_size,
                 learning_rate,
                 num_samples,
                 num_steps,
                 include_all,
                 W= None, 
                 b= None, 
                 training = True):
        
        """ Constructor for Boltzmann Machine
        
        num_vars - a number of visible nodes/variables
        
        training_inputs - N x D matrix of training inputs
        
        """
        
        self.num_vars    = num_vars
        
        self.batch_size  = batch_size
        
        self.learning_rate = learning_rate
        
        self.num_samples = num_samples
        
        self.num_steps   = num_steps
        
        self.include_all = include_all
        
        self.side = int(np.sqrt(self.num_vars))
        
        np_rand_gen = np.random.RandomState(1234)
        
        self.theano_rand_gen =\
         theano.sandbox.rng_mrg.MRG_RandomStreams(np_rand_gen.randint(2**30))
                                         
        self.algorithm = algorithm
        
        theano.config.exception_verbosity = 'high'
        
        self.node_indices  = \
        theano.shared(np.arange(self.num_vars), name="node_indices")
        
        if training:
            
           print("Training Phase")
           
           self.updates = OrderedDict()
           
           self.N_train = training_inputs.shape[0]
           
           assert self.N_train == 55000
           
           self.train_inputs = theano.shared(np.asarray(training_inputs,
                                          dtype=theano.config.floatX),
                                          borrow= True)
                                          
           if W is None:
        
              uniform_init = np_rand_gen.uniform(-np.sqrt(3.0/num_vars),\
              np.sqrt(3.0 / num_vars), size = (num_vars, num_vars) )
        
              W0 = np.asarray(uniform_init, dtype = theano.config.floatX)
              
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
        
           self.x               = T.matrix('x')
           
           self.x_tilda         = T.matrix('x_tilda')
           
           self.train_set       = set(range(self.N_train))
        
           self.minibatch_set   = T.ivector('minibatch_set')
        
           self.sample_set      = T.ivector('sample_set')
           
           if self.algorithm == "CD1":
           
              self.x_gibbs= theano.shared(np.zeros([self.batch_size,self.num_vars],
                                          dtype=theano.config.floatX),
                                          borrow = True, name= "x_gibbs")
              
           if self.algorithm == "CSS_MF":
              
              self.mf_params = theano.shared(0.5*np.zeros([self.num_vars,self.num_samples],\
              dtype=theano.config.floatX), name= "mf_params", borrow= True)
           
    def energy_function(self, x):
    
        """ to compute energy function for fully visible Boltzmann machine
    
        W - D x D matrix of weight variables in theano
    
        b - D-dimensional column vector of bias terms in theano
    
        x - D-dimensional input vector
    
        return:
    
        - xTWx - bT x 
    
        as a symbolic variable to add to the theano
        computational graph.
    
        """
  
        return -T.dot(T.transpose(x), T.dot(self.W, x)) - T.dot(T.transpose(self.b), x)
    
    def add_css_approximation(self):
        
        """ Function to compute an approximating part of parition
        function according to Botev et al. 2017. For now uses uniform 
        importance sampling over the complementary set of training examples
        """
        
        if self.num_samples < self.N_train:
           
           approx_Z = self.compute_energy(self.x_tilda, 
                                          self.batch_size*self.num_samples)
                                        
           approx_Z = T.reshape(approx_Z, [self.batch_size, self.num_samples])
        
           approx_Z = (1.0/self.num_samples)*T.sum(T.exp(-approx_Z), axis=1)
           
           use_all_data = False
           
        if self.num_samples == self.N_train:
           
           approx_Z = self.compute_energy(self.x_tilda, self.num_samples)
           
           approx_Z = (1.0/self.num_samples)*T.sum(T.exp(-approx_Z))
           
           use_all_data = True
           
        return approx_Z, use_all_data
        
    def add_css_mf_approximation(self):
        
        """ function to define css approximation of normalizer Z
        using mean-field approximation for the importance distribution.
        This distribution is used tto sample the approximating part of Z. 
        This approximating part is complementary to the whole training 
        dataset."""
        
        if self.include_all:
            
           approx_Z_data = self.compute_energy(self.x_tilda, self.N_train)
           
           approx_Z_data = T.sum(T.exp(-approx_Z_data))
           
        else:
           
           approx_Z_data = 0.000000001
        
        mf_samples, inv_q_s = self.get_mf_samples()
        
        mf_samples  = theano.gradient.disconnected_grad(mf_samples)
        
        inv_q_s     = theano.gradient.disconnected_grad(inv_q_s)
        
        approx_Z_mf = self.compute_energy(mf_samples, self.num_samples)
        
        inv_S       = 1.0/self.num_samples
        
        approx_Z_mf = inv_S*T.sum(inv_q_s*T.exp(-approx_Z_mf))
           
        approx_Z    = approx_Z_data + approx_Z_mf
        
        return approx_Z
        
    def get_mf_evaluations(self, samples):
        
        """ function to get evaluations of mean field distribution"""
        
        evals = (self.mf_params**samples)*((1.0 - self.mf_params)**(1- samples))
        
        evals = T.prod(evals, axis=0) # axis= 0 : node index, axis=1 : nth datum
        
        evals = 1.0/(0.000000001 + evals)
        
        return evals
        
    def get_mf_samples(self):
        
        """ function to sample from mean-field distribution """
        
        samples = self.theano_rand_gen.binomial(size= (self.num_vars,self.num_samples),
                                                n   = 1, 
                                                p   = self.mf_params,
                                                dtype=theano.config.floatX)
        
        importance_weights =  self.get_mf_evaluations(samples)
        
        return T.transpose(samples), importance_weights
        
    def add_mf_updates(self):
        
        """ function to add mean-field updates"""
        
        self.mf_updates, _ = theano.scan(lambda i: self.sigmoid_output(self.mf_params,i),
                                   sequences = [T.arange(self.num_vars)])
                                   
    def do_mf_updates(self):
        
        """ function to implement mean-field updates for approximation
        of data distribution"""
        
        updates = self.add_mf_updates()
        
        update_funct = theano.function(inputs=[],
                                       outputs=[],
                                       updates = [(self.mf_params, self.mf_updates)])
        
        for step in range(self.num_steps):
            
            update_funct()
            
    def compute_energy(self, x, num_terms):
        
        """ function to evaluate energies over a given set of inputs """
        
        evals, _ = \
        theano.scan(lambda i: self.energy_function(T.transpose(x[i,:])), \
        sequences = [T.arange(num_terms)] )
        
        return evals
         
    def add_objective(self):
        
        """ function to add model objective for model optimization """ 
        
        minibatch_energy_evals = self.compute_energy(self.x, self.batch_size)
        
        if self.algorithm =="CSS_MF":
            
           approx_Z  = self.add_css_mf_approximation()
           
           if self.include_all:
            
              normalizer_term  = T.log(approx_Z)
              
           else:
              
              normalizer_term  =  \
              T.mean(T.log(T.exp(-minibatch_energy_evals) + approx_Z) ) 
           
        if self.algorithm =="CSS":
            
           approx_Z, full_dataset = self.add_css_approximation()
           
           if full_dataset:
               
              normalizer_term = T.log(approx_Z)
              
           else:
           
              normalizer_term = \
              T.mean(T.log(T.exp(-minibatch_energy_evals) + approx_Z) )
           
        if self.algorithm =="CD1":
           
           normalizer_term = self.compute_energy(self.x_gibbs, self.batch_size)
           
           normalizer_term = -T.mean(normalizer_term)
           
        #  cost is negative log likelihood   
        self.cost = T.mean(minibatch_energy_evals) + normalizer_term
        
    def add_cd_samples(self):
        
        """ function to add sampling procedure for CD approximation """ 
        
        (self.p_xi_given_x_, self.gibbs_samples), self.gibbs_updates =\
        theano.scan(self.gibbs_step_fully_visible, n_steps = self.num_steps)
        
    def get_cd_samples(self): 
        
        """ function to obtain samples for CD approxmation """
        
        get_samples = theano.function(inputs  = [self.minibatch_set],
                                      outputs = [self.p_xi_given_x_[-1], 
                                                 self.gibbs_samples[-1]], 
                                      givens  = {self.x_gibbs: 
                                      self.train_inputs[self.minibatch_set,:]},
                                      # start the chain at the data distribution
                                      updates = self.gibbs_updates)
                                      
        return get_samples
                                    
    def add_grad_updates(self, lrate):
        
        """  compute and collect gradient updates to dictionary
        
        lrate - learning rate
        """
        
        gradients = T.grad(self.cost, self.theta)
        
        for target_param, grad in zip(self.theta, gradients):
            
            if target_param.name =="W":
                
               grad = grad - T.diag(T.diag(grad)) # no x i - xi connections
               # for all i = 1, ..., D
               
            self.updates[target_param] = target_param - lrate*grad
            
            ## or T.cast(lrate, dtype = theano.config.floatX) to 
            ## guarantee compatibility with GPU
            
    def select_samples(self, minibatch_set):
        
        """ function to select samples for css approximation
        with uniform sampling
         """
        
        minibatch_size = len(minibatch_set)
        
        assert minibatch_size == self.batch_size
        
        ##  compl_inds :: complementary set of training points
        ##  for now, complementary set is computed jointly;
        ##  uses naive approach, uniform sampling
        
        if self.num_samples < self.N_train:

           compl_inds = list(self.train_set - set(minibatch_set))
            
           samples = np.random.choice(compl_inds,
                                      minibatch_size*self.num_samples, 
                                      replace=False)
        
           samples = list(samples)
        
           assert len(samples) == self.num_samples*minibatch_size
           
        if self.num_samples == self.N_train:
            
           samples = list(self.train_set)
        
        return samples  
        
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
            
           input_dict = {
            self.x      : self.train_inputs[self.minibatch_set,:],
            self.x_tilda: self.train_inputs[self.sample_set,:]
           }
           
           var_list = [self.sample_set, self.minibatch_set]
           
        if self.algorithm =="CD1":
            
           input_dict = {
            self.x  : self.train_inputs[self.minibatch_set,:],
            } 
            
           var_list = [self.minibatch_set]
           
        if self.algorithm == "CSS_MF":
             
           if self.include_all:
            
              input_dict = {
                 self.x  : self.train_inputs[self.minibatch_set,:],
                 self.x_tilda: self.train_inputs[self.sample_set,:]
              } 
            
              var_list = [self.sample_set, self.minibatch_set]
        
           else:
              
              input_dict = {
                 self.x  : self.train_inputs[self.minibatch_set,:]
              }
              
              var_list = [self.minibatch_set] 
        
        opt_step = theano.function(inputs = var_list,
                                   outputs=self.pseudo_cost,
                                   updates=self.updates,
                                   givens = input_dict)
        
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
         
    def gibbs_step_fully_visible(self):
        
        """   
        Function to inplement a Gibbs step for fully
        visible Boltzmann Machine. 
        """
        
        (get_p, get_samples), updates  =\
         theano.scan(self.gibbs_update_node, 
                     sequences =[T.arange(self.num_vars)])
        
        return (get_p, get_samples), updates
        
    def add_graph(self):
        
        """ function to build a Theano computational graph """
        
        cd_sampling = None
        
        if self.algorithm == "CD1":

           self.add_cd_samples()

           cd_sampling = self.get_cd_samples()
           
        if self.algorithm == "CSS_MF": 
        
           self.add_mf_updates()
           
        self.add_objective()

        self.add_grad_updates(self.learning_rate)   

        self.add_pseudo_cost_measure()

        optimize = self.optimization_step()
 
        return cd_sampling, optimize
        
    def sample_from_bm(self, 
                       test_inputs,
                       num_chains, 
                       num_samples,
                       save_to_path):
        
        """ function to generate images from trained 
        Boltzmann Machine (fully visible).
        """
        
        self.test_inputs = theano.shared(np.asarray(test_inputs,
                                         dtype=theano.config.floatX),
                                         borrow= True)
        
        self.num_test_examples = test_inputs.shape[0]
        
        images = np.zeros([num_chains*num_samples + num_chains, self.num_vars])
        
        select_examples = np.random.choice(self.num_test_examples, 
                                           num_chains, 
                                           replace=False)
        
        init_chains =  np.asarray(
            self.test_inputs.get_value(borrow=True)[select_examples,:],
            dtype=theano.config.floatX)
            
        images[0:num_chains,:] = init_chains
        
        self.x_gibbs = theano.shared(init_chains)
        
        theano.config.exception_verbosity = 'high'
        
        (p_xi_given_x_, x_samples), updates =\
        theano.scan(self.gibbs_step_fully_visible, n_steps = 1)
        
        get_samples = theano.function(inputs  = [],
                                      outputs = [p_xi_given_x_[0],x_samples[0]], 
                                      updates = updates)
                                      
        for ind in range(num_samples):
            
            p_out, samples_out = get_samples()
            
            self.x_gibbs.set_value(init_chains)     
            
            images[num_chains*(ind+1):num_chains*(ind+2),:] = np.round(np.transpose(p_out))
        
        make_raster_plots(images, 
                          num_samples, 
                          num_chains, 
                          reshape_to = [self.side, self.side], 
                          save_to_path = save_to_path)    
                          
    def test_get_mf_samples(self):
        
        samples, weights = self.get_mf_samples()
        
        test_time = theano.function(inputs= [],
                                    outputs=[samples, weights])
                                    
        t0 = timeit.default_timer()
        output1, output2 = test_time()
        t1 = timeit.default_timer()
        print("Time of MF sampling for %d samples is ---- %f"%
        (self.num_samples, (t1-t0)/60.0))
        
    def test_compute_energy_time(self, num_to_test):
        
        """ function to test time of computation """
        
        input_dict = {
           self.x  : self.train_inputs[self.minibatch_set,:]
        }
              
        var_list = [self.minibatch_set] 
        
        comp = self.compute_energy(self.x, num_to_test)
        
        comp = T.sum(T.exp(-comp)*comp)

        test_time = theano.function(inputs= var_list,
                                    outputs=[comp],
                                    givens = input_dict)
                                    
        t0 = timeit.default_timer()
        output = test_time(range(num_to_test))
        t1 = timeit.default_timer()
        print("Time of energy computation for %d inputs is ---- %f"%
        (num_to_test, (t1-t0)/60.0))
        
        
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
        
        
        
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
