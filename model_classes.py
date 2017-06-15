""" 
Author: Rokas Stanislovas
MSc Project: Likelihood Approximations
for Energy-Based Models
MSc Computational Statistics and 
Machine Learning
"""

import theano
import theano.tensor as T
import numpy as np
import cPickle
import Image
import sys
from   utils import make_raster_plots
from   collections import OrderedDict

class BoltzmannMachine(object):
    
    """ class to implement fully visible Boltzmann Machine """
    
    def __init__(self, 
                 num_vars, 
                 training_inputs, 
                 test_inputs, 
                 algorithm,
                 batch_size,
                 num_samples,
                 W= None, 
                 b= None, 
                 training = True):
        
        """ Constructor for Boltzmann Machine
        
        num_vars - a number of visible nodes/variables
        
        training_inputs - N x D matrix of training inputs
        
        """
        
        self.num_vars    = num_vars
        
        self.batch_size  = batch_size
        
        self.num_samples = num_samples
        
        self.side = int(np.sqrt(self.num_vars))
        
        np_rand_gen = np.random.RandomState(1234)
        
        self.theano_rand_gen =\
         T.shared_randomstreams.RandomStreams(np_rand_gen.randint(2**30))
                                         
        self.algorithm = algorithm
    
        self.num_test_examples = test_inputs.shape[0]
        
        theano.config.exception_verbosity = 'high'
        
        self.node_indices  = \
        theano.shared(np.arange(self.num_vars), name="node_indices")
        
        if training:
            
           print("Training Phase")
           
           self.updates = OrderedDict()
           
           num_examples = training_inputs.shape[0]
           
           self.train_inputs = theano.shared(np.asarray(training_inputs,
                                          dtype=theano.config.floatX),
                                          borrow= True)
                                          
           if W is None:
        
              uniform_init = np_rand_gen.uniform(-np.sqrt(3.0/num_vars),\
              np.sqrt(3.0 / num_vars), size = (num_vars, num_vars) )
        
              W0 = np.asarray(uniform_init, dtype = np.float64)
        
              self.W = theano.shared(value= W0, name='W', borrow=True)
           
           else:
            
              self.W = W
           
           if b is None:
        
              bias_init = np.zeros(num_vars, dtype = np.float64)
        
              self.b = theano.shared(value= bias_init, name='b', borrow=True)
           
           else:
            
              self.b = b
           
           self.theta           = [self.W, self.b]
        
           self.x               = T.matrix('x')
           
           self.x_tilda         = T.matrix('x_tilda')
           
           self.train_set       = set(range(num_examples))
        
           self.minibatch_set   = T.ivector('minibatch_set')
        
           self.sample_set      = T.ivector('sample_set')
           
           self.x_gibbs         = theano.shared(np.zeros([self.batch_size,self.num_vars])) 
           
           self.cd_samples      = theano.shared(np.zeros([self.batch_size,self.num_vars]))
    
        else:
            
           self.test_inputs = theano.shared(np.asarray(test_inputs,
                                         dtype=theano.config.floatX),
                                         borrow= True)
            
           print("Test Phase")
           
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
    
    def add_css_approximation(self, num_samples):
        
        """ function to compute an approximating part of parition function
        according to Botev et al. 2017. For now uses uniform 
        importance sampling over the complementary set of training examples
        """
        
        approx_Z = self.compute_energy(self.x_tilda, self.batch_size*num_samples)
        
        approx_Z = T.reshape(approx_Z, [self.batch_size,num_samples])
        
        approx_Z = (1.0/num_samples)*T.sum(T.exp(-approx_Z), axis=1)
        
        return approx_Z
        
    def compute_energy(self, x, num_terms):
        
        """ function to evaluate energies over a given set of inputs """
        
        evals, _ = \
        theano.scan(lambda i: self.energy_function(T.transpose(x[i,:])), \
        sequences = [T.arange(num_terms)] )
        
        return evals
         
    def add_objective(self, num_samples = None, num_steps = 1):
        
        """ function to add the model objective """ 
        
        minibatch_energy_evals = self.compute_energy(self.x, self.batch_size)
        
        if self.algorithm == "CSS":
            
           approx_Z = self.add_css_approximation(num_samples)
           
           normalizer_term = \
           T.mean(T.log(T.exp(-minibatch_energy_evals) + approx_Z) )
           
        if self.algorithm =="CD1":
           
           normalizer_term = self.compute_energy(self.cd_samples, self.batch_size)
           
           normalizer_term = -T.mean(normalizer_term)
           
        self.cost = T.mean(minibatch_energy_evals) + normalizer_term
        
    def add_cd_samples(self, num_steps = 1):
        
        """ function to add sampling procedure for CD approximation """ 
        
        (self.p_xi_given_x_, self.gibbs_samples), self.gibbs_updates =\
        theano.scan(self.gibbs_step_fully_visible, n_steps = num_steps)
        
    def get_cd_samples(self): 
        
        """ function to obtain samples for CD approxmation """
        
        get_samples = theano.function(inputs  = [self.minibatch_set],
                                      outputs = [self.p_xi_given_x_[-1], 
                                                 self.gibbs_samples[-1]], 
                                      givens  = {self.x_gibbs: 
                                      self.train_inputs[self.minibatch_set,:]},
                                      updates = self.gibbs_updates)
                                      
        return get_samples
                                    
    def add_grad_updates(self, lrate):
        
        """  compute and collect gradient updates to dictionary
        
        lrate - learning rate
        """
        
        if self.algorithm =="CSS":
           
           gradients = T.grad(self.cost, self.theta)
           
        if self.algorithm =="CD1":
           
           gradients = T.grad(self.cost, 
                              self.theta)
           
        for target_param, grad in zip(self.theta, gradients):
            
            self.updates[target_param] = target_param - lrate*grad
            
            ## or T.cast(lrate, dtype = theano.config.floatX) to 
            ## guarantee compatibility with GPU
            
    def select_samples(self, minibatch_set, num_samples):
        
        """ function to select samples for css approximation
        with uniform sampling
         """
        
        minibatch_size = len(minibatch_set)
        
        ##  compl_inds :: complementary set of training points
        ##  for now, complementary set is computed jointly;
        ##  uses naive approach, uniform sampling

        compl_inds = list(self.train_set - set(minibatch_set))
            
        samples = np.random.choice(compl_inds, minibatch_size*num_samples, replace=False)
        
        samples = list(samples)
        
        assert len(samples) == num_samples*minibatch_size
        
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
            self.x         : self.train_inputs[self.minibatch_set,:],
            } 
            
           var_list = [self.minibatch_set]
        
        opt_step = theano.function(inputs= var_list,
                                   outputs=self.pseudo_cost,
                                   updates=self.updates,
                                   givens = input_dict)
        
        return opt_step
         
    def sigmoid_output(self, x, var_index):
        
        """ function to compute the sigmoid output for the Gibbs step
        for fully visible Boltzmann Machine 
        
        """
        
        sigmoid_activation = self.b[self.node_indices[var_index]] +\
         2*T.dot(self.W[self.node_indices[var_index],:],x) 
         
        return T.nnet.sigmoid(sigmoid_activation)
        
    def gibbs_update_node(self, target_node):
        
        """ Gibbs sampling update for a target node for fully
        visible Boltzmann Machine
        """
        
        p_xi_given_x_ = self.sigmoid_output(T.transpose(self.x_gibbs),target_node)
        
        samples = self.theano_rand_gen.binomial(size = p_xi_given_x_.shape,
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
        
    def add_graph(self, learning_rate):
        
        """ function to build a Theano computational graph """
        
        cd_sampling = None
        
        self.add_objective(self.batch_size, self.num_samples)

        if self.algorithm == "CD1":

           self.add_cd_samples()

           cd_sampling = self.get_cd_samples()

        self.add_grad_updates(learning_rate)   

        self.add_pseudo_cost_measure()

        optimize = self.optimization_step()
 
        return cd_sampling, optimize
        
    def sample_from_bm(self, 
                       num_chains, 
                       num_samples, 
                       num_steps, 
                       save_to_path,
                       test_mode = True):
        
        """ function to generate images from trained 
        Boltzmann Machine (fully visible).
        """
        
        images = np.zeros([num_chains*num_samples + num_chains, self.num_vars])
        
        select_examples = np.random.choice(self.num_test_examples, 
                                           num_chains, 
                                           replace=False)
        
        init_chains =  np.asarray(
            self.test_inputs.get_value(borrow=True)[select_examples,:],
            dtype=theano.config.floatX)
            
        images[0:num_chains,:] = init_chains
        
        chain_vars = theano.shared(init_chains)
        
        theano.config.exception_verbosity = 'high'
        
        self.x_gibbs = chain_vars
        
        if test_mode:
        
           print(self.x_gibbs.get_value()[:,0])
           print(self.x_gibbs.get_value()[:,1])
           print(self.x_gibbs.get_value()[:,self.num_vars-1])
        
           (p_xi_given_x_, x_samples), updates =\
           theano.scan(self.gibbs_step_fully_visible, n_steps = 1) 
        
           get_samples = theano.function(inputs  = [],
                                         outputs = [p_xi_given_x_,x_samples], 
                                         updates = updates)
                                      
           p, samples = get_samples()
        
           print("")
           print(samples[0][0])
           print(samples[0][1])
           print(samples[0][self.num_vars -1])
        
           print("")
           print(self.x_gibbs.get_value()[:,0])
           print(self.x_gibbs.get_value()[:,1])
           print(self.x_gibbs.get_value()[:,self.num_vars-1])
           ### testing
           sys.exit()
        
        (p_xi_given_x_, x_samples), updates =\
        theano.scan(self.gibbs_step_fully_visible, n_steps = num_steps) 
        
        get_samples = theano.function(inputs  = [],
                                      outputs = [p_xi_given_x_[-1],x_samples[-1]], 
                                      updates = updates)
        
        for ind in range(num_samples):
            
            p_out, samples_out = get_samples()
            
            images[num_chains*(ind+1):num_chains*(ind+2),:] = np.transpose(p_out)
            
        make_raster_plots(images, 
                          num_samples, 
                          num_chains, 
                          reshape_to = [self.side, self.side], 
                          save_to_path = save_to_path)                                  
        
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
        
        
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
