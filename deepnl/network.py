# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: embedsignature=True
# cython: profile=True

"""
A neural network for NLP tagging tasks.
"""

# standard
from __future__ import print_function
import logging
import sys                      # DEBUG

import numpy as np

# local
from math import *

# ----------------------------------------------------------------------

class Variables(object):
    
    #cdef public np.ndarray input, hidden, output

    def __init__(self, input_size=0, hidden_size=0, output_size=0):
        self.input = np.empty(input_size) if input_size else None
        self.hidden = np.empty(hidden_size) if hidden_size else None
        self.output = np.empty(output_size) if output_size else None

# ----------------------------------------------------------------------

class Parameters(object):
    """
    Network parameters: weights and biases.
    Parameters are shared among threads in ASGD.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        self.output_weights = np.zeros((output_size, hidden_size))
        self.output_bias = np.zeros(output_size)
        self.hidden_weights = np.zeros((hidden_size, input_size))
        self.hidden_bias = np.zeros(hidden_size)

    def clear(self, val=0.0):
        self.output_weights[:,:] = val
        self.output_bias[:] = val
        self.hidden_weights[:,:] = val
        self.hidden_bias[:] = val

    def initialize(self, input_size, hidden_size, output_size):
        """
        Creates the weight matrices with random values.
        """
        # We must pass sizes since one cannot get shape from Cython ndarray

        # Note : optimal initialization of weights may depend on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid()
        #        compared to tanh().

        # set the seed for replicability
        #np.random.seed(42)          # DEBUG

        high = 2.38 / np.sqrt(input_size) # [Bottou-88]
        self.hidden_weights = np.random.uniform(-high, high, (hidden_size, input_size))
        self.hidden_bias = np.random.uniform(-high, high, (hidden_size))

        high = 2.38 / np.sqrt(hidden_size) # [Bottou-88]
        self.output_weights = np.random.uniform(-high, high, (output_size, hidden_size))
        self.output_bias = np.random.uniform(-high, high, (output_size))

    def copy(self, p):
        """Used in ASGD"""
        self.output_weights[:,:] = p.output_weights
        self.output_bias[:] = p.output_bias
        self.hidden_weights[:,:] = p.hidden_weights
        self.hidden_bias[:] = p.hidden_bias

    def addSquare(self, grads):
        """For adaGrad"""
        self.output_weights += grads.output_weights * grads.output_weights
        self.output_bias += grads.output_bias * grads.output_bias
        self.hidden_weights += grads.hidden_weights * grads.hidden_weights
        self.hidden_bias += grads.hidden_bias * grads.hidden_bias

    def update(self, grads, learning_rate, ada=None):
        """
        Adjust the weights.
        :param ada: cumulative square gradients for performing AdaGrad.
        AdaGrad: G_t, where G(i,i)_t = G(i,i)_t-1 + grad(i)^2
        * i.e. we cumulate the square of gradients in G for parameter p:
        * G += g^2
        * p -= LR * g / sqrt(G + eps)

        Consider using AdaDelta instead:
        http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf

        """
        if ada:
            # print('ada', ada.hidden_weights[:5,:5], ada.hidden_weights[-5:,-5:], file=sys.stderr) # DEBUG
            ada.addSquare(grads)
            self.output_weights += learning_rate * grads.output_weights / np.sqrt(ada.output_weights)
            self.output_bias += learning_rate * grads.output_bias / np.sqrt(ada.output_bias)

            # print('ada', ada.hidden_weights[:5,:5], ada.hidden_weights[-5:,-5:], file=sys.stderr) # DEBUG
            # print('uhw', learning_rate, adaEps, self.hidden_weights[:5,:5], grads.hidden_weights[:5,:5], grads.hidden_weights[-5:,-5:], file=sys.stderr) # DEBUG

            self.hidden_weights += learning_rate * grads.hidden_weights / np.sqrt(ada.hidden_weights)
            self.hidden_bias += learning_rate * grads.hidden_bias / np.sqrt(ada.hidden_bias)
        else:
            # divide by the fan-in
            self.output_weights += grads.output_weights * learning_rate / 100  # DEBUG / self.hidden_size
            self.output_bias += grads.output_bias * learning_rate / 100 # DEBUG self.hidden_size
            # print('uhw', learning_rate, self.hidden_weights[:2,:2], grads.hidden_weights[:2,:2], file=sys.stderr) # DEBUG
            self.hidden_weights += grads.hidden_weights * learning_rate / 100  # DEBUG self.input_size
            self.hidden_bias += grads.hidden_bias * learning_rate / 100 # DEBUG self.input_size

    def save(self, file):
        """
        Saves the parameters to a file.
        It saves the weights and biases.
        """
        np.savez(file, hidden_weights=self.hidden_weights,
                 output_weights=self.output_weights,
                 hidden_bias=self.hidden_bias, output_bias=self.output_bias)

    @classmethod
    def load(cls, file):
        """
        Loads the neural network from a file.
        It loads weights, biases and sizes.
        """
        data = np.load(file)
        
        p = cls.__new__(cls)
        p.hidden_weights = data['hidden_weights']
        p.output_weights = data['output_weights']
        p.hidden_bias = data['hidden_bias']
        p.output_bias = data['output_bias']

        return p

# ----------------------------------------------------------------------

class Gradients(Parameters):
    """
    Gradients for all network Parameters, plus input gradients.
    """
    
    # gradients for input variables
    #cdef public np.ndarray input

    def __init__(self, input_size, hidden_size, output_size):
        super(Gradients, self).__init__(input_size, hidden_size, output_size)
        self.input = np.zeros(input_size)

    def clear(self, val=0.0):
        self.output_weights.fill(val)
        self.output_bias.fill(val)
        self.hidden_weights.fill(val)
        self.hidden_bias.fill(val)
        self.input.fill(val)

# ----------------------------------------------------------------------

class Network(object):
    """
    Basic neural network.
    Parameters are the weights of the various layers.
    """
    # FIXME: make Parameters an array of Layers, each one with its
    # forward/backward/update methods.

    def __init__(self, input_size, hidden_size, output_size,
                 p=None):
        """
        :param input_size: number of input variables
        :param hidden_size: number of hidden variables
        :param output_size: number of output variables
        :param p: the parameters to use.

        For replicability, the seed should have been set, e.g. to
           np.random.seed(42)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if not p:
            p = Parameters(input_size, hidden_size, output_size)
            # initialize parameters to random values
            p.initialize(input_size, hidden_size, output_size)
        self.p = p

        # saver fuction
        self.saver = lambda nn: None
    
    def description(self):
        """
        Returns a textual description of the network.
        """
        desc = """
        Input layer size: %d
        Hidden layer size: %d
        Output size: %d
        """ % (self.input_size, self.hidden_size, self.output_size)
        
        return desc
    
    def variables(self, slen=1):
        """Allocate variables.
        :param slen: sequence length (for sequence or convolutional networks)
        """
        return Variables(self.input_size * slen, self.hidden_size * slen, self.output_size * slen)

    def gradients(self, slen=1):
        """Allocate variables.
        :param slen: sequence length (for sequence or convolutional networks)
        """
        return Gradients(self.input_size, self.hidden_size, self.output_size)

    def parameters(self):
        """Allocate network parameters.
        """
        return Parameters(self.input_size, self.hidden_size, self.output_size)

    def forward(self, vars):
        """
        Runs the network on the given variables: hidden and visible. 
        """
        # (hidden_size, input_size) . input_size = hidden_size
        self.p.hidden_weights.dot(vars.input, vars.hidden)
        vars.hidden += self.p.hidden_bias
        hardtanh(vars.hidden, vars.hidden)
        self.p.output_weights.dot(vars.hidden, vars.output)
        vars.output += self.p.output_bias

    def backpropagate(self, y, vars, grads):
        """
        Cost is the hinge loss.
        Compute the gradients of the cost for each layer.
        :param y: the correct outcome.
        :param vars: the network variables.
        :param grads: were to store the gradients.
        :return: the hinge loss.
        """

        # Multiclass hinge loss (Crammer&Singer):
        # hl(x, y) = max(0, 1 + max_t!=y f(x)[t] - f(x)[y])
        # Hinge loss is 0 if the score of the correct label exceeds the score
        # of every other label by a margin of at least 1.
        # m = argmax_t!=y f(x)[t]
        # dhl / df [y] = -1 if f(x)[m] - f(x)[y] > 1, else 0
        # dhl / df [t] = +1 if f(x)[t] - f(x)[y] > 1, else 0
        fx_y = vars.output[y]
        fx_m = np.NINF # negative infinity
        for i, v in enumerate(vars.output):
            if i == y:
                continue
            if v > fx_m:
                fx_m = v
        hinge_loss = max(0.0, 1 + fx_m - fx_y)

        if hinge_loss == 0.0:
            return hinge_loss
        p = self.p

        # minimizing C(f_4)
        # f_4 = W_2 f_3 + b_2
        # dC / db_2 = dC / df_4
        # negative gradient:
        grads.output_bias[:] = np.where(vars.output - fx_y > -1, -1, 0) # -1
        grads.output_bias[y] = +1

        # dC / dW_2 = dC / df_4 f_3
        # (output_size) x (hidden_size) = (output_size, hidden_size)
        np.outer(grads.output_bias, vars.hidden, grads.output_weights)
        # dC / df_3 = dC / df_4 * W_2
        # (output_size) * (output_size, hidden_size) = (hidden_size)
        grads.output_bias.dot(p.output_weights, grads.hidden_bias) # temporary

        # f_3 = hardtanh(f_2)
        # dC / df_2 = dC / df_3 * hardtanh'(f_2)
        self.hardtanh_back(vars.hidden, grads.hidden_bias, grads.hidden_bias)

        # f_2 = W_1 f_1 + b_1
        # dC / db_1 = dC / df_2

        # dC / dW_1 = dC / df_2 * f_1
        # (hidden_size) x (input_size) = (hidden_size, input_size)
        np.outer(grads.hidden_bias, vars.input, grads.hidden_weights)

        # dC / df_1 = dC / df_2 * W_1
        # (hidden_size) * (hidden_size, input_size) = (input_size)
        grads.hidden_bias.dot(p.hidden_weights, grads.input)

        # Lookup layer
        # f_1 = W_0 f_0
        # dC / dW_0 = dC / df_1 * W_0

        return hinge_loss

    def update(self, grads, learning_rate, ada=None):
        self.p.update(grads, learning_rate, ada)

    def save(self, file):
        """
        Saves the neural network to a file.
        It saves the parameters.
        """
        self.p.save(file)

    @classmethod
    def load(cls, file):
        """
        Loads the neural network from a file.
        It loads weights, biases and sizes.
        """
        nn = cls.__new__(cls)
        nn.p = Parameters.load(file)
        nn.input_size = nn.p.hidden_weights.shape[1]
        nn.hidden_size = nn.p.hidden_weights.shape[0]
        nn.output_size = nn.p.output_weights.shape[0]

        return nn
    
    def hardtanh_back(y, grads, grads_in):
        """backward of hardtanh in terms of y = hardtanh(x)
        Propagates the output gradients to the input, by multiplying with the
        derivative p hardtanh.
        grads:      gradients of output.
        grads_in :  output array in which to place the result.
        """
        it = np.nditer([y, grads, grads_in],
                    op_flags = [['readonly'], ['readonly'], ['writeonly']])
        for w, g, o in it:
            if w == -1.0 or w == 1.0:
                o[...] = 0.0
            else:
                o[...] = g[...]
        return grads_in
    
    def hardtanh(y, out=None):
        """Hard hyperbolic tangent.
        out : array_like, optional
            Alternative output array in which to place the result.
        """
        it = np.nditer([y, out],
                    op_flags = [['readonly'],
                                ['writeonly', 'allocate', 'no_broadcast']])
        for w, o in it:
            if w < -1:
                o[...] = -1
            elif w > 1:
                o[...] = 1
            else:
                o[...] = w
        return it.operands[1]