#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:15:56 2023

@author: vman
"""



import pandas as pd
import numpy as np
from scipy.stats import gamma, beta, uniform
from utilities import *
# Define model types
class ModelHybridWalker(object):

    def __init__(self, taskData):
        self.numParams = 3
        self.param_bounds = dict2class(dict(beta=(1,20), # min, max                                           
                                            gamma=(0,1),
                                            epsilon=(0,1)
                                            ))
        
        self.taskData = taskData
        
    def mixture(self, layer1, layer2, weight):
        cossim_layer = weight * layer1 + (1 - weight) * layer2
        return(cossim_layer)
    
    def memory(self, V, history, tI, gam):
        # Iterate over past t trials
        if tI > 0:
            # Make vector of exponentiated gammas of length k
            gam_vec = np.ones(len(history), dtype = float) * gam
            gam_vec = gam_vec**(tI - np.arange(tI+1))
            # Weigh memory layer
            mem_layer = V[history,:]
            cossim_layer_ = (mem_layer.T * gam_vec)
            # Compute sum over history
            cossim_layer = np.sum(cossim_layer_ , axis=1)
            # Downweight nodes corresponding to recently chosen nodes
            cossim_layer[history] *= np.flip(gam_vec)
        else:
            # Get vector of cosine similarity to all other nodes
            cossim_layer = V[history[0],:]
        return(cossim_layer)
    
    def actor(self, V, beta):
        """
        Args are the inputs to the model, besides the general model params:
        Args:
            V: input to the softmax function
            beta: softmax inverse temperature; scalar
        """
        # Compute transition probabilities
        trans_prob = np.exp(beta*V) / np.sum(np.exp(beta*V), axis=0)
        # Pick an option given the softmax probabilities (return index)
        
        next_idx = np.random.choice(np.arange(len(trans_prob)), 1, p=trans_prob)[0]
        # next_idx = np.argmax(trans_prob)

        return(next_idx, trans_prob)
            
    def initSeeds(self):
        # Set up seed searching
        self.numSeeds = 10**self.numParams
        seeds = np.zeros((self.numParams, self.numSeeds), dtype='float')
        for i in np.arange(self.numParams):
            seeds[i,:] = np.linspace(-5,5, self.numSeeds)
            
        return seeds   
    
    
    def likelihood(self, param):
        # Unpack parameter values
        # smB = param
        smB, gam, eps = self.transformParams(param)
        # Get seed word for this subject (init ref node)
        ref_node = self.taskData.sub_word.word.iloc[0].lower()
        ref_node_lemma = self.taskData.sub_word.lemma.iloc[0].lower()
        # Get index of initial node
        respIdx = np.where(self.taskData.wordbag_all.word == ref_node)[0][0]

        # Init store of indices of chosen words (from subject)
        history_respIdx = []        
        
        # Initialize log likelihood
        LL = 0
        # Iterate over trials
        for tI in np.arange(self.taskData.numTrials-1):
            if not ref_node != ref_node: # check if string or np.nan
            
                # Store indices of chosen words
                history_respIdx.append(respIdx)
                
                # Get transformed cosine similarity layer based on history of previous choices
                mem_layer = self.memory(self.taskData.cosmat,
                                           history_respIdx,
                                           tI,
                                           gam)
                
                # Get vector of cosine similarity to all other nodes
                markov_layer = self.taskData.cosmat[respIdx,:]
                
                                
                # Mix memory and markovian layers
                cossim_layer = self.mixture(mem_layer, markov_layer, eps)
                cossim_layer = cossim_layer.copy()
                
                                    
                # Downweigh nodes corresponding to inflections of current ref word
                self_inflect_idx = np.where(self.taskData.wordbag_all.lemma == ref_node_lemma)[0]
                cossim_layer[self_inflect_idx] = 0
                                                                      
                # Estimate softmax output (probabilities)
                _, trans_prob = self.actor(cossim_layer, smB)
                    
                # Get next chosen node
                ref_node = self.taskData.sub_word.word.iloc[tI + 1]       
                ref_node_lemma = self.taskData.sub_word.lemma.iloc[tI + 1]       
                if not ref_node != ref_node: 
                    # Get index of initial node
                    respIdx = np.where(self.taskData.wordbag_all.word == ref_node)[0][0]
                    # Get log likelihood of actual chosen next node, given model
                    LL += np.log(trans_prob[respIdx])                
                else: 
                    ref_node = self.taskData.sub_word.word.iloc[tI + 1]       
                    ref_node_lemma = self.taskData.sub_word.lemma.iloc[tI + 1]       
                    
        return LL * -1


        
    def transformParams(self, param):
        transParams = param.astype(float)
        # Transform offered values into model parameters
        transParams[0] = max(self.param_bounds.beta) / (1 + np.exp(-1 * transParams[0])) # smB        
        transParams[1] = max(self.param_bounds.gamma) / (1 + np.exp(-1 * transParams[1])) # gam         
        transParams[2] = max(self.param_bounds.epsilon) / (1 + np.exp(-1 * transParams[1])) # eps               
        return transParams  
    
    def simulate(self, param):
        # Retrieve fitted param
        smB, gam, eps = param
        # Get seed word for this subject
        ref_node = self.taskData.sub_word.word.iloc[0].lower()
        ref_node_lemma = self.taskData.sub_word.lemma.iloc[0].lower()                
        # Initialize list of model outputs
        mod_gen = []
        mod_gen_idx = []
        # Iterate over trials
        for tI in np.arange(self.taskData.numTrials):
            # Get index of current node
            respIdx = np.where(self.taskData.wordbag_all.word == ref_node)[0][0]
            # Store current word
            mod_gen.append(ref_node)
            mod_gen_idx.append(respIdx)
    
            # Get transformed cosine similarity layer based on history of previous choices
            mem_layer = self.memory(self.taskData.cosmat,
                                       mod_gen_idx,
                                       tI,
                                       gam)
            # Get vector of cosine similarity to all other nodes
            markov_layer = self.taskData.cosmat[respIdx,:]
            
            # Mix memory and markovian layers
            cossim_layer = self.mixture(mem_layer, markov_layer, eps)
            cossim_layer = cossim_layer.copy()
                        
            
            # Downweigh nodes corresponding to inflections of current ref word
            self_inflect_idx = np.where(self.taskData.wordbag_all.lemma == ref_node_lemma)[0]
            cossim_layer[self_inflect_idx] = 0
    
            # Estimate softmax output (probabilities)
            next_idx , trans_prob = self.actor(cossim_layer, smB)
    
            # redefine reference node as the walker's new position
            ref_node = self.taskData.wordbag_all.word.iloc[next_idx]       
            ref_node_lemma = self.taskData.wordbag_all.lemma.iloc[next_idx]   
                        
        return mod_gen

     