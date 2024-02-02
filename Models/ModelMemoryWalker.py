#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:00:19 2023

@author: vman
"""

import pandas as pd
import numpy as np
from scipy.stats import gamma, beta, uniform
from utilities import *
# Define model types
class ModelMemoryWalker(object):

    def __init__(self, taskData):
        self.numParams = 2
        self.param_bounds = dict2class(dict(beta=(1,20), # min, max                                           
                                            gamma=(0, 1)
                                            ))
        
        self.taskData = taskData
        
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
        smB, gam = self.transformParams(param)
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
                cossim_layer = self.memory(self.taskData.cosmat,
                                           history_respIdx,
                                           tI,
                                           gam)
                # Normalize cossim-layer (max-subtract) to prevent softmax overflow
                cossim_layer = cossim_layer.copy()            
                cossim_layer  = cossim_layer - np.max(cossim_layer)
                
                    
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
        return transParams  
    
    def simulate_onpolicy(self, param):
        '''
            Only simulates using lemmas
            Previously chosen words are downweighted to prevent recurrent lops 
            Single-letter words are re-chosen
        '''
        # Retrieve fitted param
        smB, gam = param
        # Get seed word for this subject
        ref_node = self.taskData.sub_word.lemma.iloc[0].lower()
        # Initialize list of model outputs
        mod_gen = []
        mod_gen_idx = []
        # Iterate over trials
        for tI in np.arange(self.taskData.numTrials):
            # Get index of current node
            respIdx = np.where(self.taskData.wordbag_all.lemma == ref_node)[0][0]
            # Store current word
            mod_gen.append(ref_node)
            mod_gen_idx.append(respIdx)
    
            # Get transformed cosine similarity layer based on history of previous choices
            cossim_layer = self.memory(self.taskData.cosmat,
                                       mod_gen_idx,
                                       tI,
                                       gam)
            cossim_layer = cossim_layer.copy()            
            
            # Downweigh nodes corresponding to inflections of current ref word
            self_inflect_idx = np.where(self.taskData.wordbag_all.lemma == ref_node)[0]
            cossim_layer[self_inflect_idx] = 0
            # Downweight nodes of all chosen words
            cossim_layer[mod_gen_idx] = 0 
    
            # Estimate softmax output (probabilities)
            next_idx , trans_prob = self.actor(cossim_layer, smB)

    
            # redefine reference node as the walker's new position
            ref_node = self.taskData.wordbag_all.lemma.iloc[next_idx]       
            
            # Repeat if single-letter word generated
            while True:
                # Evaluate that the chosen word is not nan and more than 1 letter
                if (ref_node == ref_node) and len(ref_node) > 1:
                    break
                else: 
                    next_idx , trans_prob = self.actor(cossim_layer, smB)
                    ref_node = self.taskData.wordbag_all.lemma.iloc[next_idx]   

        return mod_gen
    
    
    def simulate_offpolicy(self, param):
        '''
            Only simulates using lemmas
            Previously chosen words are downweighted to prevent recurrent lops 
            Single-letter words are re-chosen
        '''
        # Retrieve fitted param
        smB, gam = param
        # Get seed word for this subject (init ref node)
        ref_node = self.taskData.sub_word.word.iloc[0].lower()
        ref_node_lemma = self.taskData.sub_word.lemma.iloc[0].lower()
        # Get index of initial node
        respIdx = np.where(self.taskData.wordbag_all.word == ref_node)[0][0]
        # Initialize list of model outputs
        mod_gen = [ref_node]
        mod_gen_idx = [respIdx] 
            
        
        # Init store of indices of chosen words (from subject)
        history_respIdx = []        
        
        # Iterate over trials
        for tI in np.arange(self.taskData.numTrials-1):
            if not ref_node != ref_node: # check if string or np.nan
            
                # Store indices of chosen words
                history_respIdx.append(respIdx)
                
                # Get transformed cosine similarity layer based on history of previous choices
                cossim_layer = self.memory(self.taskData.cosmat,
                                           history_respIdx,
                                           tI,
                                           gam)
                # Normalize cossim-layer (max-subtract) to prevent softmax overflow
                cossim_layer = cossim_layer.copy()            
                cossim_layer  = cossim_layer - np.max(cossim_layer)
                
                    
                # Downweigh nodes corresponding to inflections of current ref word
                self_inflect_idx = np.where(self.taskData.wordbag_all.lemma == ref_node_lemma)[0]
                cossim_layer[self_inflect_idx] = 0
                                                                      
                # Estimate softmax output (probabilities)
                next_idx, trans_prob = self.actor(cossim_layer, smB)
                # Store model-predicted word corresponding to next idx
                sim_node = self.taskData.wordbag_all.lemma.iloc[next_idx] 
                mod_gen.append(sim_node)
                mod_gen_idx.append(next_idx)
                                     
                    
                # Get next chosen node
                ref_node = self.taskData.sub_word.word.iloc[tI + 1]       
                ref_node_lemma = self.taskData.sub_word.lemma.iloc[tI + 1]       
                if not ref_node != ref_node: 
                    # Get index of initial node
                    respIdx = np.where(self.taskData.wordbag_all.word == ref_node)[0][0]     
                else: 
                    ref_node = self.taskData.sub_word.word.iloc[tI + 1]       
                    ref_node_lemma = self.taskData.sub_word.lemma.iloc[tI + 1]       

        return mod_gen
    