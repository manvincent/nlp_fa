#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:07:54 2022

@author: vman
"""

# Set up a container for model parameter estimates (fits)
class fitParamContain():
    def __init__(self,  NLL, numParams, AIC, BIC, invHess):
        self.nll = NLL 
        self.numParams = numParams
        self.AIC = AIC
        self.BIC = BIC
        self.invHess = invHess        
        return

    def ModelMW(self, params):
        self.beta = params[0]
        return
    
    def ModelMemW(self, params):
        self.beta = params[0]
        self.gamma = params[1]
        return
    
    def ModelHW(self, params):
        self.beta = params[0]
        self.gamma = params[1]
        self.epsilon = params[2]        
        return
    