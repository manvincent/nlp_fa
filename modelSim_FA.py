#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:12:00 2023

@author: vman
"""

# Import general modules
import numpy as np
import pandas as pd
import os
import time
# cwd = os.path.dirname(os.path.realpath(__file__))
# os.chdir(cwd)
os.chdir('/media/Data/Projects/NLP/Analysis_2022-2023/modelling')
# Import specific modules 
import warnings
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from optimizer import *
from utilities import *
from Models import *

### Model fitting script
def initialize():
    # Set groupID, subjectIDs, sessionIDs
    group = 'fa1'
    simsPerSub = 100
    # Set paths
    homeDir = '/media/Data/Projects/NLP/Analysis_2022-2023'
    datDir = f'{homeDir}/data'
    modelDir = f'{homeDir}/modelling'
    outDir = f'{homeDir}/results'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    simDir = f'{datDir}/simData'
    if not os.path.exists(simDir):
        os.makedirs(simDir)
    
    # Specify sublist
    subList = np.unique([i.split('_')[0] for i in os.listdir(f'{datDir}/subData')]).astype(int) 
    # Initialize the Model class
    initDict = dict2class(dict(group=group,
                               simsPerSub=simsPerSub,
                               subList=subList,
                               homeDir=homeDir,
                               datDir=datDir,
                               modelDir=modelDir,
                               outDir=outDir,
                               simDir=simDir))
    return(initDict)

def preprocess(data): 
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    for column in data: 
        data['lemma'] = data[column].apply(lemmatizer.lemmatize)
    # Remove single-letter entries 
    single_entry_idx = np.where(data.lemma.str.len() < 2)[0]
    data.lemma.loc[single_entry_idx] = np.nan
    return data, single_entry_idx


def simModel():
    # Initialize paths and sublist
    initDict = initialize()

    # Load in general data 
    # Import wordbags as list
    wordbag_all = pd.read_csv(f'{initDict.datDir}/fa_wordbag.csv', header = None, names=['word'])
    wordbag_all, _  = preprocess(wordbag_all)
    
    # Import cosine similarity matrix
    full_cosmat = pd.read_csv(f'{initDict.datDir}/cos_sim_matrix.csv', header = None).to_numpy()
    
    # Check if there are estimated parameter resuts: 
    resFile = f'{initDict.outDir}/model_results.csv'
    if os.path.exists(resFile):
        modCompareDF = pd.read_csv(resFile)
    else:        
        raise Exception('Error! Results file not found. Did you fit the model?') 
        
    for subIdx, subID in enumerate(initDict.subList):
        t0 = time.time()        
        print(f'Simulating by parameters of subject: {subID}, {subIdx}/{len(initDict.subList)}')
        # Get subject directory
        sub_datDir = f'{initDict.datDir}/subData'
                    
        # Import subject word list
        sub_word = pd.read_csv(f'{sub_datDir}/{subID}_word.csv', names=['word'])
        # Preprocess subject word list
        sub_word, nan_idx = preprocess(sub_word)
        # NA single-letter entires 
        sub_word.iloc[nan_idx] = np.nan
        
        # Get total # of trials (for iterations)
        numTrials = len(sub_word)
                                
        # Package relevant info into taskData dict
        taskData = dict2class(dict(numTrials=numTrials,
                                   sub_word = sub_word,
                                   wordbag_all = wordbag_all,
                                   cosmat = full_cosmat))

        # Get relevant model results for this subject 
        subModDF = modCompareDF[(modCompareDF.subID == subID)]         
        finished_models = subModDF.model.unique() 
        
        # Simulate with the Markov Walker Model
        model_label = 'MW'
        outFile = f'{initDict.simDir}/{subID}_{model_label}_sim_offpolicy.csv'
        if not os.path.exists(outFile):
            if model_label in finished_models:                 
                print(f'Model {model_label}')            
                # Get model results
                modRes = subModDF[subModDF.model == model_label]
                # Get estimated parameters
                fitParam = modRes.beta.item() 
                mod = ModelMarkovWalker(taskData) 
                # Simulate and iterate over # sims per sub
                mod_simDF = pd.DataFrame()  
                for s in np.arange(initDict.simsPerSub): 
                    mod_simDF[f'sim_{s}'] = mod.simulate_offpolicy(fitParam)
                # Export                     
                mod_simDF.to_csv(outFile,
                                 na_rep=np.nan, 
                                 header = False, index=False)
            
        
        # Simulate with the Memory Walker Model
        model_label = 'MemW'
        outFile = f'{initDict.simDir}/{subID}_{model_label}_sim_offpolicy.csv'
        if not os.path.exists(outFile):
            if model_label in finished_models:                 
                print(f'Model {model_label}')            
                # Get model results
                modRes = subModDF[subModDF.model == model_label]
                # Get estimated parameters
                fitParam = modRes.beta.item(), modRes.gamma.item()  
                mod = ModelMemoryWalker(taskData) 
                # Simulate and iterate over # sims per sub
                mod_simDF = pd.DataFrame()  
                for s in np.arange(initDict.simsPerSub): 
                    mod_simDF[f'sim_{s}'] = mod.simulate_offpolicy(fitParam)
                # Export                     
                mod_simDF.to_csv(outFile,
                                 na_rep=np.nan, 
                                 header = False, index=False)
                
        # Print time
        print(f'Elapsed time: {time.time() - t0} sec')
                        
            
    return

if __name__ == "__main__":
    simModel()
