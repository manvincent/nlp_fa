#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:28:14 2020

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
    # Set paths
    homeDir = '/media/Data/Projects/NLP/Analysis_2022-2023'
    datDir = f'{homeDir}/data'
    modelDir = f'{homeDir}/modelling'
    outDir = f'{homeDir}/results'
    # Import dependences
    os.chdir(modelDir)
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    # Specify sublist
    subList = np.unique([i.split('_')[0] for i in os.listdir(f'{datDir}/subData')]).astype(int) 
    # Initialize the Model class
    initDict = dict2class(dict(group=group,
                               subList=subList,
                               homeDir=homeDir,
                               datDir=datDir,
                               modelDir=modelDir,
                               outDir=outDir))
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


def fitModel():
    # Initialize paths and sublist
    initDict = initialize()

    # Load in general data 
    # Import wordbags as list
    wordbag_all = pd.read_csv(f'{initDict.datDir}/fa_wordbag.csv', header = None, names=['word'])
    wordbag_all, _  = preprocess(wordbag_all)
    
    # Import cosine similarity matrix
    full_cosmat = pd.read_csv(f'{initDict.datDir}/cos_sim_matrix.csv', header = None).to_numpy()
    
    # Check if there are interim resuts: 
    outFile = f'{initDict.outDir}/model_results.csv'
    if os.path.exists(outFile):
        modCompareDF = pd.read_csv(outFile)
    else:        
        # Create structures for storing estimates across subjects
        modCompareDF = pd.DataFrame()
        
    for subIdx, subID in enumerate(initDict.subList):
        t0 = time.time()
        print(f'Fitting subject: {subID}, {subIdx}/{len(initDict.subList)}')
        
        
        sub_datDir = f'{initDict.datDir}/subData'
    
    
        # Import subject word list
        sub_word = pd.read_csv(f'{sub_datDir}/{subID}_word.csv', names=['word'])
        # Preprocess subject word list
        sub_word, nan_idx = preprocess(sub_word)
        # NA single-letter entires 
        sub_word.iloc[nan_idx] = np.nan
        
        # Get # of trials  (for AIC/BIC)
        numLLTrials = sub_word.word.notnull().sum()
        # Get total # of trials (for iterations)
        numTrials = len(sub_word)
                
        # Import subject word embeddings 
        sub_embed = pd.read_csv(f'{sub_datDir}/{subID}_embed.csv', header=None).to_numpy()
        
        # Import subject rt 
        sub_rt = pd.read_csv(f'{sub_datDir}/{subID}_rt.csv', names=['rt'])
        sub_rt.rt.loc[nan_idx] = np.nan
                
        # Package relevant info into taskData dict
        taskData = dict2class(dict(numTrials=numTrials,
                                   numLLTrials=numLLTrials,
                                   sub_embed = sub_embed,
                                   sub_word = sub_word,
                                   sub_rt = sub_rt,
                                   wordbag_all = wordbag_all,
                                   cosmat = full_cosmat))

     
        # Initialize the optimizer
        modOptimizer = Optimizer()
        
        # Check if the model has been run already        
        finished_models = modCompareDF.model[modCompareDF.subID == subID].unique()
        
        ## Fit the Markov Walker Model
        model_label = 'MW'
        if model_label not in finished_models:                 
            print(f'Model {model_label}')            
            ModelMW_res, ModelMW_estParams = modOptimizer.fitModel(ModelMarkovWalker, taskData)
            # Update results container with estimated model parameters 
            ModelMW_res.ModelMW(ModelMW_estParams)
            # Append results for model comparison
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                modCompareDF = modCompareDF.append(dict(
                                                        subIdx = subIdx,
                                                        subID=subID,
                                                        **dict({'model': model_label},
                                                        **ModelMW_res.__dict__)),
                                                    ignore_index=True)
                # Output to dataframe
                modCompareDF.to_csv(outFile, index=False)
        
        ## Fit the Memory Walker Model
        model_label = 'MemW'
        if model_label not in finished_models:                 
            print(f'Model {model_label}')            
            ModelMemW_res, ModelMemW_estParams = modOptimizer.fitModel(ModelMemoryWalker, taskData)
            # Update results container with estimated model parameters 
            ModelMemW_res.ModelMemW(ModelMemW_estParams)
            # Append results for model comparison
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                modCompareDF = modCompareDF.append(dict(
                                                        subIdx = subIdx,
                                                        subID=subID,
                                                        **dict({'model': model_label},
                                                        **ModelMemW_res.__dict__)),
                                                    ignore_index=True)
                # Output to dataframe
                modCompareDF.to_csv(outFile, index=False)
        
        # # Fit the Memory Walker Model
        # model_label = 'HW'
        # if model_label not in finished_models:                 
        #     print(f'Model {model_label}')            
        #     ModelHW_res, ModelHW_estParams = modOptimizer.fitModel(ModelHybridWalker, taskData)
        #     # Update results container with estimated model parameters 
        #     ModelHW_res.ModelHW(ModelHW_estParams)
        #     # Append results for model comparison
        #     with warnings.catch_warnings():
        #         warnings.simplefilter(action='ignore', category=FutureWarning)
        #         modCompareDF = modCompareDF.append(dict(
        #                                                 subIdx = subIdx,
        #                                                 subID=subID,
        #                                                 **dict({'model': model_label},
        #                                                 **ModelHW_res.__dict__)),
        #                                             ignore_index=True)
        #         # Output to dataframe
        #         modCompareDF.to_csv(outFile, index=False)
                      

        
        
        # Print time
        print(f'Elapsed time: {time.time() - t0} sec')
        
        
    return

if __name__ == "__main__":
    fitModel()
