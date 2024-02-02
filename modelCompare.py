#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 08:46:09 2022

@author: vman
"""
import numpy as np
import pandas as pd
# Numpy options
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
homeDir = '/media/Data/Projects/NLP/Analysis_2022-2023'
datDir = f'{homeDir}/data'
modelDir = f'{homeDir}/modelling'
resDir = f'{homeDir}/results'
# Load in subject list 
subList = np.unique([i.split('_')[0] for i in os.listdir(f'{datDir}/subData')]).astype(int) 

# Load in model comparison results
groupModDF = pd.read_csv(f'{resDir}/model_results.csv')

# Group by subject and get diff between models
# MemW model minus MW model
modDiffDF = (groupModDF.sort_values('model')
        .groupby('subID')
        .agg(lambda x: (x.iat[-1]-x.iat[0]))
        .reset_index())
modDiffDF = modDiffDF.loc[:,['subID','nll','AIC','BIC']]
modDiffDF.columns = ['subID','delta_nll','delta_AIC','delta_BIC']
# If positive values, MW winner (lower BIC)
# If negative values, MewW winner
modDiffDF['MW_win'] = modDiffDF.delta_BIC > 0
modDiffDF['winMod'] = modDiffDF.MW_win.map({True: 'MW',
                                            False: 'MemW'})
outFile = f'{resDir}/model_compare_sub.csv'
modDiffDF.to_csv(outFile, index=False)
