#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:05:23 2021

@author: rcj
"""
import pandas as pd

def readClassification(classes):
    dfClasses = pd.read_excel(classes)
    for i in range(1, len(dfClasses['File Name'])):
        if dfClasses['Frame Number'][i] > dfClasses['Frame Number'][i-1]:
            dfClasses.loc[i, 'File Name'] = dfClasses['File Name'][i-1]
    
    return dfClasses['File Name'].values.tolist(), dfClasses['Frame Number'].values.tolist(), dfClasses['Classification'].values.tolist()