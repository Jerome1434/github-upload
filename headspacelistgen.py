# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:54:52 2020

@author: s144339
"""
import os
import numpy as np
import ast
import random

path = 'C:\\Users\\s144339\\headspaceOnline\\subjects\\'
files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file and 'person' in file:
            files.append(os.path.join(r, file))

stringdict = dict()

for i in files:
    j = open(i, 'r')
    stringdict[i[-9:-4]] = j.read()
    j.close()

for i in stringdict:
    tempstring = stringdict[i]
    stringdict[i] = {}
    tempstring = tempstring.splitlines()
    for j in range(len(tempstring)):
        tempstring[j] = tempstring[j].split(':')
        stringdict[i][tempstring[j][0]] = tempstring[j][1]

for i in list(stringdict):
    if stringdict[i]['age'] == '':
        stringdict[i]['age'] = '0'
    
    if (stringdict[i]['beardDescriptor'] != 'none' or
    stringdict[i]['moustacheDescriptor'] != 'none' or
    stringdict[i]['expression'] != 'neutral' or
    stringdict[i]['spectaclesFlag'] != '0' or
    stringdict[i]['underChinHoleFlag'] != '0' or
    stringdict[i]['qualityDescriptor'] != 'none' or
    int(stringdict[i]['age']) <= 24):
        stringdict.pop(i)
        
selection = random.sample(list(stringdict), 100)