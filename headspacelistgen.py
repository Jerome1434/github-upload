# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:54:52 2020

@author: s144339
"""
import os
import random
import csv
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

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

for i in list(stringdict.keys()):
    if stringdict[i]['age'] == '':
        stringdict[i]['age'] = '0'
    
    if stringdict[i]['beardDescriptor'] != 'none':
        stringdict.pop(i)
    elif stringdict[i]['moustacheDescriptor'] != 'none':
        stringdict.pop(i)
    elif stringdict[i]['expression'] != 'neutral':
        stringdict.pop(i)
    elif stringdict[i]['spectaclesFlag'] != '0':
        stringdict.pop(i)
    elif stringdict[i]['underChinHoleFlag'] != '0':
        stringdict.pop(i)
    elif stringdict[i]['qualityDescriptor'] != 'none':
        stringdict.pop(i)
    elif int(stringdict[i]['age']) < 18:
        stringdict.pop(i)
    elif stringdict[i]['declaredethnicGroup'] != 'white british':
        stringdict.pop(i)

gender = Counter()
agecat = Counter()
age_gendercat = Counter()
idlist = list()
stringdict_grouped = {'young female':{},
                      'young male':{},
                      'lower-middle-aged female':{},
                      'lower-middle-aged male':{},
                      'upper-middle-aged female':{},
                      'upper-middle-aged male':{},
                      'elderly female':{},
                      'elderly male':{},
                      'young transgender':{}}

for i in stringdict.keys():
    idlist.append([stringdict[i]['id'], stringdict[i]['gender'], 
                   stringdict[i]['age']])
    if int(stringdict[i]['age']) < 40:
        agecat['young (18 <= age < 40)'] += 1
        age_gendercat['young '+stringdict[i]['gender']] += 1
        stringdict_grouped['young '+stringdict[i]['gender']][i] = stringdict[i]
    elif (int(stringdict[i]['age']) >= 40 and
          int(stringdict[i]['age']) < 55):
        agecat['lower-middle-aged (40 <= age < 55)'] += 1
        age_gendercat['lower-middle-aged '+stringdict[i]['gender']] += 1
        stringdict_grouped['lower-middle-aged '+stringdict[i]['gender']][i] = stringdict[i]
    elif (int(stringdict[i]['age']) >= 55 and
          int(stringdict[i]['age']) < 70):
        agecat['upper-middle-aged (55 <= age < 70)'] += 1
        age_gendercat['upper-middle-aged '+stringdict[i]['gender']] += 1
        stringdict_grouped['upper-middle-aged '+stringdict[i]['gender']][i] = stringdict[i]
    elif int(stringdict[i]['age']) >= 70:
        agecat['elderly (age >= 70)'] += 1
        age_gendercat['elderly '+stringdict[i]['gender']] += 1
        stringdict_grouped['elderly '+stringdict[i]['gender']][i] = stringdict[i]
        
    gender[stringdict[i]['gender']] += 1

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
# fig,axes = plt.subplots(ncols=len(stringdict_grouped), constrained_layout=True)
counter = 1
for i in list(stringdict_grouped):
    age = []
    for j in list(stringdict_grouped[i]):
        age.append(int(stringdict_grouped[i][j]['age']))
    agerange = max(age) - min(age)
    if agerange == 0:
        agerange = 1
    ax = fig.add_subplot(2, 5, counter)
    N, bins, patches = ax.hist(age, bins=agerange, density=True)
    ax.set_title(i+', N='+str(len(age)))
    ax.set_xlabel('age')
    ax.set_ylabel('frequency')
    counter += 1
fig.set_size_inches([18, 9])
plt.savefig(os.getcwd()+'\\output\\agedistribution.png', dpi=600)