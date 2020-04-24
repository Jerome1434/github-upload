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

gender = Counter()
agecat = Counter()
ethnicity = Counter()
idlist = list()

for i in stringdict.keys():
    idlist.append([stringdict[i]['id'], stringdict[i]['gender'], 
                   stringdict[i]['age'], stringdict[i]['declaredethnicGroup']])
    if int(stringdict[i]['age']) < 35:
        agecat['young (18 <= age <= 34)'] += 1
    elif (int(stringdict[i]['age']) >= 35 and
          int(stringdict[i]['age']) < 44):
        agecat['lower-middle-aged (35 <= age <= 43)'] += 1
    elif (int(stringdict[i]['age']) >= 44 and
          int(stringdict[i]['age']) < 65):
        agecat['upper-middle-aged (44 <= age <= 64)'] += 1
    elif int(stringdict[i]['age']) >= 65:
        agecat['elderly (age >= 65)'] += 1
    
    # if int(stringdict[i]['age']) <= 39:
    #     agecat['young (18<age<40)'] += 1
    # elif (int(stringdict[i]['age']) >= 40 and
    #       int(stringdict[i]['age']) <= 59):
    #     agecat['middle-aged (40<age<60)'] += 1
    # elif int(stringdict[i]['age']) >= 60:
    #     agecat['elderly (age>60)'] += 1
        
    ethnicity[stringdict[i]['declaredethnicGroup']] += 1
    gender[stringdict[i]['gender']] += 1

# for i in range(len(row)):
#     temp = row[i]
#     row[i] = int(temp.replace('"',''))
# row = np.array(row, dtype='int')
# stringdictarray = np.zeros(len(stringdict))

# counter = 0
# for i in list(stringdict):
#     stringdictarray[counter] = int(stringdict[i]['id'])
#     counter += 1
# stringdictarray = stringdictarray.astype(int)

# selection = np.random.choice(stringdictarray, size = 16)
# selection.sort()