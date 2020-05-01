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
    elif stringdict[i]['declaredethnicGroup'] != 'white british':
        stringdict.pop(i)

gender = Counter()
agecat = Counter()
age_gendercat = Counter()
idlist = list()

for i in stringdict.keys():
    idlist.append([stringdict[i]['id'], stringdict[i]['gender'], 
                   stringdict[i]['age']])
    if int(stringdict[i]['age']) < 40:
        agecat['young (18 <= age < 40)'] += 1
        age_gendercat['young '+stringdict[i]['gender']] += 1
    elif (int(stringdict[i]['age']) >= 40 and
          int(stringdict[i]['age']) < 55):
        agecat['lower-middle-aged (40 <= age < 55)'] += 1
        age_gendercat['lower-middle-aged '+stringdict[i]['gender']] += 1
    elif (int(stringdict[i]['age']) >= 55 and
          int(stringdict[i]['age']) < 70):
        agecat['upper-middle-aged (55 <= age < 70)'] += 1
        age_gendercat['upper-middle-aged '+stringdict[i]['gender']] += 1
    elif int(stringdict[i]['age']) >= 70:
        agecat['elderly (age >= 70)'] += 1
        age_gendercat['elderly '+stringdict[i]['gender']] += 1
    
    # if int(stringdict[i]['age']) <= 39:
    #     agecat['young (18<age<40)'] += 1
    # elif (int(stringdict[i]['age']) >= 40 and
    #       int(stringdict[i]['age']) <= 59):
    #     agecat['middle-aged (40<age<60)'] += 1
    # elif int(stringdict[i]['age']) >= 60:
    #     agecat['elderly (age>60)'] += 1
        
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