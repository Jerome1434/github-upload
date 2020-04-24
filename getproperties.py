# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:25:28 2020

@author: s144339
"""
import numpy as np
import matplotlib.pyplot as plt
import os

path = 'C:\\Users\\s144339\\headspaceOnline\\subject STLs - meshlab\\'
generated = os.listdir(path)
for i in range(len(generated)):
    generated[i] = generated[i][7:-4]
generated = list(set(generated))
generated.sort()

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

generatedlist = []

for i in range(len(generated)):
    if generated[i] in stringdict.keys():
        temp = stringdict[generated[i]]
        temp = list(temp.values())
        generatedlist.append(temp)

totalage = []
for i in list(stringdict):
    if stringdict[i]['age'] == '':
        stringdict.pop(i)
    else:
        totalage.append(stringdict[i]['age'])

totalage = [float(i) for i in totalage]
meantotalage = np.mean(totalage)
mediantotalage = np.median(totalage)
stdtotalage = np.std(totalage)
mintotalage = np.min(totalage)
maxtotalage = np.max(totalage)


age = [float(row[3]) for row in generatedlist]
meanage = np.mean(age)
medianage = np.median(age)
stdage = np.std(age)
minage = np.min(age)
maxage = np.max(age)

gender = [row[1] for row in generatedlist]
male = gender.count('male')
female = gender.count('female')
ethnicity = [row[2] for row in generatedlist]
whitebritish = ethnicity.count('white british')
