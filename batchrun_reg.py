# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:50:27 2020

@author: s144339
"""

from ampscan import AmpObject, vis
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import math
from scipy import spatial
from scipy.optimize import minimize
import custom_align as c_al
import custom_registration as c_reg

import warnings
warnings.filterwarnings('ignore')

#generate a list of all inputfiles as well as a list
#of mask files
facelist = os.listdir(os.getcwd()+'\input')
masklist = os.listdir(os.getcwd()+'\output')

temp = []
for i in range(len(masklist)):
    if '.stl' in masklist[i]:
        temp.append(masklist[i])
masklist = temp

for i in range(len(masklist)):
    mask = AmpObject(os.getcwd()+'\output\\'+masklist[i])
    for j in range(len(facelist)):
        if masklist[i][-9:-4] == facelist[j][-9:-4]:
            face = AmpObject(os.getcwd()+'\input\\'+facelist[j])
            reg = c_reg.customregistration(mask,face,method='point2point')

            c_reg.generateRegCsv(os.getcwd()+'\\reg\mask_'+
                                   masklist[i][10:15]+'_subject_'+
                                   facelist[j][-9:-4]+'.csv', reg)