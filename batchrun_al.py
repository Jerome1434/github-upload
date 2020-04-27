# -*- coding: utf-8 -*-

from ampscan import AmpObject, align, registration, vis
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
inputlist = os.listdir(os.getcwd()+'\input')
masklist = os.listdir(os.getcwd()+'\masks')

temp = []
for i in range(len(masklist)):
    if '.stl' in masklist[i]:
        temp.append(masklist[i])
masklist = temp

for i in range(len(masklist)):
    target = AmpObject(os.getcwd()+'\masks\\'+masklist[i])
    for j in range(len(inputlist)):
        base = AmpObject(os.getcwd()+'\input\\'+inputlist[j])
        alignedobjects = c_al.customalign(base, target, method='max-dist')
                
        alignedobjects.m.save(os.getcwd()+'\output\mask_'+masklist[i][:-4]+'_input_'+inputlist[j])