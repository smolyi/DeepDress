# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:46:39 2020

@author: ilansmoly
"""
import pandas as pd
import os

paths = ['asin2url.toptee.txt','asin2url.dress.txt','asin2url.shirt.txt']
for p in paths:
    mat = pd.read_csv(p,header =None,sep='\t')
    for url in mat[1]:
        os.system('wget '+url + ' -P img/')

