# -*- coding: utf-8 -*-
"""
@author: Hilal Karakurt
@author:Nisanur Alıcı
@author:Nisa Dilara Korkmaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import scipy.stats
import mne
import pywt
import eeglib

#EEG verilerini alındığı noktalara göre ayırma
chs = {'FP1': [-0.03, 0.08],
       'FP2': [0.03, 0.08],
       'F7': [-0.073, 0.047],
       'F3': [-0.04, 0.041],
       'Fz': [0, 0.038],
       'F4': [0.04, 0.041],
       'F8': [0.073, 0.047],
       'T3': [-0.085, 0],
       'C3': [-0.045, 0],
       'Cz': [0, 0],
       'C4': [0.045, 0],
       'T4': [0.085, 0],
       'T5': [-0.073, -0.047],
       'P3': [-0.04, -0.041],
       'Pz': [0, -0.038],
       'P4': [0.04, -0.041],
       'T6': [0.07, -0.047],
       'O1': [-0.03, -0.08],
       'O2': [0.03, -0.08]}
channels = pd.DataFrame(chs).transpose()

for key in chs.keys():
    chs[key]+=[0]

mont = mne.channels.make_dig_montage(chs)
#mont.plot()
#plt.show()  #EEG'nın alındığı bölgeleri çizer

#Verilerin tanımlanması
df = pd.read_csv('C:/Users/DİGİTAL/Desktop/eeg.csv')

#Kayıp veriler
mis = df.isna().sum()
sep_col = mis[mis == df.shape[0]].index[0]
df = df.loc[:, 'mainDisorder':sep_col].drop(sep_col, axis=1)


def reformat_name(name):
    '''
    reformat from XX.X.band.x.channel to band.channel
    '''
    band, _, channel = name[5:].split(sep='.')
    return f'{band}.{channel}'

reformat_vect = np.vectorize(reformat_name)
new_colnames = np.concatenate((df.columns[:2],reformat_vect(df.columns[2:])))
df.set_axis(new_colnames, axis=1, inplace=True)
