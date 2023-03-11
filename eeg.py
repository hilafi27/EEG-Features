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

#Panic Rahatsızlığı verileri ve öznitelik bulma
panicData = df[df.specificDisorder == "Panic disorder"]
bant =panicData.index 

panic_varyans = pd.DataFrame.var(panicData.iloc[:,2:], axis=1)
bantp = panic_varyans.index=bant

panic_std= pd.DataFrame.std(panicData.iloc[:,2:], axis=1)
panic_std.index=bant

panic_mean = pd.DataFrame.mean(panicData.iloc[:,2:], axis=1)
panic_mean.index=bant

panic_carpiklik = scipy.stats.skew(panicData.iloc[:,2:], axis=1) #çarpıklık
panic_skewness = pd.Series(index=bantp,data=panic_carpiklik)

panic_basiklik = scipy.stats.kurtosis(panicData.iloc[:,2:], axis=1) #basıklık
panic_kurtosis = pd.Series(index=bantp,data=panic_basiklik)

pE= entropy(panicData.iloc[:,2:], axis=1 , base=59)
panicEntropy=pd.Series(index=bantp,data=pE)

pne=max(pE)-pE
panic_negengtropi=pd.Series(index=bantp,data=pne)

panic=pd.DataFrame.transpose(panicData)
pdata = panic.iloc[2:,:]

pcoef , pfreq  = pywt.cwt(pdata, np.arange(1,60),'morl' )
pfreq2=pd.Series(index=bantp,data=pfreq)
#plt.matshow(pcoef) 
#plt.show() 

#hjort
pactivity = eeglib.features.hjorthActivity(panic.iloc[2:,:])
pactivity.index=bant

pcomplexity = eeglib.features.hjorthComplexity(panic.iloc[2:,:])
pcomplexity.index=bant

pmobility = eeglib.features.hjorthMobility(panic.iloc[2:,:])
pmobility.index=bant

new_data=[panic_varyans,panic_std,panic_mean,panic_skewness,panic_kurtosis,panicEntropy,
          panic_negengtropi,pactivity,pcomplexity,pmobility,pfreq2]
datalar=pd.DataFrame(new_data)
datap=pd.DataFrame.transpose(datalar)

#Anxiety Rahatsızlığı verileri ve öznitelik bulma
anxietyData = df[df.specificDisorder == "Social anxiety disorder"]
banta =anxietyData.index 

anxiety_varyans = pd.DataFrame.var(anxietyData.iloc[:,2:],axis=1)
bantan = anxiety_varyans.index=banta

anxiety_std = pd.DataFrame.std(anxietyData.iloc[:,2:],axis=1)
anxiety_std.index=banta

anxiety_mean = pd.DataFrame.mean(anxietyData.iloc[:,2:],axis=1)
anxiety_mean.index=banta

anxiety_carpiklik = scipy.stats.skew(anxietyData.iloc[:,2:],axis=1) #çarpıklık
anxiety_skewness = pd.Series(index=banta,data=anxiety_carpiklik)

anxiety_basiklik = scipy.stats.kurtosis(anxietyData.iloc[:,2:],axis=1) #basıklık
anxiety_kurtosis = pd.Series(index=banta,data=anxiety_basiklik)

aE= entropy(anxietyData.iloc[:,2:], axis=1,base=50)
anxietyEntropy=pd.Series(index=banta,data=aE)

ane=max(aE)-aE
anxiety_negentropi=pd.Series(index=banta,data=ane)
anxiety=pd.DataFrame.transpose(anxietyData)

adata = anxiety.iloc[2:,:]
acoef , afreq  = pywt.cwt(adata , np.arange(1,49), 'morl')
afreq2=pd.Series(index=banta,data=afreq)
#plt.matshow(acoef) 
#plt.show() 

#hjort
aactivity = eeglib.features.hjorthActivity(anxiety.iloc[2:,:])
aactivity.index=banta

acomplexity = eeglib.features.hjorthComplexity(anxiety.iloc[2:,:])
acomplexity.index=banta

amobility = eeglib.features.hjorthMobility(anxiety.iloc[2:,:])
amobility.index=banta

new_data1=[anxiety_varyans,anxiety_std,anxiety_mean,anxiety_skewness,anxiety_kurtosis,
           anxietyEntropy,anxiety_negentropi,aactivity,acomplexity,amobility,afreq2]
datalar1=pd.DataFrame(new_data1)
dataa=pd.DataFrame.transpose(datalar1)

datah=datap.append(dataa,ignore_index=True)

banta=["anxiety"] * 48
bant=["panic"] * 59
newcolumn=list(bant+banta)
datah['hastalik']=pd.Series(newcolumn)
datah.rename(columns = { 0:'varyans', 1:'std',2:'ortalama',3:'skewness',4:'kurtosis',5:'entropi',
                         6 :'neg_entropi',7:'h_activity',8:'h_complexity',9:'h_mobility',10:'w_freq'}, inplace = True)

datah = datah.sample(frac=1).reset_index(drop=True)
datah.to_csv('feature_pa.csv')
