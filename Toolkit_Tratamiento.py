# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:41:02 2022

@author: user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as tm
import os as os

def setTratamientoCotasSup(base,var_T,q):
    p=base[var_T].quantile(q=q)
    qq=(q+0.05) if (q+0.05<1) else q
    print(base[var_T].quantile(q=[q-0.05,q,qq]))
    print(f"{var_T}(Percentil_{int(q*100)}: {p})")
    base[f"{var_T}_TCS"]=base[var_T].apply(lambda x: p if x>=p else x)
    base[f"{var_T}_TCS"].plot(kind="box")

def setTratamientoCotasInf(base,var_T,q):
    p=base[var_T].quantile(q=q)
    qq=(q-0.05) if (q-0.05>0) else q
    print(base[var_T].quantile(q=[qq,q,q+0.05]))
    print(f"{var_T}(Percentil_{int(q*100)}: {p})")
    base[f"{var_T}_TCI"]=base[var_T].apply(lambda x: p if x<=p else x)
    base[f"{var_T}_TCI"].plot(kind="box")

def setTratamientoMissingsVarContinua(base,var_T):
    imputer=base[var_T].mean()
    base[f"{var_T}_TM"]=base[var_T].fillna(imputer)
    print(f"{var_T}_TM(Mean: {imputer})")
    base[f"{var_T}_TM"].plot(kind="hist")
    
# Modelamiento
def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    dset = dset.sort_values(by='WoE')
    return iv #dset, iv

def getFeactureIV_Importance(df,feactures,target):
    feactureIV_Importance=list()
    #feacture,target
    for v in feactures:
      iv=calculate_woe_iv(df,v,target)
      feactureIV_Importance.append(iv)
    #pd.DataFrame({"Feacture":feactures,"IV":feactureIV_Importance}).sort_values("IV")
    display(pd.DataFrame({"Feacture":feactures,"IV":feactureIV_Importance}).sort_values("IV"))