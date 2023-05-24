import datetime
import numpy as np
import pyelf.elutil as eu
import pandas as pd
import itertools
from .Factor_Industry import *

def ResampleDayDF(df, cperiod, rule='mean', shift=0, if_reset_index=True, if_cuttime=False, cuttime_st=None, cuttime_ed=None, if_indexC=False):

    idf = df.copy()
    cols_ = idf.columns.tolist()
    idf['index'] = idf.index
    idf['indexC'] = [ix+datetime.timedelta(minutes=cperiod) for ix in idf.index]

    index0 = idf[idf['indexC'].apply(lambda x: x.time())==datetime.time(15,0)]['indexC'].values

    idf['day_clabel'] = 1*(idf['indexC'].apply(lambda x: x.time()) == datetime.time(15,0))
    idf['day_olabel'] = idf['day_clabel'].shift(1+shift)

    idf.loc[idf.index[0+shift], 'day_olabel'] = 1
    # idf['day_olabel'] = idf['day_olabel'].fillna(0)
    idf['day_clabel'] = idf['day_clabel'].cumsum()
    idf['day_olabel'] = idf['day_olabel'].cumsum()
    idf['day_olabel'] = idf['day_olabel'].fillna(method='bfill')

    if if_cuttime:
        if cuttime_st is not None:
            idf = idf[idf['index'].apply(lambda x: x.time())>=cuttime_st]
        if cuttime_ed is not None:
            idf = idf[idf['index'].apply(lambda x: x.time())<cuttime_ed]


    if rule=='max':
        idf_out = idf[cols_].groupby(idf['day_olabel']).max()
    elif rule=='min':
        idf_out = idf[cols_].groupby(idf['day_olabel']).min()
    elif rule=='first':
        idf_out = idf[cols_].groupby(idf['day_olabel']).first()
    elif rule=='sum':
        idf_out = idf[cols_].groupby(idf['day_olabel']).sum()
    elif rule=='mean':
        idf_out = idf[cols_].groupby(idf['day_olabel']).mean()
    elif rule=='last':
        idf_out = idf[cols_].groupby(idf['day_olabel']).last()
    else:
        idf_out = idf[cols_].groupby(idf['day_olabel']).last()

    if if_indexC:
        idf_out.index = idf['indexC'].groupby(idf['day_olabel']).last()
    else:
        idf_out.index = idf['index'].groupby(idf['day_olabel']).last()
    if if_reset_index:
        idf_out.index = index0
    idf_out.index.name = 'index'

    return idf_out

def GSR(xdf):
    xdf_max = xdf.max(axis=0)
    xdf_min = xdf.min(axis=0)
    xdf_new = xdf.copy()
    xdf_new[xdf.sub(xdf_max, axis=1)==0] = np.nan
    xdf_new[xdf.sub(xdf_min, axis=1)==0] = np.nan
    return xdf_new.mean(axis=0,skipna=True)/xdf_new.std(axis=0,skipna=True)

def GSRII(xdf):
    xdf_new = xdf.copy()
    xdf_open = xdf_new.iloc[:1,]
    xdf_newII = xdf_new.iloc[1:]
    # xdf_new.loc[xdf_new.index[0], xdf_open.mean()[((xdf_newII.mean()*xdf_open.mean()).apply(np.sign)<0)].index.tolist()] = np.nan
    # scomms = xdf_open.mean()[((xdf_newII.mean()*xdf_open.mean()).apply(np.sign)<0)].index.tolist()
    scomms = xdf_open.mean()[(((xdf_newII.mean()*xdf_open.mean()).apply(np.sign)<0)&(((xdf_newII.fillna(0)+1).prod()-1).abs()>0.02))].index.tolist()
    xdf_new.loc[xdf_new.index[0], scomms] = -1*xdf_new.loc[xdf_new.index[0], scomms]
    return xdf_new.mean(axis=0,skipna=True)/xdf_new.std(axis=0,skipna=True)

def GetKN(df, cperiod, dayed_time=datetime.time(15,0)):
    shift = 0
    idf = df.copy()

    idf['index'] = idf.index
    idf['indexC'] = [ix+datetime.timedelta(minutes=cperiod) for ix in idf.index]

    idf['day_clabel'] = 1*(idf['indexC'].apply(lambda x: x.time()) == dayed_time)
    idf['day_olabel'] = idf['day_clabel'].shift(1+shift)
    idf.loc[idf.index[0+shift], 'day_olabel'] = 1

    idf['kn'] = range(len(idf))
    idf['kn_open'] = np.nan
    idf.loc[idf['day_olabel']==1, 'kn_open'] = idf.loc[idf['day_olabel']==1, 'kn']
    idf['kn_open'] = idf['kn_open'].fillna(method='ffill')

    idf['kn'] = idf['kn']-idf['kn_open']

    return idf['kn']

def ResampleDayCumFunc(df, cperiod, rule='sum', dayed_time=datetime.time(15,0)):
    idf = df.copy()
    cols_ = idf.columns.tolist()
    idf['index'] = idf.index
    idf['indexC'] = [ix+datetime.timedelta(minutes=cperiod) for ix in idf.index]

    shift = 0
    idf['day_clabel'] = 1*(idf['indexC'].apply(lambda x: x.time()) == dayed_time)
    idf['day_olabel'] = idf['day_clabel'].shift(1+shift)
    idf.loc[idf.index[0+shift], 'day_olabel'] = 1

    idf['kn'] = range(len(idf))
    idf['kn_open'] = np.nan
    idf.loc[idf['day_olabel']==1, 'kn_open'] = idf.loc[idf['day_olabel']==1, 'kn']
    idf['kn_open'] = idf['kn_open'].fillna(method='ffill')

    idf['kn'] = idf['kn']-idf['kn_open']

    idf['day_clabel'] = idf['day_clabel'].cumsum()
    idf['day_olabel'] = idf['day_olabel'].cumsum()
    idf['day_olabel'] = idf['day_olabel'].fillna(method='bfill')

    if rule=='sum':
        idf_out = idf[cols_].groupby(idf['day_olabel']).cumsum()
    elif rule=='max':
        idf_out = idf[cols_].groupby(idf['day_olabel']).cummax()
    elif rule=='min':
        idf_out = idf[cols_].groupby(idf['day_olabel']).cummin()
    elif rule=='count':
        idf_out = idf[cols_].groupby(idf['day_olabel']).cumcount()
    elif rule=='mean':
        idf_out = idf[cols_].groupby(idf['day_olabel']).cumsum().div(1+idf[cols_].groupby(idf['day_olabel']).cumcount(), axis=0)
    
    return idf_out

def ResampleDayDFKN(df, cperiod, rule='mean', shift=0, if_resample_index=True, if_cutkn=False, cutkn_st=None, cutkn_ed=None, if_indexC=False, dayed_time=datetime.time(15,0), aux_data={}):
    idf = df.copy()
    cols_ = idf.columns.tolist()
    idf['index'] = idf.index
    idf['indexC'] = [ix+datetime.timedelta(minutes=cperiod) for ix in idf.index]


    idf['day_clabel'] = 1*(idf['indexC'].apply(lambda x: x.time()) == dayed_time)
    idf['day_olabel'] = idf['day_clabel'].shift(1+shift)
    idf.loc[idf.index[0+shift], 'day_olabel'] = 1

    idf['kn'] = range(len(idf))
    idf['kn_open'] = np.nan
    idf.loc[idf['day_olabel']==1, 'kn_open'] = idf.loc[idf['day_olabel']==1, 'kn']
    idf['kn_open'] = idf['kn_open'].fillna(method='ffill')

    idf['kn'] = idf['kn']-idf['kn_open']

    idf['day_clabel'] = idf['day_clabel'].cumsum()
    idf['day_olabel'] = idf['day_olabel'].cumsum()
    idf['day_olabel'] = idf['day_olabel'].fillna(method='bfill')

    index0 = idf['indexC'].groupby(idf['day_olabel']).last()

    if if_cutkn:
        if cutkn_st is not None:
            idf = idf[idf['kn']>=cutkn_st]
        if cutkn_ed is not None:
            idf = idf[idf['kn']<cutkn_ed]


    if rule=='max':
        idf_out = idf[cols_].groupby(idf['day_olabel']).max()
    elif rule=='min':
        idf_out = idf[cols_].groupby(idf['day_olabel']).min()
    elif rule=='first':
        idf_out = idf[cols_].groupby(idf['day_olabel']).first()
    elif rule=='sum':
        idf_out = idf[cols_].groupby(idf['day_olabel']).sum()
    elif rule=='mean':
        idf_out = idf[cols_].groupby(idf['day_olabel']).mean()
    elif rule=='vmean':
        V = aux_data['V']
        idf_out = (V[cols_]*idf[cols_]).groupby(idf['day_olabel']).sum()/(V[cols_]).groupby(idf['day_olabel']).sum()
    elif rule=='last':
        idf_out = idf[cols_].groupby(idf['day_olabel']).last()
    elif rule=='std':
        idf_out = idf[cols_].groupby(idf['day_olabel']).std()
    elif rule=='skew':
        idf_out = idf[cols_].groupby(idf['day_olabel']).skew()
    elif rule=='kurt':
        idf_out = idf[cols_].groupby(idf['day_olabel']).kurt()
    elif rule=='max_zs':
        idf_out = (idf[cols_].groupby(idf['day_olabel']).max()-idf[cols_].groupby(idf['day_olabel']).mean())/idf[cols_].groupby(idf['day_olabel']).std()
    elif rule=='sr':
        idf_out = idf[cols_].groupby(idf['day_olabel']).mean()/idf[cols_].groupby(idf['day_olabel']).std()
    elif rule=='gsr':
        idf_out = idf[cols_].groupby(idf['day_olabel']).apply(GSRII)
    elif rule=='pos':
        idf_out = (idf[cols_].groupby(idf['day_olabel']).last()-idf[cols_].groupby(idf['day_olabel']).min())/(idf[cols_].groupby(idf['day_olabel']).max()-idf[cols_].groupby(idf['day_olabel']).min())
    elif rule=='me_pos':
        idf_out = (idf[cols_].groupby(idf['day_olabel']).mean()-idf[cols_].groupby(idf['day_olabel']).min())/(idf[cols_].groupby(idf['day_olabel']).max()-idf[cols_].groupby(idf['day_olabel']).min())
    elif rule=='me_minmax':
        idf_out = (idf[cols_].groupby(idf['day_olabel']).mean())/(idf[cols_].groupby(idf['day_olabel']).max()-idf[cols_].groupby(idf['day_olabel']).min())
    else:
        idf_out = idf[cols_].groupby(idf['day_olabel']).last()

    if if_indexC:
        idf_out.index = idf['indexC'].groupby(idf['day_olabel']).last()
    else:
        idf_out.index = idf['index'].groupby(idf['day_olabel']).last()

    if if_resample_index:
        idf_out.index = index0
    idf_out.index.name = 'index'

    return idf_out

def GetKNSR(cutkn_ed, factor_diff, cperiod):
    idf = ResampleDayDFKN(factor_diff, cperiod, rule='sr', shift=0, if_resample_index=False, if_cutkn=True, cutkn_st=None, cutkn_ed=cutkn_ed, if_indexC=False, dayed_time=datetime.time(15,0), aux_data={})
    return idf
    
def DayCumSR(factor_diff, cperiod, kn_st, kn_ed):
    allkns = list(range(kn_st+1, kn_ed+2, 1))
    df_list = eu.multiprocess(GetKNSR, paras=allkns, n_processes=100, factor_diff=factor_diff, cperiod=cperiod)
    df_concat = pd.concat(df_list, axis=0)
    df_concat = df_concat.sort_index()
    df_concat = df_concat[~df_concat.index.duplicated(keep='first')]
    return df_concat

def med_extrem_cut(factor):
    facmid = factor.median(axis=1)
    factor_mid_abs = factor.sub(facmid, axis=0).abs().median(axis=1)
    up = facmid + 3 * 1.4826 * factor_mid_abs
    down = facmid - 3 * 1.4826 * factor_mid_abs
    factor = factor.clip(down, up, axis=0)
    if factor.std(axis=0).mean() < 0.000001:
        factor = factor * 0
    factor.fillna(0, inplace=True)
    return factor

def factor_hznm(factordf_in, if_med_cut=True, if_jy=None):
    factordf = factordf_in.copy()
    if if_jy is not None:
        factordf[if_jy==0] = np.nan
    if if_med_cut:
        factordf = med_extrem_cut(factordf)
    mean_ = factordf.mean(axis=1)
    std_ = factordf.std(axis=1)
    factordf_norm = (factordf.subtract(mean_, axis=0)).div(std_, axis=0)
    return factordf_norm

def GetFactor(para, data):
    factorname, xp = para
    factordf, name_ = eval(factorname)(data, xp)

    factordf[data['if_jy']==0] = np.nan

    return factordf, name_

def GetFactorCorr(factor1, factor2, type='val', ydf0=None):
    if type=='val':
        return factor1.corrwith(factor2, axis=1).mean()
    elif type=='gp':
        factor_rank1 = factor1.rank(pct=True, axis=1)
        psdf1 = 2*(factor_rank1-0.5)

        factor_rank2 = factor2.rank(pct=True, axis=1)
        psdf2 = 2*(factor_rank2-0.5)

        tradedf1 = psdf1*ydf0
        tradedf2 = psdf2*ydf0

        return tradedf1.corrwith(tradedf2, axis=1).mean()

def SingleCorr(para, factordf_dict, type='val', ydf0=None):
    name1, name2 = para
    return GetFactorCorr(factordf_dict[name1], factordf_dict[name2], type=type, ydf0=ydf0)

def GetCorrMatrix(factordf_dict, type='val', ydf0=None):
    name_list = list(factordf_dict.keys())
    allparas = list(itertools.product(name_list, name_list))
    results = eu.multiprocess(SingleCorr, paras=allparas, n_processes=min(50, len(allparas)), factordf_dict=factordf_dict, type=type, ydf0=ydf0)
    corr_dict = dict(zip(allparas, results))
    corrdf = pd.DataFrame(index=name_list, columns=name_list)
    for name1 in name_list:
        for name2 in name_list:
            corrdf.loc[name1, name2] = corr_dict[(name1, name2)]
    return corrdf