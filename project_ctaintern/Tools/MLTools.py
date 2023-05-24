# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:37:29 2020

@author: KXL
"""
import pandas as pd
import os
import datetime
import pyelf.elutil as eu

import numpy as np

import pyelf.performance as pm
import pyelf.elplot as elplot
import matplotlib.pyplot as plt


from scipy.stats import pearsonr,spearmanr

import platform

import matplotlib.dates as mdates
import bisect

if int(platform.python_version().split('.')[0])>=3:
    import joblib
else:
    from sklearn.externals import joblib

import warnings

warnings.filterwarnings('ignore')



def LoadData(filename, date_b=None,date_e=None,ref_ind=None,time_str=None,cols=None,if_day=False, if_parquet=False):
    if if_parquet:
        tmpdf = pd.read_parquet(filename)
    else:
        tmpdf = pd.read_csv(filename)
        cols0 = tmpdf.columns[1:]
        if time_str is not None:
            tmpdf['time'] = time_str
            tmpdf[tmpdf.columns[0]] = pd.to_datetime(tmpdf[tmpdf.columns[0]]+' '+tmpdf['time'])
        else:
            tmpdf[tmpdf.columns[0]] = pd.to_datetime(tmpdf[tmpdf.columns[0]])
        tmpdf = tmpdf.set_index(tmpdf.columns[0])
    if date_b is not None:
        tmpdf = tmpdf[tmpdf.index>=date_b]
    if date_e is not None:
        tmpdf = tmpdf[tmpdf.index<date_e]
    if if_day:
        tmpdf = tmpdf.resample('1D').last()
        tmpdf = tmpdf.dropna(how='all')
    if ref_ind is not None:
        tmpdf = tmpdf[~tmpdf.index.duplicated()]
        tmpdf = tmpdf.reindex(ref_ind)
        tmpdf = tmpdf.fillna(method='ffill')
    tmpdf.index.name='index'
    if cols is not None:
        cols0 = cols
    tmpdf = tmpdf[~tmpdf.index.duplicated()]
    return tmpdf[cols0]

def CTA_LoadData(rootdir, period, aux_cols=[], date_b=None, date_e=None, ref_ind=None, cols=None, if_day=False, if_tur=False):
    O = LoadData(os.path.join(rootdir, '%dmin' % period, 'O-%dmin.csv' % period), date_b=date_b, date_e=date_e, ref_ind=ref_ind,
                 cols=cols, if_day=if_day)
    C = LoadData(os.path.join(rootdir, '%dmin' % period, 'C-%dmin.csv' % period), date_b=date_b, date_e=date_e, ref_ind=ref_ind,
                 cols=cols, if_day=if_day)
    H = LoadData(os.path.join(rootdir, '%dmin' % period, 'H-%dmin.csv' % period), date_b=date_b, date_e=date_e, ref_ind=ref_ind,
                 cols=cols, if_day=if_day)
    L = LoadData(os.path.join(rootdir, '%dmin' % period, 'L-%dmin.csv' % period), date_b=date_b, date_e=date_e, ref_ind=ref_ind,
                 cols=cols, if_day=if_day)
    V = LoadData(os.path.join(rootdir, '%dmin' % period, 'V-%dmin.csv' % period), date_b=date_b, date_e=date_e, ref_ind=ref_ind,
                 cols=cols, if_day=if_day)
    I = LoadData(os.path.join(rootdir, '%dmin' % period, 'OPI-%dmin.csv' % period), date_b=date_b, date_e=date_e, ref_ind=ref_ind,
                 cols=cols, if_day=if_day)

    eps = 0.0001
    O[O < eps] = np.nan
    O = O.fillna(method='ffill')
    C[C < eps] = np.nan
    C = C.fillna(method='ffill')
    H[H < eps] = np.nan
    H = H.fillna(method='ffill')
    L[L < eps] = np.nan
    L = L.fillna(method='ffill')
    I[I < eps] = np.nan
    I = I.fillna(method='ffill')

    if if_tur:
        Tur = LoadData(os.path.join(rootdir, '%dmin' % period, 'Tur-%dmin.csv' % period), date_b=date_b, date_e=date_e, 
                       ref_ind=ref_ind, cols=cols, if_day=if_day)
        Tur[Tur < eps] = np.nan
        Tur = Tur.fillna(method='ffill')

    data = {}
    data['period'] = period
    data['O'] = O
    data['C'] = C
    data['H'] = H
    data['L'] = L
    data['V'] = V
    data['I'] = I
    if if_tur:
        data['Tur'] = Tur

    if len(aux_cols) > 0:
        for icol in aux_cols:
            iaux = LoadData(os.path.join(rootdir, '%dmin' % period, '%s-%dmin.csv' % (icol, period)), date_b=date_b, date_e=date_e,
                            ref_ind=ref_ind, cols=cols, if_day=if_day)
            data[icol] = iaux

    return data

def CTA_LoadData_AutoFactor(rootdir, period, shift=0, aux_cols=[], date_b=None, date_e=None, ref_ind=None, cols=None, if_day=False, if_parquet=True):
    data = {}
    data['period'] = period

    names_dict = {}
    names_dict['O'] = 'openprice'
    names_dict['H'] = 'highprice'
    names_dict['L'] = 'lowprice'
    names_dict['C'] = 'closeprice'
    names_dict['V'] = 'volume'
    names_dict['I'] = 'opi'

    for ikey in names_dict.keys():
        idf = LoadData(os.path.join(rootdir, '%dmin_shift%d' % (period, shift), '%s.parquet' % names_dict[ikey]), date_b=date_b, date_e=date_e, ref_ind=ref_ind,
                    cols=cols, if_day=if_day, if_parquet=if_parquet)
        data[ikey] = idf
    

    if len(aux_cols) > 0:
        for icol in aux_cols:
            iaux = LoadData(os.path.join(rootdir, '%dmin_shift%d' % (period, shift), '%s.parquet' % icol), date_b=date_b, date_e=date_e,
                            ref_ind=ref_ind, cols=cols, if_day=if_day, if_parquet=if_parquet)
            data[icol] = iaux

    return data


def myMakeDir(dstdir):
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)

def GetPerfMetric(profit):
    profit = profit - profit.iloc[0]
    sr = profit.diff().mean()/profit.diff().std()
    sr = sr*np.sqrt(252)
    
    dd = profit.cummax()-profit
    mdd = dd.max()
    
    rf = profit.iloc[-1]/mdd
    
    dprofit = profit.diff()
    dprofit.iloc[0] = profit.iloc[0]
    dayp = dprofit.values
    
    p = len(dayp[dayp>0])*1.0/len(dayp)
    q = 1-p
    b = np.mean(dayp[dayp>0])/abs(np.mean(dayp[dayp<0]))
    f = p-q/b
    
    # print profit.iloc[-1],sr,rf,p,b,f
    
    return profit.iloc[-1],sr,rf,p,b,f


def CalculateTur(psdf):
    trades_vol = abs(psdf.diff()).sum(axis=1)
    sum_pos = abs(psdf).sum(axis=1)
    
    trades_vol = trades_vol.resample('1D').sum().dropna(how='all')
    sum_pos = sum_pos.resample('1D').mean().dropna(how='all')
    
    trades_vol = trades_vol.reindex(sum_pos.index)
    
    avg_tur = trades_vol.mean()/sum_pos.mean()
    return avg_tur


def getStdFactor(inData, cuts=5, ifqcut=True):
    Data = inData.dropna()

    per95 = Data.quantile(0.975, interpolation='nearest')
    per05 = Data.quantile(0.025, interpolation='nearest')
    Data.loc[Data > per95] = per95
    Data.loc[Data < per05] = per05
    eps = 0.001
    ibins = np.linspace(per05-eps, per95+eps, cuts+1, endpoint=True)
    ilabels = list(range(1, cuts+1, 1))
    
    try:
        tmpfactor = pd.qcut(Data, q=cuts, duplicates='drop')
    except:
        return pd.Series(index=inData.index)
    binsnum = len(tmpfactor.cat.categories)
    ilabels = ilabels[:binsnum]

    if ifqcut:
        factor,ibins = pd.qcut(Data, q=cuts, labels=ilabels, duplicates='drop', retbins=True)
    else:
        factor = pd.cut(Data, bins=ibins, labels=ilabels)

    factor = factor.astype(float)
    factor = factor.reindex(inData.index)

    return factor


def FactorPerform_HZ_Old(factor_df, y_df, plt_ycol='', bin_ycols=[], cuts=2, ifqcut=True, if_savefig=False, figpath='', nameflag_='', if_show_ls=True, if_show_fig=True, if_show_gp=False):
    # print('calculate IC')

    factor_df = factor_df.dropna(how='all')
    y_df = y_df.reindex(factor_df.index)
    
    icdf = factor_df.corrwith(y_df, axis=1)
    icdfM = icdf.resample('1M').mean()
    icmean = 1 * icdfM.mean()
    icir = icmean / icdfM.std()
    
    icdf = icdf.replace([np.inf, -np.inf], np.nan).fillna(0)
    icdf = icdf.cumsum()
    
    ic_sr = np.sqrt(252)*icdf.diff().mean()/icdf.diff().std()
    
    TN = (~pd.isnull(factor_df)).sum(axis=1)
    TNbias = 1.0/TN
    
    # print('factor_group')
    eps = 0.001
    factor_rank = factor_df.rank(pct=True, axis=1)
    factor_rank = factor_rank.sub(TNbias/2.0,axis=0)
    factor_rank = factor_rank.fillna(0.5)
    factor_rank[factor_rank>1-eps] = 1-eps
    factor_rank[factor_rank<eps] = eps
    factor_group = (factor_rank*cuts).apply(np.floor)+1
    factor_group = factor_group.astype(int)
    factor_group[factor_df.apply(np.isnan)==True]=np.nan
    factor_group = factor_group.dropna(how='all')
    
    psdf = 2 * ((factor_group-1)*1.0/(cuts-1) - 0.5)

    
    avg_tur = CalculateTur(psdf)


    rankdf = factor_df.rank(pct=True, axis=1)
    totaldf = rankdf.sum(axis=1)

    psdf0 = rankdf.div(totaldf, axis=0)
    avg_psdf0 = psdf0.mean(axis=1)

    psdf1 = psdf0.sub(avg_psdf0, axis=0)
    
    avg_tur1 = CalculateTur(psdf1)

    gpdf = pd.DataFrame()
    gpdf['gp'] = (psdf1.fillna(0)*y_df.fillna(0)).sum(axis=1).cumsum()
    
    sr1 = np.sqrt(252)*gpdf['gp'].diff().mean()/gpdf['gp'].diff().std()

    
    # print('plot')
    y_df = y_df.reindex(factor_group.index)
    y_group = pd.DataFrame(index=list(range(1,cuts+1,1)))
    yg_ts = pd.DataFrame(index=factor_df.index)
    y_group['y'] = np.nan
    for i in range(1,cuts+1,1):
        y_group.loc[i, 'y'] = y_df[factor_group==i].unstack().dropna().mean()
        yg_ts['y_%d'%i] = y_df[factor_group==i].mean(axis=1)
    
    yg_ts = yg_ts.fillna(0).cumsum()
    plt_cols = yg_ts.columns.tolist()
    
    yg_ts['LongShort'] = yg_ts['y_%d'%(cuts)]-yg_ts['y_%d'%(1)]


    if if_show_ls:
        plt_cols.append('LongShort')
    
    if if_show_ls:
        sr = np.sqrt(252)*yg_ts['LongShort'].diff().mean()/yg_ts['LongShort'].diff().std()
    else:
        sr = ic_sr
    
    yvalues = y_group.values
    xvalues = list(range(cuts))
    icvalue = spearmanr(xvalues, yvalues.reshape(1,cuts).tolist()[0])[0]
    deltanp = yg_ts['y_%d'%(cuts)].values[-1]-yg_ts['y_%d'%(1)].values[-1]
    
#     yg_ts = yg_ts[['LongShort']]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 24))
    if nameflag_=='':
        nameflag = '%s-(%.1f, %.2f, %.2f)'%('TT',sr, avg_tur, icmean)
    else:
        nameflag = '%s-(%.1f, %.2f, %.2f)'%(nameflag_, sr, avg_tur, icmean)

    y_group.plot(ax=ax1, kind='bar', grid=True, title=nameflag)
    yg_ts[plt_cols].plot(ax=ax2, grid=True, title='sr %.2f'%(sr))
    if if_show_gp:
        gpdf['gp'].plot(ax=ax3, grid=True, title='SR %.2f, tor %.2f'%(sr1, avg_tur1))
    else:
        icdf.plot(ax=ax3, grid=True, title='ICmean %.2f, ICIR %.1f, ICSR %.2f, tor %.2f'%(icmean, icir, ic_sr, avg_tur))
    
    
    if if_savefig:
        myMakeDir(figpath)
        plt.savefig(os.path.join(figpath, '%s.png'%(nameflag)))
    if if_show_fig:
        plt.show()
    plt.close()
    return nameflag.split('-')[0], icvalue, deltanp, sr, factor_group


def FactorPerform_HZ(factor_df, y_df, plt_ycol='', bin_ycols=[], cuts=2, ifqcut=True, if_savefig=False, figpath='', nameflag_='', if_show_ls=True, if_show_fig=True, if_show_gp=False, ic_mode=1, plt_date=None):
    # print('calculate IC')

    if plt_date is not None:
        factor_df = factor_df[factor_df.index>plt_date]
        y_df = y_df[y_df.index>plt_date]

    factor_df = factor_df.dropna(how='all')
    y_df = y_df.reindex(factor_df.index)
    
    icdf = factor_df.corrwith(y_df, axis=1)
    icdfM = icdf.resample('1M').mean()
    icmean = 1 * icdfM.mean()
    icir = icmean / icdfM.std()
    
    icdf = icdf.replace([np.inf, -np.inf], np.nan).fillna(0)
    icdf = icdf.cumsum()

    icdf_day = icdf.resample('1D').last().dropna(how='all')
    
    ic_sr = np.sqrt(252)*icdf_day.diff().mean()/icdf_day.diff().std()
    
    TN = (~pd.isnull(factor_df)).sum(axis=1)
    TNbias = 1.0/TN
    
    # print('factor_group')
    eps = 0.001
    factor_rank = factor_df.rank(pct=True, axis=1)
    factor_rank = factor_rank.sub(TNbias/2.0,axis=0)
    factor_rank = factor_rank.fillna(0.5)
    factor_rank[factor_rank>1-eps] = 1-eps
    factor_rank[factor_rank<eps] = eps
    factor_group = (factor_rank*cuts).apply(np.floor)+1
    factor_group = factor_group.astype(int)
    factor_group[factor_df.apply(np.isnan)==True]=np.nan
    factor_group = factor_group.dropna(how='all')
    
    psdf = 2 * ((factor_group-1)*1.0/(cuts-1) - 0.5)

    
    avg_tur = CalculateTur(psdf)


    rankdf = factor_df.rank(pct=True, axis=1)
    totaldf = rankdf.sum(axis=1)

    psdf0 = rankdf.div(totaldf, axis=0)
    avg_psdf0 = psdf0.mean(axis=1)

    psdf1 = psdf0.sub(avg_psdf0, axis=0)
    
    avg_tur1 = CalculateTur(psdf1)

    gpdf = pd.DataFrame()
    gpdf['gp'] = (psdf1.fillna(0)*y_df.fillna(0)).sum(axis=1).cumsum()
    
    sr1 = np.sqrt(252)*gpdf['gp'].diff().mean()/gpdf['gp'].diff().std()

    
    # print('plot')
    y_df = y_df.reindex(factor_group.index)
    y_group = pd.DataFrame(index=list(range(1,cuts+1,1)))
    yg_ts = pd.DataFrame(index=factor_df.index)
    y_group['y'] = np.nan
    for i in range(1,cuts+1,1):
        y_group.loc[i, 'y'] = y_df[factor_group==i].unstack().dropna().mean()
        yg_ts['y_%d'%i] = y_df[factor_group==i].mean(axis=1)
    
    yg_ts = yg_ts.fillna(0).cumsum()
    plt_cols = yg_ts.columns.tolist()
    
    yg_ts['LongShort'] = yg_ts['y_%d'%(cuts)]-yg_ts['y_%d'%(1)]
    yg_ts_day = yg_ts.resample('1D').last().dropna(how='all')


    if if_show_ls:
        plt_cols.append('LongShort')
    
    if if_show_ls:
        sr = np.sqrt(252)*yg_ts_day['LongShort'].diff().mean()/yg_ts_day['LongShort'].diff().std()
    else:
        sr = ic_sr
    
    yvalues = y_group.values
    xvalues = list(range(cuts))
    icvalue = spearmanr(xvalues, yvalues.reshape(1,cuts).tolist()[0])[0]
    deltanp = yg_ts['y_%d'%(cuts)].values[-1]-yg_ts['y_%d'%(1)].values[-1]
    
#     yg_ts = yg_ts[['LongShort']]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 24))
    if nameflag_=='':
        nameflag = '%s-(%.1f, %.2f, %.2f)'%('TT',sr, avg_tur, icmean)
    else:
        nameflag = '%s-(%.1f, %.2f, %.2f)'%(nameflag_, sr, avg_tur, icmean)

    y_group.plot(ax=ax1, kind='bar', grid=True, title=nameflag)
    yg_ts[plt_cols].plot(ax=ax2, grid=True, title='sr %.2f'%(sr))
    if if_show_gp:
        gpdf['gp'].plot(ax=ax3, grid=True, title='SR %.2f, tor %.2f'%(sr1, avg_tur1))
    else:
        if ic_mode==0:
            icdf.plot(ax=ax3, grid=True, title='ICmean %.2f, ICIR %.2f, ICSR %.2f, tor %.2f'%(icmean, icir, ic_sr, avg_tur))
        elif ic_mode==1:
            fonsize = 24
            ax3.bar(icdfM.index.values, icdfM.values, width=fonsize/2, color='#1f77b4')

            mons = mdates.MonthLocator(interval=3)
            mons_ = mdates.DateFormatter('%Y-%m')
            ax3.xaxis.set_major_locator(mons)
            ax3.xaxis.set_major_formatter(mons_)
            plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
            ax3.set_ylim(np.min(icdfM.values)-0.01, np.max(icdfM.values)+0.01)
            ax3.set_title(label='ICmean %.2f, ICIR %.1f, ICSR %.2f, tor %.2f'%(icmean, icir, ic_sr, avg_tur))

            ax3_twin = ax3.twinx()
            ax3_twin.plot(icdfM.index.values, icdfM.cumsum().values)
            ax3_twin.grid(visible=False)
            ax3_twin.set_ylim(np.min(icdfM.cumsum().values)-1, np.max(icdfM.cumsum().values)+1)

            # mpl_axes_aligner.align.yaxes(ax3, 0, ax3_twin, 0, 0.5)
    
    
    if if_savefig:
        myMakeDir(figpath)
        plt.savefig(os.path.join(figpath, '%s.png'%(nameflag)))
    if if_show_fig:
        plt.show()
    plt.close()
    return nameflag.split('-')[0], icvalue, deltanp, sr, factor_group

def FactorPerform(factor_df, y_df, plt_ycol='', bin_ycols=[], cuts=2, ifqcut=True, if_savefig=False, figpath='', nameflag_='', if_show_ls=True, if_show_fig=True, if_show_gp=False, ic_mode=1, plt_date=None):
    if isinstance(factor_df, pd.Series):
        name_, icvalue, deltanp, sr, factor_group = FactorPerform_TS(factor_df, y_df, plt_ycol, bin_ycols, cuts, ifqcut, if_savefig, figpath, nameflag_, if_show_ls)
    else:
        name_, icvalue, deltanp, sr, factor_group = FactorPerform_HZ(factor_df, y_df, plt_ycol, bin_ycols, cuts, ifqcut, if_savefig, figpath, nameflag_, if_show_ls, if_show_fig, if_show_gp, ic_mode, plt_date)
    return name_, icvalue, deltanp, sr, factor_group

def PlotPerformance(tmp_gdf, comms, figpath, nameflag_='', enddate=datetime.datetime(2010,1,1), base_val=100, if_show_fig=True):
    tmp_gdf = tmp_gdf[tmp_gdf.index>enddate]
    tmp_gdf[['GrossProfit','Profit']] = tmp_gdf[['GrossProfit','Profit']]-tmp_gdf[['GrossProfit','Profit']].iloc[0]
    try:
        ps_cols = list(['ps_%s' % x for x in comms])
        psdf = tmp_gdf[ps_cols]
        avgtor = CalculateTur(psdf)
    except:
        avgtor = 1
    tmp_gdf = tmp_gdf.resample('1D').last().dropna(how='all')
    sr = np.sqrt(252) * tmp_gdf['Profit'].diff().mean() / tmp_gdf['Profit'].diff().std()
    print(('sr: ', sr))

    tmp_gdf['index'] = tmp_gdf.index
    acc_np = tmp_gdf['Profit'].values
    acc_gp = tmp_gdf['GrossProfit'].values

    dd = acc_np - np.maximum.accumulate(acc_np)
    mdd = np.min(dd)
    rf = 0 if mdd == 0 else -acc_np[-1] / mdd
    
    ddr = 1.0*dd/(base_val+np.maximum.accumulate(acc_np))
    ddr = eu.nz(ddr)

    timestamps = tmp_gdf['index'].tolist()
    month_rate = pm.get_period_win_rate(timestamp=timestamps, net_accu_profits=acc_np, period='1M')
    week_rate = pm.get_period_win_rate(timestamp=timestamps, net_accu_profits=acc_np, period='1W')
    title = 'GP: %.2f NP: %.2f MDD: %.2f m_wr: %.2f, w_wr: %.2f, rf: %.2f, sr: %.2f, tor: %.2f' \
            % (
                acc_gp[-1], acc_np[-1], mdd, month_rate,
                week_rate, rf, sr, avgtor)

    f, ax = plt.subplots(2)
    elplot.plot_profits(ax=ax[0], accu_profits=acc_gp, net_accu_profits=acc_np, timestamps=timestamps,
                        draw_down=dd,
                        title=title)
    elplot.plot_profit_per_month(ax=ax[1], net_accu_profits=acc_np, timestamp=timestamps, fonsize=12)
    # elplot.plot_dd(ax[2], dd=ddr, timestamp=timestamps, ylabel='DD',fonsize=12)

    if nameflag_=='':
        title_name = 'factordf-(%.2f, %.2f)'%(sr, avgtor)
    else:
        title_name = '%s-(%.2f, %.2f)'%(nameflag_, sr, avgtor)
    myMakeDir(figpath)
    fig_fn = os.path.join(figpath, '%s.png' % (title_name))

    f.set_size_inches(22, 21)
    f.savefig(fig_fn)
    if if_show_fig:
        plt.show()
    else:
        plt.close()

    return sr, avgtor
    



def GetMinProfit(psdf, data, comms, cost_penalty=0, if_normTN=True, if_pct=True):
    C = data['C']
    
    ps_cols = map(lambda x: 'ps_%s' % (x), comms)


    pdf = 100 * (C[comms] / C[comms].shift(1)-1)
    pdf = pdf.replace([np.inf,-np.inf],np.nan).fillna(0)
    pdf.columns = comms

    gdf = pd.DataFrame(index=C.index)


    for x in comms:
        gdf['close_%s' % x] = C[x].fillna(method='ffill')
        gdf['ps_%s' % x] = psdf[x]
        
        cost, tick_size, weight = eu.get_unit_cost(x)
        cost = cost * tick_size / 2.0

        # print (x, cost)

        if if_pct:
            gdf['GP_%s' % (x)] = psdf[x].shift(1) * pdf[x]
        else:
            gdf['GP_%s' % (x)] = psdf[x].shift(1) * gdf['close_%s' % x].diff() * tick_size

        if if_pct:
            cost = cost * 1.0 / (tick_size * C[x])

        cost = cost*cost_penalty
        if if_pct:
            gdf['cost_%s' % (x)] = 100.0 * abs(psdf[x].diff().fillna(0)) * cost
        else:
            gdf['cost_%s' % (x)] = abs(psdf[x].diff().fillna(0)) * cost

        gdf['NP_%s' % x] = gdf['GP_%s' % (x)] - gdf['cost_%s' % (x)]

    gp_cols = list(map(lambda x: 'GP_%s' % (x), comms))
    ct_cols = list(map(lambda x: 'cost_%s' % (x), comms))
    np_cols = list(map(lambda x: 'NP_%s' % (x), comms))
    ps_cols = list(map(lambda x: 'ps_%s' % (x), comms))

    labeldf = C[comms]
    labeldf = 1 - 1 * pd.isnull(labeldf)
    labeldf = labeldf.replace(0, np.nan)
    labeldf = labeldf.fillna(method='ffill').fillna(0)

    gdf['TN'] = labeldf.sum(axis=1)
    gdf['TradeN'] = abs(gdf[ps_cols]).sum(axis=1)
    
    if if_normTN:
        gdf['GrossProfit'] = gdf[gp_cols].sum(axis=1) / gdf['TN'] 
        gdf['cost'] = gdf[ct_cols].sum(axis=1) / gdf['TN'] 
        gdf['Profit'] = gdf[np_cols].sum(axis=1) / gdf['TN']
    else:
        gdf['GrossProfit'] = gdf[gp_cols].sum(axis=1)
        gdf['cost'] = gdf[ct_cols].sum(axis=1)
        gdf['Profit'] = gdf[np_cols].sum(axis=1)

    gdf[gp_cols + np_cols + ct_cols] = gdf[gp_cols + np_cols + ct_cols].fillna(0).cumsum()

    gdf['GrossProfit'] = gdf['GrossProfit'].fillna(0).cumsum()
    gdf['cost'] = gdf['cost'].fillna(0).cumsum()
    gdf['Profit'] = gdf['Profit'].fillna(0).cumsum()

#     gdf = gdf.resample('1D', how='last').dropna(how='all')

    return gdf


'''
TS Analysis
'''

def getGroupBins(inData, cuts=5, ifqcut=True):
    Data = inData.dropna()

    per95 = Data.quantile(0.975, interpolation='nearest')
    per05 = Data.quantile(0.025, interpolation='nearest')
    Data.loc[Data > per95] = per95
    Data.loc[Data < per05] = per05
    eps = 0.001
    ibins = np.linspace(per05-eps, per95+eps, cuts+1, endpoint=True)
    ilabels = range(1, cuts+1, 1)
    
    try:
        tmpfactor = pd.qcut(Data, q=cuts, duplicates='drop')
    except:
        return pd.Series(index=inData.index)
    binsnum = len(tmpfactor.cat.categories)
    ilabels = ilabels[:binsnum]

    if ifqcut:
        factor,ibins = pd.qcut(Data, q=cuts, labels=ilabels, duplicates='drop', retbins=True)
    else:
        factor = pd.cut(Data, bins=ibins, labels=ilabels)

    factor = factor.astype(float)
    factor = factor.reindex(inData.index)

    return factor, ibins

def GetAllTGroupV2(comms, factordf, ydf, if_jy, cuts=50, ifqcut=True, startdate=None, enddate=None, weightdf=None):
    factordf[if_jy==0] = np.nan
    ixdf = factordf[comms].stack(dropna=False)
    ixdf = ixdf.reset_index()

    ixdf.columns = ['index', 'ticker', 'factor_raw']
    ixdf = ixdf.set_index('index')

    iydf = ydf.copy()
    iydf[if_jy==0] = np.nan

    iydf = iydf.reindex(factordf.index)
    iydf = iydf[comms].stack(dropna=False)
    iydf = iydf.reset_index()

    iydf.columns = ['index', 'ticker', 'y']
    ixdf['y'] = iydf['y'].values

    ixdf = ixdf.replace([np.inf, -np.inf], np.nan)
    ixdf = ixdf.dropna(how='any')
    ixdf0 = ixdf.copy()
    
    if startdate is not None:
        ixdf = ixdf[ixdf.index>=startdate]
    if enddate is not None:
        ixdf = ixdf[ixdf.index<enddate]

    # print ('qcut time range: ', ixdf.sort_index().index[0], ixdf.sort_index().index[-1])

    group, bins = getGroupBins(ixdf['factor_raw'], cuts=cuts, ifqcut=ifqcut)

    bins_len = bins[-1]-bins[0]
    bins[0] = bins[0]-20*bins_len
    bins[-1] = bins[-1]+20*bins_len

    ixdf0['factor_raw_clip'] = ixdf0['factor_raw'].clip(bins[0]+bins_len, bins[-1]-bins_len)

    # print ('bins: ', bins)

    ixdf0['group'] = ixdf0['factor_raw_clip'].apply(lambda x: bisect.bisect(bins, x))

    return ixdf0


def TSFactorAnalysis(factordf, ydf, if_jy, cuts, ifqcut, startdate, enddate, nameflag_='', scomms=[], plt_groups=[], if_plot_all=False, if_plot_ls=False, if_savefig=True, figpath=''):
    if len(scomms)==0:
        scomms = factordf.columns.tolist()

    factordf[if_jy==0] = np.nan

    dfall = GetAllTGroupV2(scomms, factordf=factordf, ydf=ydf, if_jy=if_jy, cuts=cuts, ifqcut=ifqcut, startdate=startdate, enddate=enddate, weightdf=None)
    lossdf = dfall.groupby('group').agg({'y':'mean', 'factor_raw':'mean', 'factor_raw_clip':'count'})
    
    groups = dfall['group'].dropna().unique().tolist()
    groups.sort()


        
    nfactor = dfall.reset_index().pivot(index='index',columns='ticker',values='group')
    nfactor = nfactor.reindex(factordf.index)

    # print (nfactor.tail())

    lkbperiod = 21*1
    tmpdf = pd.DataFrame()
    pptdf = pd.DataFrame()
    for group in groups:
        tmpdf['G%d'%int(group)] = ydf[nfactor==group].fillna(0).sum(axis=1).cumsum()

        sumdf = ydf[nfactor==group].fillna(0).rolling(window=lkbperiod,min_periods=1).sum().sum(axis=1)
        ntrdf = (~pd.isnull(ydf[nfactor==group])).rolling(window=lkbperiod,min_periods=1).sum().sum(axis=1)
        ipptdf = sumdf/ntrdf

        pptdf['G%d'%int(group)] = ipptdf

    # tmpdf = pd.DataFrame()
    # for group in groups:
    #     tmpdf['G%d'%int(group)] = ydf[nfactor==group].fillna(0).sum(axis=1).cumsum()
        
    tmpdf = tmpdf.fillna(method='ffill').fillna(0)

    tmpdf['LS'] = tmpdf['G%d'%int(groups[-1])] - tmpdf['G%d'%int(groups[0])]
    pptdf['LS'] = (pptdf['G%d'%int(groups[-1])] - pptdf['G%d'%int(groups[0])])/2.0
    tmpdf['All'] = ydf.fillna(0).sum(axis=1).cumsum()/cuts
    

    npsdf = nfactor/cuts
    npsdf = 2*((nfactor-1)/(cuts-1)-0.5)

    tor = CalculateTur(npsdf.fillna(0))

    if len(plt_groups)==0:
        plt_groups = ['G%d'%int(groups[0]), 'G%d'%int(groups[-1])]

    if if_plot_all:
        print ('all length: ', len(dfall), tmpdf['All'].iloc[-1]*cuts)
        plt_groups.append('All')

    if if_plot_ls:
        plt_groups.append('LS')

    if nameflag_=='':
        title_name = 'TT'
    else:
        title_name = '%s'%(nameflag_)

    sr = np.sqrt(252)*tmpdf[plt_groups].diff().mean()/tmpdf[plt_groups].diff().std()
    title_list = list(map(lambda x: '%s %.1f'%(x, sr[x]), sr.index))

    gsr = sr.values.tolist()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 24))
    lossdf['y'].plot(ax=ax1, kind='bar', title=title_name)
    pptdf['LS'].resample('1M').last().plot(ax=ax2, kind='bar', rot=15)

    ax2.xaxis.set_major_locator(mdates.WeekdayLocator())

    tmpdf[plt_groups].plot(ax=ax3, title='SR: '+', '.join(title_list)+' TOR: %.2f'%tor)
    if startdate is not None:
        ax3.axvline(startdate, color="green", linestyle="--")
    if enddate is not None:
        ax3.axvline(enddate, color="red", linestyle="--")
    
    if if_savefig:
        myMakeDir(figpath)
        
        if nameflag_=='':
            title_name = 'TT-(%.2f, %.2f)'%(gsr[-1], tor)
        else:
            title_name = '%s-(%.2f, %.2f)'%(nameflag_, gsr[-1], tor)

        fig.savefig(os.path.join(figpath, '%s.png'%(title_name)))

        plt.close()
    else:
        plt.show()

    return gsr[-1], tor, npsdf, nfactor


def FactorPerform_TS(factordf, ydf, if_jy, cuts, ifqcut, startdate, enddate, nameflag_='', scomms=[], plt_groups=[], if_plot_all=False, if_plot_ls=False, if_savefig=True, figpath=''):
    if len(scomms)==0:
        scomms = factordf.columns.tolist()

    factordf[if_jy==0] = np.nan

    dfall = GetAllTGroupV2(scomms, factordf=factordf, ydf=ydf, if_jy=if_jy, cuts=cuts, ifqcut=ifqcut, startdate=startdate, enddate=enddate, weightdf=None)

    dfall['month'] = list(map(lambda x: x.month, dfall.index))
    dfall['month_flag'] = dfall['month']!=dfall['month'].shift(1)
    dfall['month_num'] = dfall['month_flag'].cumsum()

    dfall['index'] = dfall.index
    icdfM = dfall.groupby(dfall['month_num']).apply(lambda x: x['factor_raw'].corr(x['y']))
    icdfM.index = dfall['index'].groupby(dfall['month_num']).last()

    icmean = 1 * icdfM.mean()
    icir = icmean / icdfM.std()
    ic_sr = np.sqrt(12)*icdfM.mean()/icdfM.std()
    

    lossdf = dfall.groupby('group').agg({'y':'mean', 'factor_raw':'mean', 'factor_raw_clip':'count'})
    
    groups = dfall['group'].dropna().unique().tolist()
    groups.sort()

    nfactor = dfall.reset_index(drop=True).pivot(index='index',columns='ticker',values='group')
    nfactor = nfactor.reindex(factordf.index)

    # print (nfactor.tail())

    tmpdf = pd.DataFrame()
    for group in groups:
        tmpdf['y_%d'%int(group)] = ydf[nfactor==group].fillna(0).sum(axis=1).cumsum()

    tmpdf = tmpdf.fillna(method='ffill').fillna(0)
    tmpdf['LongShort'] = tmpdf['y_%d'%int(groups[-1])] - tmpdf['y_%d'%int(groups[0])]
    tmpdf['All'] = ydf.fillna(0).sum(axis=1).cumsum()/cuts


    npsdf = nfactor/cuts
    npsdf = 2*((nfactor-1)/(cuts-1)-0.5)

    tor = CalculateTur(npsdf.fillna(0))

    if len(plt_groups)==0:
        # plt_groups = ['G%d'%int(groups[0]), 'G%d'%int(groups[-1])]
        plt_groups = list(map(lambda x:'y_%d'%x, groups))

    if if_plot_all:
        print ('all length: ', len(dfall), tmpdf['All'].iloc[-1]*cuts)
        plt_groups.append('All')

    if if_plot_ls:
        plt_groups.append('LongShort')

    if nameflag_=='':
        title_name = 'TT'
    else:
        title_name = '%s'%(nameflag_)

    tmpdf_day = tmpdf[['LongShort']].resample('1D').last().dropna(how='all')

    sr = np.sqrt(252)*tmpdf_day['LongShort'].diff().mean()/tmpdf_day['LongShort'].diff().std()

    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 24))
    lossdf['y'].plot(ax=ax1, kind='bar', title=title_name)
    tmpdf[plt_groups].plot(ax=ax2, title='SR: %.2f'%sr)


    fonsize = 24
    ax3.bar(icdfM.index.values, icdfM.values, width=fonsize/2, color='#1f77b4')

    mons = mdates.MonthLocator(interval=3)
    mons_ = mdates.DateFormatter('%Y-%m')
    ax3.xaxis.set_major_locator(mons)
    ax3.xaxis.set_major_formatter(mons_)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    ax3.set_ylim(np.min(icdfM.values)-0.01, np.max(icdfM.values)+0.01)
    ax3.set_title(label='ICmean %.2f, ICIR %.2f, ICSR %.2f, tor %.2f'%(icmean, icir, ic_sr, tor))

    ax3_twin = ax3.twinx()
    ax3_twin.plot(icdfM.index.values, icdfM.cumsum().values)
    ax3_twin.grid(visible=False)
    ax3_twin.set_ylim(np.min(icdfM.cumsum().values)-1, np.max(icdfM.cumsum().values)+1)

    if startdate is not None:
        ax3.axvline(startdate, color="green", linestyle="--")
    if enddate is not None:
        ax3.axvline(enddate, color="red", linestyle="--")

    if if_savefig:
        myMakeDir(figpath)
        
        if nameflag_=='':
            nameflag_ - 'TT'
        title_name = '%s-(%.1f, %.2f, %.2f)'%(nameflag_, sr, tor, icmean)

        fig.savefig(os.path.join(figpath, '%s.png'%(title_name)))

        plt.close()
    else:
        plt.show()

    return sr, tor, npsdf, nfactor



def GetAllTGroupV3(comms, factordf, ydf_dict, if_jy, cuts=50, ifqcut=True, startdate=None, enddate=None, weightdf=None):
    factordf[if_jy==0] = np.nan
    ixdf = factordf[comms].stack(dropna=False)
    ixdf = ixdf.reset_index()

    ixdf.columns = ['index', 'ticker', 'factor_raw']
    ixdf = ixdf.set_index('index')

    for iykey in list(ydf_dict.keys()):
        iydf = ydf_dict[iykey].copy()
        iydf[if_jy==0] = np.nan

        iydf = iydf.reindex(factordf.index)
        iydf = iydf[comms].stack(dropna=False)
        iydf = iydf.reset_index()

        iydf.columns = ['index', 'ticker', 'y']
        ixdf[iykey] = iydf['y'].values

    ixdf = ixdf.replace([np.inf, -np.inf], np.nan)
    ixdf = ixdf.dropna(how='any')
    ixdf0 = ixdf.copy()
    
    if startdate is not None:
        ixdf = ixdf[ixdf.index>=startdate]
    if enddate is not None:
        ixdf = ixdf[ixdf.index<enddate]

    # print ('qcut time range: ', ixdf.sort_index().index[0], ixdf.sort_index().index[-1])

    group, bins = getGroupBins(ixdf['factor_raw'], cuts=cuts, ifqcut=ifqcut)

    bins_len = bins[-1]-bins[0]
    bins[0] = bins[0]-20*bins_len
    bins[-1] = bins[-1]+20*bins_len

    ixdf0['factor_raw_clip'] = ixdf0['factor_raw'].clip(bins[0]+bins_len, bins[-1]-bins_len)

    # print ('bins: ', bins)

    ixdf0['group'] = ixdf0['factor_raw_clip'].apply(lambda x: bisect.bisect(bins, x))

    return ixdf0

def FactorPerform_TSS(factordf, ydf_dict, y_show, if_jy, cuts, ifqcut, startdate, enddate, nameflag_='', scomms=[], plt_groups=[], if_plot_all=False, if_plot_ls=False, if_savefig=True, figpath=''):
    if len(scomms)==0:
        scomms = factordf.columns.tolist()

    factordf[if_jy==0] = np.nan

    dfall = GetAllTGroupV3(scomms, factordf=factordf, ydf_dict=ydf_dict, if_jy=if_jy, cuts=cuts, ifqcut=ifqcut, startdate=startdate, enddate=enddate, weightdf=None)

    ylabels = list(ydf_dict.keys())

    dfall['month'] = list(map(lambda x: x.month, dfall.index))
    dfall['month_flag'] = dfall['month']!=dfall['month'].shift(1)
    dfall['month_num'] = dfall['month_flag'].cumsum()

    dfall['index'] = dfall.index
    icdfM = dfall.groupby(dfall['month_num']).apply(lambda x: x['factor_raw'].corr(x[y_show]))
    icdfM.index = dfall['index'].groupby(dfall['month_num']).last()

    dflist = list()
    iclist = list()

    for ylabel in ylabels:
        idf = dfall.groupby(dfall['month_num']).apply(lambda x: x['factor_raw'].corr(x[ylabel]))
        idf.name = ylabel
        dflist.append(idf)
        iclist.append(idf.mean())

    icdfMdf = pd.concat(dflist, axis=1)
    icdfMdf.index = icdfM.index

    icmeandf = pd.DataFrame(iclist, index=ylabels)
    icmeandf.columns = ['ICmean']

    icmean = 1 * icdfM.mean()
    icir = icmean / icdfM.std()
    ic_sr = np.sqrt(12)*icdfM.mean()/icdfM.std()
    
    lossdf = dfall.groupby('group').mean()
    
    groups = dfall['group'].dropna().unique().tolist()
    groups.sort()

    nfactor = dfall.reset_index(drop=True).pivot(index='index',columns='ticker',values='group')
    nfactor = nfactor.reindex(factordf.index)

    # print (nfactor.tail())

    ydf = ydf_dict[y_show]

    tmpdf = pd.DataFrame()
    for group in groups:
        tmpdf['y_%d'%int(group)] = ydf[nfactor==group].fillna(0).sum(axis=1).cumsum()

    tmpdf = tmpdf.fillna(method='ffill').fillna(0)
    tmpdf['LongShort'] = tmpdf['y_%d'%int(groups[-1])] - tmpdf['y_%d'%int(groups[0])]
    tmpdf['All'] = ydf.fillna(0).sum(axis=1).cumsum()/cuts


    npsdf = nfactor/cuts
    npsdf = 2*((nfactor-1)/(cuts-1)-0.5)

    tor = CalculateTur(npsdf.fillna(0))

    if len(plt_groups)==0:
        # plt_groups = ['G%d'%int(groups[0]), 'G%d'%int(groups[-1])]
        plt_groups = list(map(lambda x:'y_%d'%x, groups))

    if if_plot_all:
        print ('all length: ', len(dfall), tmpdf['All'].iloc[-1]*cuts)
        plt_groups.append('All')

    if if_plot_ls:
        plt_groups.append('LongShort')

    if nameflag_=='':
        title_name = 'TT'
    else:
        title_name = '%s'%(nameflag_)

    tmpdf_day = tmpdf[['LongShort']].resample('1D').last().dropna(how='all')

    sr = np.sqrt(252)*tmpdf_day['LongShort'].diff().mean()/tmpdf_day['LongShort'].diff().std()

    
    fig = plt.figure(figsize=(16, 32))
    ax11 = fig.add_subplot(4,1,1)
    lossdf[ylabels].plot(ax=ax11, kind='bar', title=title_name)
    
    ax21 = fig.add_subplot(4,2,3)
    icmeandf.plot(ax=ax21, title='ICMmean')
    ax22 = fig.add_subplot(4,2,4)
    icdfMdf.cumsum().plot(ax=ax22, title='ICMcum')

    ax2 = fig.add_subplot(4,1,3)
    tmpdf[plt_groups].plot(ax=ax2, title='[%s] SR: %.2f'%(y_show, sr))

    ax3 = fig.add_subplot(4,1,4)
    fonsize = 24
    ax3.bar(icdfM.index.values, icdfM.values, width=fonsize/2, color='#1f77b4')

    mons = mdates.MonthLocator(interval=3)
    mons_ = mdates.DateFormatter('%Y-%m')
    ax3.xaxis.set_major_locator(mons)
    ax3.xaxis.set_major_formatter(mons_)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    ax3.set_ylim(np.min(icdfM.values)-0.01, np.max(icdfM.values)+0.01)
    ax3.set_title(label='[%s] ICmean %.2f, ICIR %.2f, ICSR %.2f, tor %.2f'%(y_show, icmean, icir, ic_sr, tor))

    ax3_twin = ax3.twinx()
    ax3_twin.plot(icdfM.index.values, icdfM.cumsum().values)
    ax3_twin.grid(visible=False)
    ax3_twin.set_ylim(np.min(icdfM.cumsum().values)-1, np.max(icdfM.cumsum().values)+1)

    if if_savefig:
        myMakeDir(figpath)
        
        if nameflag_=='':
            nameflag_ - 'TT'
        title_name = '%s-(%.1f, %.2f, %.2f)'%(nameflag_, sr, tor, icmean)

        fig.savefig(os.path.join(figpath, '%s.png'%(title_name)))

        plt.close()
    else:
        plt.show()

    return sr, tor, npsdf, nfactor

def FactorPerform_HZS(factor_df, ydf_dict, y_show, cuts=5, if_savefig=False, figpath='', nameflag_='', if_show_ls=True, if_show_fig=True, if_show_gp=False, ic_mode=1, plt_date=None):
    # print('calculate IC')
    ylabels = list(ydf_dict.keys())

    dflist = list()
    iclist = list()

    for ylabel in ylabels:
        y_df = ydf_dict[ylabel].copy()

        if plt_date is not None:
            factor_df = factor_df[factor_df.index>plt_date]
            y_df = y_df[y_df.index>plt_date]

        factor_df = factor_df.dropna(how='all')
        y_df = y_df.reindex(factor_df.index)
        
        icdf = factor_df.corrwith(y_df, axis=1)
        icdf = icdf.replace([np.inf, -np.inf], np.nan).fillna(0)

        icdfM = icdf.resample('1M').mean()
        icdfM.name = ylabel

        dflist.append(icdfM)
        iclist.append(icdfM.mean())

    icdfMdf = pd.concat(dflist, axis=1)
    icdfMdf.index = icdfM.index

    icmeandf = pd.DataFrame(iclist, index=ylabels)
    icmeandf.columns = ['ICmean']

    y_df = ydf_dict[y_show].copy()

    if plt_date is not None:
        factor_df = factor_df[factor_df.index>plt_date]
        y_df = y_df[y_df.index>plt_date]

    factor_df = factor_df.dropna(how='all')
    y_df = y_df.reindex(factor_df.index)
    
    icdf = factor_df.corrwith(y_df, axis=1)
    icdf = icdf.replace([np.inf, -np.inf], np.nan).fillna(0)

    icdfM = icdf.resample('1M').mean()

    icmean = 1 * icdfM.mean()
    icir = icmean / icdfM.std()
    
    icdf = icdf.cumsum()

    icdf_day = icdf.resample('1D').last().dropna(how='all')
    
    ic_sr = np.sqrt(252)*icdf_day.diff().mean()/icdf_day.diff().std()
    
    TN = (~pd.isnull(factor_df)).sum(axis=1)
    TNbias = 1.0/TN
    
    # print('factor_group')
    eps = 0.001
    factor_rank = factor_df.rank(pct=True, axis=1)
    factor_rank = factor_rank.sub(TNbias/2.0,axis=0)
    factor_rank = factor_rank.fillna(0.5)
    factor_rank[factor_rank>1-eps] = 1-eps
    factor_rank[factor_rank<eps] = eps
    factor_group = (factor_rank*cuts).apply(np.floor)+1
    factor_group = factor_group.astype(int)
    factor_group[factor_df.apply(np.isnan)==True]=np.nan
    factor_group = factor_group.dropna(how='all')
    
    psdf = 2 * ((factor_group-1)*1.0/(cuts-1) - 0.5)

    
    avg_tur = CalculateTur(psdf)


    rankdf = factor_df.rank(pct=True, axis=1)
    totaldf = rankdf.sum(axis=1)

    psdf0 = rankdf.div(totaldf, axis=0)
    avg_psdf0 = psdf0.mean(axis=1)

    psdf1 = psdf0.sub(avg_psdf0, axis=0)
    
    avg_tur1 = CalculateTur(psdf1)

    gpdf = pd.DataFrame()
    gpdf['gp'] = (psdf1.fillna(0)*y_df.fillna(0)).sum(axis=1).cumsum()
    
    sr1 = np.sqrt(252)*gpdf['gp'].diff().mean()/gpdf['gp'].diff().std()

    
    # print('plot')
    
    y_group = pd.DataFrame(index=list(range(1,cuts+1,1)))
    yg_ts = pd.DataFrame(index=factor_df.index)
    
    for ylabel in ylabels:
        y_group[ylabel] = np.nan
        y_df = ydf_dict[ylabel].copy()
        y_df = y_df.reindex(factor_group.index)
        for i in range(1,cuts+1,1):
            y_group.loc[i, ylabel] = y_df[factor_group==i].unstack().dropna().mean()
            if ylabel==y_show:
                yg_ts['y_%d'%i] = y_df[factor_group==i].mean(axis=1)
    
    yg_ts = yg_ts.fillna(0).cumsum()
    plt_cols = yg_ts.columns.tolist()
    
    yg_ts['LongShort'] = yg_ts['y_%d'%(cuts)]-yg_ts['y_%d'%(1)]
    yg_ts_day = yg_ts.resample('1D').last().dropna(how='all')


    if if_show_ls:
        plt_cols.append('LongShort')
    
    if if_show_ls:
        sr = np.sqrt(252)*yg_ts_day['LongShort'].diff().mean()/yg_ts_day['LongShort'].diff().std()
    else:
        sr = ic_sr
    

    
    fig = plt.figure(figsize=(16, 32))
    if nameflag_=='':
        nameflag = '%s-(%.1f, %.2f, %.2f)'%('TT',sr, avg_tur, icmean)
    else:
        nameflag = '%s-(%.1f, %.2f, %.2f)'%(nameflag_, sr, avg_tur, icmean)

    fig = plt.figure(figsize=(16, 32))
    ax1 = fig.add_subplot(4,1,1)
    y_group.plot(ax=ax1, kind='bar', grid=True, title=nameflag)

    ax21 = fig.add_subplot(4,2,3)
    icmeandf.plot(ax=ax21, title='ICMmean')
    ax22 = fig.add_subplot(4,2,4)
    icdfMdf.cumsum().plot(ax=ax22, title='ICMcum')

    ax2 = fig.add_subplot(4,1,3)
    yg_ts[plt_cols].plot(ax=ax2, grid=True, title='[%s] sr %.2f'%(y_show, sr))
    ax3 = fig.add_subplot(4,1,4)
    if if_show_gp:
        gpdf['gp'].plot(ax=ax3, grid=True, title='[%s] SR %.2f, tor %.2f'%(y_show, sr1, avg_tur1))
    else:
        if ic_mode==0:
            icdf.plot(ax=ax3, grid=True, title='[%s] ICmean %.3f, ICIR %.3f, ICSR %.2f, tor %.2f'%(y_show, icmean, icir, ic_sr, avg_tur))
        elif ic_mode==1:
            fonsize = 24
            ax3.bar(icdfM.index.values, icdfM.values, width=fonsize/2, color='#1f77b4')

            mons = mdates.MonthLocator(interval=3)
            mons_ = mdates.DateFormatter('%Y-%m')
            ax3.xaxis.set_major_locator(mons)
            ax3.xaxis.set_major_formatter(mons_)
            plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
            ylim_dn = np.min(icdfM.values)
            ylim_up = np.max(icdfM.values)
            ax3.set_ylim(ylim_dn-0.01, ylim_up+0.01)
            ax3.set_title(label='[%s] ICmean %.2f, ICIR %.1f, ICSR %.2f, tor %.2f'%(y_show, icmean, icir, ic_sr, avg_tur))

            ax3_twin = ax3.twinx()
            ax3_twin.plot(icdfM.index.values, icdfM.cumsum().values)
            ax3_twin.grid(visible=False)
            ax3_twin.set_ylim(np.min(icdfM.cumsum().values)-1, np.max(icdfM.cumsum().values)+1)

            # mpl_axes_aligner.align.yaxes(ax3, 0, ax3_twin, 0, 0.5)
    
    
    if if_savefig:
        myMakeDir(figpath)
        plt.savefig(os.path.join(figpath, '%s.png'%(nameflag)))
    if if_show_fig:
        plt.show()
    plt.close()
    return nameflag.split('-')[0], sr, icmean, icir, ic_sr, avg_tur, factor_group