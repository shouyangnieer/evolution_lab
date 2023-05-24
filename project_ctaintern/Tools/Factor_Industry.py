# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pyelf.elutil as eu

# PSY 
# 心理线指标 PSY n日内上涨的天数/n*100。本因子的计算窗口为12日。
def PSY(data, xp):
    cdf = data['C']
    odf = data['O']
    change = cdf-odf>0
    PSYdf = change.rolling(window=xp,min_periods=1).sum()/xp
    nameflag_ = 'PSY%d'%xp
    return PSYdf, nameflag_

#  TVMA6 
# 6日成交金额的移动平均值 TVMA6 6日成交金额的移动平均值
def TVMA(data, xp):
    adf = data['amount']
    TVMAdf = adf.rolling(window=xp,min_periods=1).mean()
    nameflag_ = 'TVMA%d'%xp
    return TVMAdf, nameflag_

#  VR 
# 成交量比率（Volume Ratio） VR VR=N日内上涨日成交额总和/N日内下降日成交额总和
def VR(data, xp):
    cdf = data['C']
    odf = data['O']
    adf = data['amount']
    change1 = cdf-odf>0
    AVS = change1*adf
    change2 = cdf-odf<0
    BVS = change2*adf
    VRdf = AVS.rolling(window=xp,min_periods=1).sum()/BVS.rolling(window=xp,min_periods=1).sum()
    nameflag_ = 'VR%d'%xp
    return VRdf, nameflag_

#  BR 
# 意愿指标 BR BR=N日内（当日最高价－昨日收盘价）之和 / N日内（昨日收盘价－当日最低价）之和×100 n设定为26 
def BR(data, xp):
    cdf = data['C']
    hdf = data['H']
    ldf = data['L']
    br1 = hdf - cdf.shift(1)
    br2 = cdf.shift(1) - ldf
    BRdf = br1.rolling(window=xp,min_periods=1).sum()/br2.rolling(window=xp,min_periods=1).sum()
    nameflag_ = 'BR%d'%xp
    return BRdf, nameflag_

#  WVAD 
# 威廉变异离散量 WVAD (收盘价－开盘价)/(最高价－最低价)×成交量，再做加和，使用过去6个交易日的数据
def WVAD(data,x):
    cdf = data['C']
    odf = data['O']
    hdf = data['H']
    ldf = data['L']
    vdf = data['V']
    wva = (cdf-odf)/(hdf-ldf)*vdf
    WVADdf = wva.rolling(window=x, min_periods=1).sum()
    nameflag_ = 'WVAD%d' % x
    return WVADdf, nameflag_

#  AR 
# 人气指标 AR AR=N日内（当日最高价—当日开市价）之和 / N日内（当日开市价—当日最低价）之和 * 100，n设定为26
def AR(data, xp):
    odf = data['O']
    hdf = data['H']
    ldf = data['L']
    ar1 = hdf - odf
    ar2 = odf - ldf
    ARdf = ar1.rolling(window=xp,min_periods=1).sum()/ar2.rolling(window=xp,min_periods=1).sum()
    nameflag_ = 'AR%d'%xp
    return ARdf, nameflag_

#  ARBR 
# ARBR 因子 AR 与因子 BR 的差
def ARBR(data, xp):
    cdf = data['C']
    odf = data['O']
    hdf = data['H']
    ldf = data['L']
    ar1 = hdf - odf
    ar2 = odf - ldf
    br1 = hdf - cdf.shift(1)
    br2 = cdf.shift(1) - ldf
    ARdf = ar1.rolling(window=xp,min_periods=1).sum()/ar2.rolling(window=xp,min_periods=1).sum()
    BRdf = br1.rolling(window=xp,min_periods=1).sum()/br2.rolling(window=xp,min_periods=1).sum()
    ARBRdf = ARdf - BRdf
    nameflag_ = 'ARBR%d'%xp
    return ARBRdf, nameflag_

#  SKEW 
# 个股收益的偏度 Skewness 取交易日的收盘价数据，计算日收益率，再计算其偏度
def SKEW(data, xp):
    cdf = data['C']
    ret = cdf.pct_change()  
    SKEWdf = ret.rolling(window=xp, min_periods=1).skew()
    nameflag_ = 'SKEW%d'%xp
    return SKEWdf, nameflag_

#  SharpRatio 
# 夏普比率 sharpe_ratio （Rp - Rf） / Sigma p 其中，Rp是个股的年化收益率，Rf是无风险利率（在这里设置为0.04），Sigma p是个股的收益波动率（标准差）
def SignSR(data, xp):
    cdf = data['C']
    ret1 = cdf.pct_change()
    srdf = ret1.rolling(window=xp,min_periods=1).mean()/ret1.rolling(window=xp,min_periods=1).std()
    nameflag_ = 'SignSR%d'%xp
    return srdf, nameflag_

def SignEMASR(data, xp):
    cdf = data['C']
    ret1 = cdf.pct_change()
    srdf = ret1.ewm(span=xp, min_periods=xp).mean()/ret1.ewm(span=xp, min_periods=xp).std()
    nameflag_ = 'SignEMASR%d'%xp
    return srdf, nameflag_

def SignTR(data, xp):
    cdf = data['C']
    trdf = np.log(cdf/cdf.shift(xp))/abs(np.log(cdf/cdf.shift(1))).rolling(window=xp).sum()
    nameflag_ = 'SignTR%d'%xp
    return trdf, nameflag_

def SignTRR(data, xp):
    cdf = data['C']
    trdf = np.log(cdf/cdf.shift(xp))/abs(np.log(cdf/cdf.shift(1))).rolling(window=xp).sum()*abs(np.log(cdf/cdf.shift(xp)))
    nameflag_ = 'SignTRR%d'%xp
    return trdf, nameflag_

#  BollUp 

def bu(data,x):
    cdf = data['C']
    bu = (cdf.rolling(window=x, min_periods=1).mean()+2*cdf.rolling(window=x, min_periods=1).std())/cdf
    nameflag_ = 'bu%d' % x
    return bu, nameflag_

#  BollDown 
def bd(data,x):
    cdf = data['C']
    bd = (cdf.rolling(window=x, min_periods=1).mean()-2*cdf.rolling(window=x, min_periods=1).std())/cdf
    nameflag_ = 'bd%d' % x
    return bd, nameflag_

#  BIAS5 
# 5日乖离率 BIAS5 （收盘价-收盘价的N日简单平均）/ 收盘价的N日简单平均*100，在此n取5
def bia(data,x):
    cdf = data['C']
    bia = (cdf - cdf.rolling(window=x, min_periods=1).mean())/cdf.rolling(window=x, min_periods=1).mean()
    nameflag_ = 'bia%d' % x
    return bia, nameflag_

#  CCI 
def cci(data,x):
    cdf = data['C']
    hdf = data['H']
    ldf = data['L']
    typ = (cdf+hdf+ldf)/3
    cci = (typ-typ.rolling(window=x, min_periods=1).mean())/(0.015 * typ.rolling(window=x, min_periods=1).std())

    nameflag_ = 'ldx%d' % x
    return cci, nameflag_

#  超买 

def overbought(data, xp):
    cdf = data['C']
    rolling_mean = cdf.rolling(window=xp).mean()
    rolling_std = cdf.rolling(window=xp).std()
    upper_band = rolling_mean + 2 * rolling_std
    overbought = cdf > upper_band
    overboughtdf = overbought.rolling(window=xp).mean()

    nameflag_ = 'overbought%d'%xp
    return overboughtdf, nameflag_

#  超卖 

def oversold(data, xp):
    cdf = data['C']
    rolling_mean = cdf.rolling(window=xp).mean()
    rolling_std = cdf.rolling(window=xp).std()
    lower_band = rolling_mean - 2 * rolling_std
    oversold = cdf < lower_band
    oversolddf = oversold.rolling(window=xp).mean()

    nameflag_ = 'oversold%d'%xp
    return oversolddf, nameflag_

# （超买-超卖） 

def over(data, xp):
    cdf = data['C']
    rolling_mean = cdf.rolling(window=xp).mean()
    rolling_std = cdf.rolling(window=xp).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    over = 1 * (cdf > upper_band) - 1 * (cdf < lower_band)
    overdf = over.rolling(window=xp).mean()

    nameflag_ = 'over%d'%xp
    return overdf, nameflag_

#  大小k线背离 

def kline_divergence(data, xp):
    cdf = data['C']
    ldf = data['L']
    hdf = data['H']

    kline = (ldf.rolling(window=xp).min() + hdf.rolling(window=xp).max()) / 2
    rolling_mean = kline.rolling(window=xp).mean()
    rolling_std = kline.rolling(window=xp).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    divergence = 1*(cdf > upper_band) - 1*(cdf < lower_band)
    kline_divergencedf = divergence.rolling(window=xp).mean()

    nameflag_ = 'kline_divergence%d'%xp
    return kline_divergencedf, nameflag_

#  量价背离？
def volume_price_divergence(data, xp):
    cdf = data['C']
    vdf = data['V']

    vwap = (cdf * vdf).rolling(window=xp).sum() / vdf.rolling(window=xp).sum()
    rolling_mean = vwap.rolling(window=xp).mean()
    rolling_std = vwap.rolling(window=xp).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    divergence =  1*(cdf > upper_band) - 1*(cdf < lower_band)
    volume_price_divergencedf = divergence.rolling(window=xp).mean()

    nameflag_ = 'volume_price_divergence%d'%xp
    return volume_price_divergencedf, nameflag_

def stdskew(data, xp):
    cdf = data['C'].copy()
    ret = np.log(cdf) - np.log(cdf.shift(1))
    stdskewdf =  (ret.rolling(window=xp, min_periods=1).std() * np.sign(ret)).rolling(window=xp, min_periods=1).skew()
    nameflag_ = 'stdskew%d'%xp
    return stdskewdf, nameflag_

#  随机指标 hyh
# 来自cursor，high的最大值和low的最低值（和其他指标相关性较低）
# 补充：前一个量价背离因子与其他因子的相关性极低
def low_high(data,xp):
    ldf = data['L']
    hdf = data['H']
    cdf = data['C']
    
    lowest_low = ldf.rolling(window=xp).min()
    highest_high = hdf.rolling(window=xp).max()
    k_percent = 100 * ((cdf - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=3).mean()
    low_highdf = d_percent
    nameflag_ = 'low_high%d'%xp
    return low_highdf, nameflag_

# RSI 
# RSI = 100 - 100 / (1 + RS)，其中RS = n天内收盘价上涨幅度 / n天内收盘价下跌幅度


def rsi_hyh(data,xp):
    cdf = data['C']

    delta = cdf.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=xp).mean()
    avg_loss = loss.rolling(window=xp).mean().abs()

    rs = avg_gain / avg_loss
    RSIdf = 100 - (100 / (1 + rs))
    nameflag_ = 'RSI%d'%xp
    return RSIdf, nameflag_

#  动态窗口·伪 hyh
# 距离上一次上穿均线的距离-下穿距离。相关性挺低的
def ema_cross_distance(data,xp):
    cdf = data['C']
    ma_series = cdf.ewm(span=xp, min_periods=xp).mean()
    cross_above1 = (cdf > ma_series) & (cdf.shift(1) <= ma_series.shift(1))
    distance1 = cross_above1.apply(lambda x: x.groupby((x != x.shift()).cumsum()).rank(method='first')) #距离上一次上穿均线的距离
    cross_above2 = (cdf < ma_series) & (cdf.shift(1) >= ma_series.shift(1))
    distance2 = cross_above2.apply(lambda x: x.groupby((x != x.shift()).cumsum()).rank(method='first')) #距离上一次下穿均线的距离
    dynamic_windowdf = distance2 - distance1

    nameflag_ = 'ema_cross_distance%d'%xp
    return dynamic_windowdf, nameflag_

def ema_cross_meandiv(data,xp):
    cdf = data['C']
    ma_series = cdf.ewm(span=xp, min_periods=xp).mean()
    cross_above = ((cdf < ma_series) & (cdf.shift(1) >= ma_series.shift(1))) | ((cdf > ma_series) & (cdf.shift(1) <= ma_series.shift(1)))
    distance = cross_above.apply(lambda x: x.groupby((x != x.shift()).cumsum()).rank(method='first'))

    roc_df = pd.DataFrame(index=data['C'].index)
    for x in data['C'].columns.tolist():
        cc = data['C'][x].values
        pp = distance[x].astype(int).values
        aroc = cc/eu.rollingmean(cc, pp)-1
        roc_df[x] = aroc

    dynamic_windowsdf = roc_df

    nameflag_ = 'ema_cross_meandiv%d'%xp
    return dynamic_windowsdf, nameflag_

def ema_cross_roc(data,xp):
    cdf = data['C']
    ma_series = cdf.ewm(span=xp, min_periods=xp).mean()
    cross_above = ((cdf < ma_series) & (cdf.shift(1) >= ma_series.shift(1))) | ((cdf > ma_series) & (cdf.shift(1) <= ma_series.shift(1)))
    distance = cross_above.apply(lambda x: x.groupby((x != x.shift()).cumsum()).rank(method='first'))

    roc_df = pd.DataFrame(index=data['C'].index)
    for x in data['C'].columns.tolist():
        cc = data['C'][x].values
        pp = distance[x].astype(int).values
        aroc = cc/eu.ref(cc, -pp)-1
        roc_df[x] = aroc

    dynamic_windowsdf = roc_df

    nameflag_ = 'ema_cross_roc%d'%xp
    return dynamic_windowsdf, nameflag_

def meandiv(data,xp):
    cdf = data['C']
    static_windowsdf = cdf/cdf.rolling(window=xp, min_periods=1).mean()-1
    nameflag_ = 'meandiv%d'%xp
    return static_windowsdf, nameflag_

#  双均线 
def ema_cross(data, xp, xp2):
    cdf = data['C']
    ema1 = cdf.ewm(span=xp, min_periods=xp).mean()
    ema2 = cdf.ewm(span=xp2, min_periods=xp2).mean()
    ema_crossdf = (ema1 - ema2) / cdf
    nameflag_ = 'ema_cross(%d,%d)'%(xp, xp2)
    return ema_crossdf, nameflag_


def gap(data,xp):
    cdf = data['C']
    odf = data['O']
    hdf = data['H']
    ldf = data['L']
    
    gap_up = (odf > hdf.shift(1)) 
    gap_down = (ldf.shift(1) > odf) 
    gap_both = (odf > hdf.shift(1)) | (ldf.shift(1) > odf) 
    gap_updf = gap_up.apply(lambda x: x.groupby((x != x.shift()).cumsum()).rank(method='first'))
    gap_downdf = gap_down.apply(lambda x: x.groupby((x != x.shift()).cumsum()).rank(method='first'))

    proc_df = pd.DataFrame(index=data['C'].index)
    nroc_df = pd.DataFrame(index=data['C'].index)   
    for x in data['C'].columns.tolist():
        cc = data['C'][x].values
        pp = gap_updf[x].astype(int).values
        nn = gap_downdf[x].astype(int).values
        proc = cc/eu.ref(cc, -pp)-1 
        nroc = cc/eu.ref(cc, -nn)-1
        proc_df[x] = proc
        nroc_df[x] = nroc

    gapfinaldf = 2 * proc_df + nroc_df

    nameflag_ = 'gap%d'%xp
    return gapfinaldf, nameflag_



if __name__ == '__main__':
    print ('hello')