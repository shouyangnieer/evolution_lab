import numpy as np
import pickle
import os
from multiprocessing import Process, Manager
from scipy.stats import norm
import datetime
import talib as ta
import copy
from numba import jit
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
import traceback
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


def get_trade_time(name, if_night=False):
    name = name.lower()
    if name in ['btcq', 'ethq', 'ltcq', 'bchq', 'eosq', 'etcq', 'xrpq', 'btcindex','binance_btc',
                'bitfinex_btc', 'btcn', 'huobipro_btc', 'okex_btc', 'xbtusd']:
        if if_night:
            return [[0, 0], [24, 0]]
        else:
            return [[0, 0], [24, 0]]
    elif name == 'a':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'al':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'ag':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'au':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'bu':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'c':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'cf':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'cs':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'cu':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'fg':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'hc':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'i':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'j':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'jm':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'jd':
        if if_night:
            return [[9, 0], [15, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'l':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'm':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'ma':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'me':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'ni':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'oi':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'p':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'pb':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'pb2':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'pp':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'rb':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'rb_zl':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'rb_czl':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'rmz':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'rm':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'ru':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'sr':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'ap':
        if if_night:
            return [[9, 0], [15, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'sc':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'ta':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'eg':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'cj':
        if if_night:
            return [[9, 0], [15, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'yy' or name == 'y':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'zc':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'zn':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'b':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 't':
        if if_night:
            return [[9, 15], [15, 15]]
        else:
            return [[9, 15], [15, 15]]
    elif name == 'tf':
        if if_night:
            return [[9, 15], [15, 15]]
        else:
            return [[9, 15], [15, 15]]
    elif name == 'ts':
        if if_night:
            return [[9, 15], [15, 15]]
        else:
            return [[9, 15], [15, 15]]
    elif name == 'sm':
        if if_night:
            return [[9, 0], [15, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'sp':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'fu':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'sf':
        if if_night:
            return [[9, 0], [15, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'v':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'eb':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'ur':
        if if_night:
            return [[9, 0], [15, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'sn':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'nr':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'ss':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'pg':
        if if_night:
            return [[9, 0], [24, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'sf':
        if if_night:
            return [[9, 0], [15, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'sa':
        if if_night:
            return [[9, 0], [23, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'cy':
        if if_night:
            return [[9, 0], [23, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'lu':
        if if_night:
            return [[9, 0], [23, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'pf':
        if if_night:
            return [[9, 0], [23, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'lh':
        if if_night:
            return [[9, 0], [15, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'pk':
        if if_night:
            return [[9, 0], [15, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'si':
        if if_night:
            return [[9, 0], [15, 0]]
        else:
            return [[9, 0], [15, 0]]
    elif name == 'if':
        if if_night:
            return [[9, 30], [15, 0]]
        else:
            return [[9, 30], [15, 0]]
    elif name == 'ic':
        if if_night:
            return [[9, 30], [15, 0]]
        else:
            return [[9, 30], [15, 0]]
    elif name == 'im':
        if if_night:
            return [[9, 30], [15, 0]]
        else:
            return [[9, 30], [15, 0]]
    else:
        print ('[get_trade_time]Warning: no this commodity!')
        if if_night:
            return [[0, 0], [24, 0]]
        else:
            return [[9, 30], [15, 0]]



def get_unit_cost(name):
    name = name.lower()
    if name in ['btcq']:
        cost = 0.002
        tick_size = 1
        weight = 100
    elif name in ['ethq', 'ltcq', 'bchq', 'xrpq', 'eosq']:
        cost = 0.002
        tick_size = 1
        weight = 100
    elif name == 'a':
        cost = 10.0
        tick_size = 10
        weight = 40.0
    elif name == 'al':
        tick_size = 5
        cost = 25.0
        weight = 23.0
    elif name == 'ag':
        tick_size = 15.0
        cost = 18.0
        weight = 24.0
    elif name == 'au':
        tick_size = 1000
        cost = 25.0
        weight = 5.0
    elif name == 'bu':
        tick_size = 10
        cost = 20.0
        weight = 78.0
    elif name == 'c':
        tick_size = 10
        cost = 15.0
        weight = 98.0
    elif name == 'cf':
        tick_size = 5
        cost = 30.0
        weight = 19.0
    elif name == 'cs':
        tick_size = 10
        cost = 15.0
        weight = 80.0
    elif name == 'cu':
        tick_size = 5
        cost = 50.0
        weight = 8.0
    elif name == 'fg':
        tick_size = 20
        cost = 10.0
        weight = 65.0
    elif name == 'hc':
        tick_size = 10
        cost = 15.0
        weight = 52.0
    elif name == 'i':
        tick_size = 100
        cost = 50.0
        weight = 34.0
    elif name == 'j':
        tick_size = 100
        cost = 50.0
        weight = 10.0
    elif name == 'jm':
        tick_size = 60
        cost = 50.0
        weight = 20.0
    elif name == 'jd':
        tick_size = 10
        cost = 10.0
        weight = 10.0
    elif name == 'l':
        tick_size = 5
        cost = 25.0
        weight = 31.0
    elif name == 'm':
        tick_size = 10
        cost = 10.0
        weight = 51.0
    elif name == 'ma':
        tick_size = 10
        cost = 10.0
        weight = 65.0
    elif name == 'me':
        tick_size = 10
        cost = 10.0
        weight = 0.0
    elif name == 'ni':
        tick_size = 1
        cost = 15
        weight = 18.0
    elif name == 'oi':
        tick_size = 10
        cost = 15.0
        weight = 21.0
    elif name == 'p':
        tick_size = 10
        cost = 20.0
        weight = 25.0
    elif name == 'pb2':
        tick_size = 5
        cost = 25.0
        weight = 0.0
    elif name == 'pp':
        tick_size = 5
        cost = 10.0
        weight = 37.0
    elif name == 'rb':
        cost = 10.0
        tick_size = 10
        weight = 59.0
    elif name == 'pb':
        cost = 25.0
        tick_size = 5
        weight = 15.0
    elif name == 'rmz':
        tick_size = 10
        cost = 10.0
        weight = 0.0
    elif name == 'rm':
        tick_size = 10
        cost = 10.0
        weight = 64.0
    elif name == 'ru':
        tick_size = 10
        cost = 50.0
        weight = 11.0
    elif name == 'sm':
        tick_size = 5
        cost = 10.0
        weight = 35.0
    elif name == 'sf':
        tick_size = 5
        cost = 10.0
        weight = 35.0
    elif name == 'sr':
        tick_size = 10
        cost = 10.0
        weight = 26.0
    elif name == 'ta':
        tick_size = 5
        cost = 10.0
        weight = 59.0
    elif name == 'cj':
        tick_size = 5
        cost = 25.0
        weight = 22.0
    elif name == 'eg':
        tick_size = 10
        cost = 10.0
        weight = 30.0
    elif name == 'yy' or name == 'y':
        tick_size = 10
        cost = 20.0
        weight = 22.0
    elif name == 'zc':
        tick_size = 100
        cost = 24.0
        weight = 25.0
    elif name == 'ap':
        tick_size = 10
        cost = 30.0
        weight = 3.75
    elif name == 'sc':
        tick_size = 1000
        cost = 100.0
        weight = 3.8
    elif name == 'zn':
        tick_size = 5
        cost = 25.0
        weight = 16.0
    elif name == 'fu':
        tick_size = 10
        cost = 10.0
        weight = 55.0
    elif name == 'sp':
        tick_size = 10
        cost = 20.0
        weight = 30.0
    elif name == 't':
        tick_size = 10000
        cost = 50.0
        weight = 5.0
    elif name == 'tf':
        tick_size = 10000
        cost = 50.0
        weight = 5.0
    elif name == 'ts':
        tick_size = 20000
        cost = 100.0
        weight = 5.0
    elif name == 'v':
        tick_size = 5
        cost = 25.0
        weight = 45.0
    elif name == 'b':
        tick_size = 10
        cost = 10.0
        weight = 40.0
    elif name == 'sn':
        tick_size = 1
        cost = 10.0
        weight = 12.0
    elif name == 'ur':
        tick_size = 20
        cost = 20.0
        weight = 49.0
    elif name == 'eb':
        tick_size = 5
        cost = 5.0
        weight = 45.0
    elif name == 'nr':
        tick_size = 10
        cost = 50.0
        weight = 15.0
    elif name == 'ss':
        tick_size = 5
        cost = 25
        weight = 22
    elif name == 'pg':
        tick_size = 20
        cost = 20
        weight = 23
    elif name == 'sf':
        tick_size = 5
        cost = 10
        weight = 49
    elif name == 'sa':
        tick_size = 20
        cost = 20
        weight = 50
    elif name == 'cy':
        tick_size = 5
        cost = 25
        weight = 14
    elif name == 'lu':
        tick_size = 10
        cost = 10
        weight = 40
    elif name == 'pf':
        tick_size = 5
        cost = 10
        weight = 30
    elif name == 'lh':
        tick_size = 16
        cost = 80
        weight = 2.5
    elif name == 'pk':
        tick_size = 5
        cost = 10
        weight = 20
    elif name == 'si':
        tick_size = 5
        cost = 25
        weight = 11 
    elif name == 'if':
        tick_size = 300
        cost = 60.0 * 2.5
        weight = 1.0
    elif name == 'ic':
        tick_size = 200
        cost = 40.0 * 3.5
        weight = 0.8
    elif name == 'im':
        tick_size = 200
        cost = 40.0 * 3.5
        weight = 1.0
    elif name == 'ih':
        tick_size = 300
        cost = 60.0 * 2.5
        weight = 1.0
    else:
        print ('[get_unit_cost]Warning: no this commodity! ', name)
        cost = 10.0
        tick_size = 10
        weight = 10.0
    cost = cost * 2.0 / tick_size
    return cost, tick_size, weight



# --------------- end of linear_regression ----------------- #


def dump(fn, data):
    pkl_file = open(fn, 'wb')
    pickle.dump(data, pkl_file)
    pkl_file.close()


def load(fn):
    pkl_file = open(fn, 'rb')
    res = pickle.load(pkl_file)
    pkl_file.close()
    return res


# ----------- multiprocessing ---------------------- #

def single_process(func, index, d, para_l, prefix, args):
    results = []
    if para_l is not None:
        for p in para_l:
            #print 'single_process', index, p,
            res = func(p, **args)
            #print re
            results.append(res)
    else:
        results = func(**args)
    fn = 'tmp/%s_%d.p' % (prefix, index)
    d[fn] = results
    #dump(fn, results)


def multiprocess(func, paras=[], name=None, n_processes=1, **args):
    d = Manager().dict()
    result = []
    if n_processes == 1:
        if paras == []:
            return func(**args)
        else:
            for p in paras:
                result.append(func(p, **args))
        return result
    l = len(paras)
    size = l // n_processes
    j = 0
    jobs = []
    prefix = func.__name__ + '_%s' % os.getpid()
    for i in range(n_processes):
        # print i, self.feature.para_range[j:end_]
        if i < l % n_processes:
            gap = size + 1
        else:
            gap = size
        if l == 0:
            tmp_paras = None
        else:
            tmp_paras = paras[j:j + gap]
        if name == None:
            namep = 'process%d' % (i+1)
        else:
            namep = name[i]
        p = Process(target=single_process, name=namep, args=(func, i, d, tmp_paras, prefix, args))
        jobs.append(p)
        j += gap
        if (j >= l) and (l != 0):
            break
    n_processes = len(jobs)
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()
    return collect_data(prefix, n_processes, d)


def collect_data(prefix, num, d):
    data = []
    for i in range(num):
        fn = 'tmp/%s_%d.p' % (prefix, i)
        data.extend(d[fn])
    return data

def cross(l1, l2):
    ind = ((l1 > l2) & (ref(l1, 1) <= ref(l2, 1)))
    return ind

@jit
def get_ps(b_sig, s_sig, c_sig):  # add
    l = len(b_sig)
    ps = np.zeros(l)
    for i in range(l):
        if c_sig[i] == 1 or (b_sig[i] > 0 and s_sig[i] > 0):
            ps[i] = 0
        elif b_sig[i] > 0 and s_sig[i] == 0:
            ps[i] = b_sig[i]
        elif b_sig[i] == 0 and s_sig[i] > 0:
            ps[i] = -s_sig[i]
        else:
            if i == 0:
                ps[i] = 0
            else:
                ps[i] = ps[i - 1]
    return ps


@jit
def rolling_ref(d, p, if_zero=False):
    m = np.zeros((len(d),), dtype=d.dtype)
    l = len(p)
    for i in range(l):
        if i < p[i]:
            if not if_zero:
                m[i] = d[0]
            else:
                m[i] = 0
        else:
            m[i] = d[i - p[i]]
    return m

def ref(d, p, if_zero=False):
    p_ = np.abs(p)
    if not isinstance(d, np.ndarray):
        d = np.array(d)
    m = np.zeros((len(d), ), dtype=d.dtype)
    try:
        len(p_)
        m = rolling_ref(d, p_, if_zero)
    except:
        if not if_zero:
            m[:p_] = d[0]
        else:
            m[:p_] = 0
        if p == 0:
            m[:] = d[:]
        else:
            m[p_:] = d[:-p_]
    return m

def resample_diff(timestamp, data, period):
    series = pd.Series(data, index=timestamp)
    last_series = series.resample(period, label='left', closed='right').last()
    last = last_series.values
    last_1 = ref(last, 1)
    last_1[0] = 0
    re = last - last_1
    timestamp_m = np.array(list(last_series.index))
    return timestamp_m, re

def nz(d):
    res = np.where((d == np.inf) | (d == -np.inf) | np.isnan(d), 0, d)
    return res

def _convert_bar_time_int(ls_bar_times_int):
    ls_tpl = []
    for start_int, end_int in ls_bar_times_int:
        start = datetime.timedelta(days=0, hours=start_int // 100, minutes=start_int % 100)
        if end_int < 900:
            # trading before dawn/wee hours/ early in the morning
            end = datetime.timedelta(days=1, hours=end_int // 100, minutes=end_int % 100)
        else:
            end = datetime.timedelta(days=0, hours=end_int // 100, minutes=end_int % 100)
        ls_tpl.append((start, end))
    return ls_tpl


@jit
def get_timenum(ts):
    tt = []
    for i in range(len(ts)):
        tt.append(10000*ts[i].time().hour+100*ts[i].time().minute+1*ts[i].time().second)
    return np.array(tt)

@jit
def get_bar_time_II(comm, period, if_night, start=None, end=None):
    comm = comm.upper()
    delt = datetime.timedelta(minutes=period)
    t = get_trade_time(comm, if_night)
    start_ind = (start is None) or (start[0] * 100 + start[1] <= t[0][0] * 100 + t[0][1])
    end_ind = (end is None) or (end[0] * 100 + end[1] >= t[1][0] * 100 + t[1][1])
    start = t[0] if start_ind else start
    end = t[1] if end_ind else end
    s = datetime.datetime(1, 1, 1, start[0], start[1], 0)
    if end[0] >= 24 :
        e = datetime.datetime(1, 1, 2, end[0] - 24, end[1], 0)
    else:
        e = datetime.datetime(1, 1, 1, end[0], end[1], 0)
    ts = []
    while s < e:
        ts.append(s.time())
        s = s + delt
    ts.append(e.time())
    # print ts
    tt = []
    if comm in ['RM', 'OI', 'TA', 'CF', 'FG', 'SR', 'TC', 'MA', 'A', 'B', 'M', 'Y', 'YY', 'J', 'I', 'JM', 'P', 'CU',
                'AL', 'NI', 'SN', 'ZN', 'PB', 'HC', 'BU', 'RB', 'RU', 'AU', 'AG', 'JD', 'C', 'L', 'M', 'CS', 'PP', 'ZC',
                'V', 'SM', 'AP', 'SC', 'FU','SP','B','EG','CJ','SN','EB','NR','UR', 'PG', 'SS',
                'SF', 'SA', 'CY', 'LU','PF','LH','PK','SI']:
        for i in range(1, len(ts)):
            if (ts[i] >= datetime.time(11, 30)) and (ts[i] <= datetime.time(13, 30)):
                if ts[i-1] >= datetime.time(11, 30):
                    continue
            if (ts[i] >= datetime.time(15, 0)) and (ts[i] <= datetime.time(21, 0)):
                if ts[i-1] >= datetime.time(15, 0):
                    continue
            tt.append([ts[i-1].hour * 100 + ts[i-1].minute,  ts[i].hour * 100+ts[i].minute])

    if comm in ['IF', 'IC', 'T', 'TF', 'IH']:
        for i in range(1, len(ts)):
            if (ts[i] >= datetime.time(11, 30)) and (ts[i] <= datetime.time(13, 0)):
                if ts[i-1] < datetime.time(11, 30):
                    tt.append([ts[i-1].hour * 100 + ts[i-1].minute, 1130])
                else:
                    continue
            elif (ts[i-1] >= datetime.time(11, 30)) and (ts[i-1] <= datetime.time(13, 0)):
                if ts[i] > datetime.time(13, 0):
                    tt.append([1300, ts[i].hour * 100 + ts[i].minute])
                else:
                    continue
            else:
                tt.append([ts[i - 1].hour * 100 + ts[i - 1].minute, ts[i].hour * 100 + ts[i].minute])

    return tt



#  动态窗口 hyh
# 将均值的回看窗口由固定的64，改为自上一次上/下传均线时起。下附一个相同逻辑的静态因子结果进行对比。
def rollingmean(d, p, if_zero=False):
    m = np.zeros((len(d),), dtype=d.dtype)
    l = len(p)
    for i in range(l):
        if i < p[i]-1:
            if not if_zero:
                m[i] = d[0]
            else:
                m[i] = 0
        else:
            m[i] = d[i-p[i]+1 : i+1].mean()
    return m

'''
止损
'''

def get_ps(b_sig, s_sig, c_sig):  # add
    l = len(b_sig)
    ps = np.zeros(l)
    for i in range(l):
        if c_sig[i] == 1 or (b_sig[i] > 0 and s_sig[i] > 0):
            ps[i] = 0
        elif b_sig[i] > 0 and s_sig[i] == 0:
            ps[i] = b_sig[i]
        elif b_sig[i] == 0 and s_sig[i] > 0:
            ps[i] = -s_sig[i]
        else:
            if i == 0:
                ps[i] = 0
            else:
                ps[i] = ps[i - 1]
    return ps
	
def bars_since_num(condition):
    l = len(condition)
    ret = np.zeros((l, ), dtype=np.int32)
    i = 0
    while i < l:
        if condition[i] == 1:
            ret[i] = 0
            i += 1
            j = i
            while j < l:
                ret[j] = ret[j - 1] + 1
                if condition[j] == 1:
                    break
                i += 1
                j += 1
        else:
            i += 1
    return ret

def iif(ind, a, b):
    return ind * 1 * a + (1 - ind) * b


@jit
def hhv_fliter(C, bars_num, start=1):
    l = len(C)
    mx_values = np.zeros((l,))
    mx_values[0] = C[0]
    mx = C[0]
    for i in range(1, l):
        if bars_num[i] != start:
            mx = max(C[i], mx)
            mx_values[i] = mx
        else:
            mx = C[i]
            mx_values[i] = mx
    return mx_values

@jit
def elf_rolling_max(d, period):
    l = d.shape[0]
    res = np.zeros(l)
    for i, p in zip(range(l), period):
        res[i] = d[max(0, i - p + 1): i + 1].max()
    return res

def MAXrolling(d, period):
    l = len(d)
    ls = min(period, l)
    res = np.zeros(l)
    res[0] = d[0]
    for i in range(1, ls):
        res[i] = ta.MAX(d[:i+1], timeperiod=i+1)[-1]
    res[ls:] = ta.MAX(d, timeperiod=ls)[ls:]
    return res

def HHV(C, bars_num, start=1):# modified
    try:
        len(bars_num)
        #mx = C[0]
        if 0 in bars_num:
            mx_values = hhv_fliter(C, bars_num, start)
        else:
            mx_values = elf_rolling_max(C, bars_num)
    except:
        if bars_num > 1:
            mx_values = MAXrolling(C, bars_num)
        else:
            mx_values = copy.copy(C)
    return mx_values

@jit
def llv_filter(C, bars_num, start=1):
    l = len(C)
    mn_values = np.zeros((l,))
    mn_values[0] = C[0]
    mn = C[0]
    for i in range(1, l):
        if bars_num[i] != start:
            mn = min(C[i], mn)
            mn_values[i] = mn
        else:
            mn = C[i]
            mn_values[i] = mn
    return mn_values


@jit
def elf_rolling_min(d, period):
    l = d.shape[0]
    res = np.zeros(l)
    for i, p in zip(range(l), period):
        res[i] = d[max(0, i - p + 1): i + 1].min()
    return res


def MINrolling(d, period):
    l = len(d)
    ls = min(l, period)
    res = np.zeros(l)
    res[0] = d[0]
    for i in range(1, ls):
        res[i] = ta.MIN(d[:i+1], timeperiod=i+1)[-1]
    res[ls:] = ta.MIN(d, timeperiod=ls)[ls:]
    return res

def LLV(C, bars_num, start=1): # modified
    try:
        len(bars_num)
        #mn = C[0]
        if 0 in bars_num:
            mn_values = llv_filter(C, bars_num, start)
        else:
            mn_values = elf_rolling_min(C, bars_num)
    except:
        if bars_num > 1:
            mn_values = MINrolling(C, bars_num)
        else:
            mn_values = copy.copy(C)
    return mn_values
