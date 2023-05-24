import  sys
sys.path = ['/data/xianliang/project_ctaintern/']+sys.path

import os
import pyelf.elutil as eu
import pyelf.commodity_list as commlist
import datetime
import os
import itertools
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"

import warnings

warnings.filterwarnings('ignore')

def LoadCsv(filepath, names, beg_date=None):
    data = pd.read_csv(filepath, names=names)
    data['datestr']=data['date'].apply(lambda x: datetime.datetime.strptime(format(x,'06d'),'%Y%m%d').strftime('%Y/%m/%d'))
    data['timestr']=data['time'].apply(lambda x: datetime.datetime.strptime(format(x,'06d'),'%H%M%S').strftime('%H:%M:%S'))

    data = data.set_index(pd.to_datetime(data['datestr']+' '+data['timestr']))
    data = data[~data.index.duplicated()]
    data.index.name = 'index'
    if beg_date is not None:
        data = data[data.index>=beg_date]
    return data

def Load1min(symbol, data_path, if_parquet=False, beg_date=None, col_type=0):
    print ('Loading 1min data...%s'%symbol)
    if if_parquet:
        filepath = os.path.join(data_path, symbol.lower() + '.parquet')
        _df_1min = pd.read_parquet(filepath)
    else:
        filepath = os.path.join(data_path, symbol.upper()+'.csv')
        if col_type==0:
            _df_1min = LoadCsv(filepath, names=['date','time','open','high','low','close','vol','open_int'])
        elif col_type==1:
            _df_1min = LoadCsv(filepath, names=['date','time','open','high','low','close','vol','open_int','turnover'])
    if beg_date is not None:
        _df_1min = _df_1min[_df_1min.index>=beg_date]
    return _df_1min

def arrange_data(data_kbar, period, comms, col_type):
    dflist = list()
    for comm in comms:
        tmpdf = data_kbar[comm]
        tmpdf.index.name = 'index'
        if col_type==0:
            tmpdf = tmpdf[['open', 'high', 'low', 'close', 'vol', 'open_int']]
            newnames = ['O','H','L','C','V','OPI']
        elif col_type==1:
            tmpdf = tmpdf[['open', 'high', 'low', 'close', 'vol', 'open_int', 'turnover']]
            newnames = ['O','H','L','C','V','OPI','amount']
        newcols = map(lambda x: '%s-%s'%(comm, x), newnames)
        tmpdf.columns = newcols
        dflist.append(tmpdf)
    tdf = pd.concat(dflist, axis=1)
    data_dict = {}
    data_dict['period'] = period

    for icol in newnames:
        cols = map(lambda x: '%s-%s'%(x, icol), comms)
        tmpdf = tdf[cols]
        tmpdf.columns = comms

        if icol=='C':
            df_ifjy = 1 - 1 * pd.isnull(tmpdf)
            data_dict['ifjy'] = df_ifjy
            tmpdf = tmpdf.fillna(method='ffill')
            data_dict['C'] = tmpdf
        elif icol in ['O','H','L','OPI']:
            tmpdf = tmpdf.fillna(method='ffill')
            data_dict[icol] = tmpdf
        elif icol in ['V', 'amount']:
            tmpdf = tmpdf.fillna(0)
            data_dict[icol] = tmpdf

    return data_dict



def getKBar(para, data1min, col_type):
    symbol, period, if_night = para

    _df_1min = data1min[symbol.lower()]

    # if period==1:
    #     return _df_1min

    tpl_start = None
    tpl_end = None

    ls_bar_times = eu.get_bar_time_II(
        symbol, period, if_night=if_night, start=tpl_start, end=tpl_end
    )

    # extract tradings days only, Monday - Friday for start time
    ls_trading_days = pd.to_datetime(sorted(list(set(_df_1min.index[_df_1min.index.weekday <= 4].date))))
    # bar time int to timedelta, to add later
    ls_bar_time_delta = eu._convert_bar_time_int(ls_bar_times)
    generator_interval = itertools.product(ls_trading_days, ls_bar_time_delta)
    ls_interval = list(map(lambda x: (x[0] + x[1][0], x[0] + x[1][1]), generator_interval))
    # [left, right), left inclusive, right exclusive, pd.Interval, to be used as groupby


    ls_interval_index = pd.IntervalIndex.from_tuples(ls_interval, closed='left')

    """
    use pd.cut to generate groupby index list
    for time outside bar time, like 02:00 am, a NaN is generated for this group. Then in df.groupby, group named by nan is
    simply skipped, and so for the data within that interval
    """
    groupby_idx = pd.cut(
        _df_1min.index, bins=ls_interval_index
    )
    grouped = _df_1min.groupby(groupby_idx)

    """
    for interval data that contains NaN, open/high/low... is calculated by skipping na. This is desired behaviour, since
    history data might not be perfect, do with what we have is the most practical way.
    """
    if col_type==0:
        df_resample = grouped.agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'sum',
            'open_int': 'last'
        }).reset_index().dropna()  # dropna due to missing data, within legit trade time
    elif col_type==1:
        df_resample = grouped.agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'sum',
            'open_int': 'last',
            'turnover': 'sum'
        }).reset_index().dropna()  # dropna due to missing data, within legit trade time
    _interval_index = pd.IntervalIndex(df_resample['index'])
    df_resample['timestamps'] = _interval_index.left
    df_resample['timestamps_end'] = _interval_index.right

    if col_type==0:
        df_resample = df_resample[['open', 'high', 'low', 'close', 'vol', 'open_int', 'timestamps', 'timestamps_end']]
    elif col_type==1:
        df_resample = df_resample[['open', 'high', 'low', 'close', 'vol', 'open_int', 'turnover', 'timestamps', 'timestamps_end']]
    _df_kbar = df_resample.set_index('timestamps')

    return _df_kbar


def getPeriodbar(period, data1min, if_night, comms, col_type):

    print ('getPeriodbar %dmin, %s......'%(period, str(if_night)))
    paras = list(itertools.product(comms, [period], [if_night]))
    kbar_history = eu.multiprocess(getKBar, paras=paras, n_processes=len(paras), data1min=data1min, col_type=col_type)
    data_kbar = dict(zip(comms, kbar_history))
    datadict = arrange_data(data_kbar, period, comms, col_type)

    print ('getPeriodbar %dmin, %s......,Finished!'%(period, str(if_night)))
    return datadict


comms = commlist.comms_all
# comms = ['pk','pf','si']

beg_date = datetime.datetime(2016,1,1)
# beg_date = datetime.datetime(2022,1,1)

label_type = 'left'

data_path = '/data/xianliang/data/assembled_data/_daynight/'
data_flag = 'index'
col_type = 0

# data_path = '/data/xianliang/future_1m_zhuli1_20100101_complexbackright/'
# data_flag = 'zl'
# col_type = 1

'''
多线程函数 multiprocess
'''

results = eu.multiprocess(Load1min, paras=comms, n_processes=min(60, len(comms)), data_path = data_path, if_parquet=False, beg_date=beg_date, col_type=col_type)
data1mincsv = dict(zip(comms, results))

print ('Data Loaded Succesfully!!!')

# data1min = arrange_data(data1mincsv, 1, comms, col_type)
data1min = getPeriodbar(1, data1min=data1mincsv, if_night=True, comms=comms, col_type=col_type)



O1min = data1min['O']
H1min = data1min['H']
L1min = data1min['L']
C1min = data1min['C']
V1min = data1min['V']
I1min = data1min['OPI']
JY1min = data1min['ifjy']

ending = C1min.index[-1]

dstdir = '/data/xianliang/project_ctaintern/data%d_%s/%s/'%(beg_date.year, label_type, data_flag)

pvdf = pd.DataFrame(index=C1min.index)
for x in comms:
    cost, tick_size, weight = eu.get_unit_cost(x)
    pvdf[x] = tick_size

Avgp1min = (O1min+2*C1min+H1min+L1min)/5.0

if col_type==0:
    Amount1min = V1min*pvdf*Avgp1min
elif col_type==1:
    Amount1min = data1min['amount']

shift = 0

target_periods = [5]
for target_period in target_periods:

    ipath = os.path.join(dstdir,'%dmin_shift%d'%(target_period, shift))
    if not os.path.exists(ipath):
        os.makedirs(ipath)

    O = O1min.resample(rule='%sT' % target_period, base=shift, label=label_type).first()
    H = H1min.resample(rule='%sT' % target_period, base=shift, label=label_type).max()
    L = L1min.resample(rule='%sT' % target_period, base=shift, label=label_type).min()
    C = C1min.resample(rule='%sT' % target_period, base=shift, label=label_type).last()
    V = V1min.resample(rule='%sT' % target_period, base=shift, label=label_type).sum()
    I = I1min.resample(rule='%sT' % target_period, base=shift, label=label_type).last()
    Twap = C1min.resample(rule='%sT' % target_period, base=shift, label=label_type).mean()

    JY = JY1min.resample(rule='%sT' % target_period, base=shift, label=label_type).max()

    Vwap = (C1min*V1min).resample(rule='%sT' % target_period, base=shift, label=label_type).sum()/V

    Amount = Amount1min.resample(rule='%sT' % target_period, base=shift, label=label_type).sum()
    precipitation = (I1min*pvdf*Avgp1min).resample(rule='%sT' % target_period, base=shift, label=label_type).last()

    O = O.dropna(how='all')
    H = H.dropna(how='all')
    L = L.dropna(how='all')
    C = C.dropna(how='all')
    Twap = Twap.dropna(how='all')

    JY = JY.reindex(C.index)

    on_day = JY.copy()
    on_day['index'] = on_day.index
    on_day = on_day.resample('1D', base=shift, label=label_type).first()
    on_day[comms] = on_day[comms].fillna(0)
    on_day[comms] = on_day[comms].cumsum()
    on_day = on_day.reset_index(drop=True).set_index('index')
    on_day = on_day[~on_day.index.duplicated()]
    # print (on_day.tail())

    on_day = on_day.reindex(C.index)
    on_day = on_day.fillna(method='ffill').fillna(0)


    V = V.reindex(C.index)
    I = I.reindex(C.index)
    Vwap = Vwap.reindex(C.index)
    df_ifjy = JY.reindex(C.index)
    Amount = Amount.reindex(C.index)

    Vwap = Vwap.fillna(method='ffill')


    O.to_parquet(os.path.join(ipath, 'openprice.parquet'))
    H.to_parquet(os.path.join(ipath, 'highprice.parquet' ))
    L.to_parquet(os.path.join(ipath, 'lowprice.parquet' ))
    C.to_parquet(os.path.join(ipath, 'closeprice.parquet' ))
    V.to_parquet(os.path.join(ipath, 'volume.parquet'))
    I.to_parquet(os.path.join(ipath, 'opi.parquet'))

    Amount.to_parquet(os.path.join(ipath, 'amount.parquet'))

    df_ifjy.to_parquet(os.path.join(ipath, 'if_jy.parquet'))

    df_ifjy.to_parquet(os.path.join(ipath, 'if_st.parquet'))
    df_ifjy.to_parquet(os.path.join(ipath, 'if_zdt.parquet'))

    Twap.to_parquet(os.path.join(ipath, 'twap.parquet'))
    Vwap.to_parquet(os.path.join(ipath, 'vwap.parquet'))
    precipitation.to_parquet(os.path.join(ipath, 'precipitation.parquet'))
    on_day.to_parquet(os.path.join(ipath, 'on_day.parquet'))

    print (C.tail())
    print ('================')

    print (Twap.tail())
    print ('================')

    print (Vwap.tail())
    print ('================')