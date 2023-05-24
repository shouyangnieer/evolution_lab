import numpy as np
from . import elutil as eu
from numba import jit
import pandas as pd
import traceback


def get_profit_factor_by_profits(profits):
    tmp = profits
    ind_win = tmp > 0
    ind_loss = tmp < 0
    loss = np.sum(tmp[ind_loss])
    if loss == 0:
        return 3
    return -np.sum(tmp[ind_win]) * 1.0 / loss


def get_period_win_rate(timestamp, net_accu_profits, period):
    timestamp_m, diff = eu.resample_diff(timestamp=timestamp, data=net_accu_profits, period=period)
    w_rate = 1.0 * np.sum(list(diff > 0)) / len(diff)
    return w_rate


def get_profit_per_trade(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close
    accu_profits = np.cumsum(ps * delta_close) * unit_cost[1]
    accu_profits[1:] = accu_profits[:-1]
    accu_profits[0] = 0

    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    net_accu_profits = accu_profits - np.cumsum(ps_change * .5 * unit_cost[0]) * unit_cost[1]
    num = get_trades_num(ps)
    res = 1.0 * net_accu_profits[-1] / num
    return res, num


def get_draw_down(accu_profits):
    return accu_profits - np.maximum.accumulate(accu_profits)


def get_win_rate_by_profits(profits):
    num_wins = sum(profits > 0)
    if sum(profits > 0) + sum(profits < 0) == 0:
        return 0
    else:
        return num_wins * 1.0 / (sum(profits > 0) + sum(profits < 0))


def get_accu_profits(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close
    accu_profits = np.cumsum(ps * delta_close) * unit_cost[1]
    accu_profits[1:] = accu_profits[:-1]
    accu_profits[0] = 0

    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    net_accu_profits = accu_profits - np.cumsum(ps_change * .5 * unit_cost[0]) * unit_cost[1]

    return accu_profits, net_accu_profits


def get_accu_profits_ab(ps, data, unit_cost, type='delta_ratio_close'):
    delta_close = data.delta_close
    accu_profits = np.cumsum(ps * delta_close) * unit_cost[1]
    accu_profits[1:] = accu_profits[:-1]
    accu_profits[0] = 0

    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    net_accu_profits = accu_profits - np.cumsum(ps_change * .5 * unit_cost[0]) * unit_cost[1]

    return accu_profits, net_accu_profits


def get_market_time_percent(ps):
    res = 100 * np.sum(np.abs(ps)) / len(ps)
    return res


def get_profits(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close
    GP = ps * delta_close * unit_cost[1]
    GP[1:] = GP[:-1]
    GP[0] = 0
    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    NP = GP - ps_change * .5 * unit_cost[0] * unit_cost[1]
    return GP, NP

def get_profits_ab(ps, data, unit_cost, type='delta_ratio_close'):
    if isinstance(data, np.ndarray):
        delta_close = data
    else:
        delta_close = data.delta_close
    GP = ps * delta_close * unit_cost[1]
    GP[1:] = GP[:-1]
    GP[0] = 0
    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    NP = GP - ps_change * .5 * unit_cost[0] * unit_cost[1]
    return GP, NP


def get_profits_less(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close
    GP = ps * delta_close * unit_cost[1]
    GP[1:] = GP[:-1]
    GP[0] = 0
    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    NP = GP - ps_change * .5 * unit_cost[0] * unit_cost[1] * 2
    return GP, NP


def get_profits_less_ab(ps, data, unit_cost, type='delta_ratio_close'):
    if isinstance(data, np.ndarray):
        delta_close = data
    else:
        delta_close = data.delta_close
    GP = ps * delta_close * unit_cost[1]
    GP[1:] = GP[:-1]
    GP[0] = 0
    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    NP = GP - ps_change * .5 * unit_cost[0] * unit_cost[1] * 2
    return GP, NP


def get_net_profit(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close
    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    assert isinstance(ps, np.ndarray)
    return np.sum(ps * delta_close - ps_change * .5 * unit_cost[0]) * unit_cost[1]

def get_waccu_profits(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close

    weight = np.ones(np.shape(ps))
    steplen = int(len(ps)/4.0)
    weight[steplen:2*steplen] = 2
    weight[2*steplen:3*steplen] = 3
    weight[3*steplen:] = 4

    accu_profits = np.cumsum(ps * weight * delta_close) * unit_cost[1]
    accu_profits[1:] = accu_profits[:-1]
    accu_profits[0] = 0

    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    net_accu_profits = accu_profits - np.cumsum(ps_change * weight * .5 * unit_cost[0]) * unit_cost[1]

    return accu_profits, net_accu_profits

def get_wnet_profit_less(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_waccu_profits(ps, data, unit_cost, type)
    res = NP[-1] - (GP[-1] - NP[-1])*5
    return res

def get_wnet_profit_less2(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_waccu_profits(ps, data, unit_cost, type)
    res = NP[-1] - (GP[-1] - NP[-1])*7
    return res

def get_wnet_profit_less3(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_waccu_profits(ps, data, unit_cost, type)
    res = NP[-1] - (GP[-1] - NP[-1])*10
    return res

def get_wnet_profit_less5(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_waccu_profits(ps, data, unit_cost, type)
    res = NP[-1] - (GP[-1] - NP[-1])*5
    return res

def get_wnet_profit_less10(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_waccu_profits(ps, data, unit_cost, type)
    res = NP[-1] - (GP[-1] - NP[-1])*10
    return res

def get_wnet_profit_less20(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_waccu_profits(ps, data, unit_cost, type)
    res = NP[-1] - (GP[-1] - NP[-1])*20
    return res

def get_wnet_profit(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close
    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    weight = np.ones(np.shape(ps))
    steplen = int(len(ps)/4.0)
    weight[steplen:2*steplen] = 2
    weight[2*steplen:3*steplen] = 3
    weight[3*steplen:] = 4
    assert isinstance(ps, np.ndarray)
    return np.sum(ps * weight * delta_close - ps_change * weight * .5 * unit_cost[0]) * unit_cost[1]


def get_net_profit_ab(ps, data, unit_cost, type='delta_ratio_close'):
    delta_close = data.delta_close
    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    assert isinstance(ps, np.ndarray)
    return np.sum(ps * delta_close - ps_change * .5 * unit_cost[0]) * unit_cost[1]


def get_max_draw_down(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close
    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    assert isinstance(ps, np.ndarray)
    net_accu_profits = np.cumsum(ps * delta_close - ps_change * .5 * unit_cost[0]) * unit_cost[1]
    draw_down = get_draw_down(net_accu_profits)
    md = np.min(draw_down)
    return md


def get_car_vs_mdd(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_accu_profits(ps, data, unit_cost, type=type)
    draw_down = get_draw_down(NP)
    mdd = np.min(draw_down)
    if mdd == 0:
        n_v_m = 50
    else:
        n_v_m = -1.0 * NP[-1] / mdd
    return n_v_m


def get_profit_factor(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close
    profits = get_trade_profits(ps, delta_close, unit_cost)
    return get_profit_factor_by_profits(profits)


def get_sharp_ratio_std(ps, data, unit_cost, type='delta_ratio_close'):
    timestamp = data.timestamps
    GP, NP = get_profits(ps, data, unit_cost, type)
    series = pd.Series(NP, index=timestamp)
    d = series.resample('D').sum()
    st = np.nanstd(d.values)
    sr = 0 if st == 0 else np.nanmean(d) / st
    return sr


def get_sharp_ratio(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_profits(ps, data, unit_cost, type)
    assert isinstance(NP, np.ndarray)
    st = np.nanstd(NP)
    sr = 0 if st == 0 else np.nanmean(NP) / st
    return sr


def get_sharp_ratio_less(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_profits_less(ps, data, unit_cost, type)
    assert isinstance(NP, np.ndarray)
    st = np.nanstd(NP)
    sr = 0 if st == 0 else np.nanmean(NP) / st
    return sr


def get_rf3(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_profits(ps, data, unit_cost, type)
    assert isinstance(NP, np.ndarray)
    acc_np = np.cumsum(NP)
    dd = np.maximum.accumulate(acc_np) - acc_np
    mdd3 = dd[dd >= np.percentile(dd, 95)].mean()
    rf3 = 0 if mdd3 == 0 else acc_np[-1] / mdd3
    return rf3


def get_win_num_by_profits(profits):
    num_wins = sum(profits > 0)
    return num_wins

def get_win_num(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close

    profits = get_trade_profits(ps, delta_close, unit_cost)
    return get_win_num_by_profits(profits)

def get_np_winnum_by_profits(profits):
    num_wins = sum(profits > 0)
    np0 = np.sum(profits)
    return num_wins*np0

def get_np_winnum(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close

    profits = get_trade_profits(ps, delta_close, unit_cost)
    return get_np_winnum_by_profits(profits)


def get_net_profit_less(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_accu_profits(ps, data, unit_cost, type)
    res = NP[-1] - (GP[-1] - NP[-1])
    return res


def get_net_profit_less_ab(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_accu_profits_ab(ps, data, unit_cost)
    res = NP[-1] - (GP[-1] - NP[-1])
    return res


def get_net_profit_less2(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_accu_profits(ps, data, unit_cost, type)
    res = NP[-1] - (GP[-1] - NP[-1]) * 2
    return res


def get_net_profit_less2_ab(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_accu_profits_ab(ps, data, unit_cost)
    res = NP[-1] - (GP[-1] - NP[-1]) * 2
    return res


def get_net_profit_less3(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_accu_profits(ps, data, unit_cost, type)
    res = NP[-1] - (GP[-1] - NP[-1]) * 3
    return res


def get_net_profit_less3_ab(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_accu_profits_ab(ps, data, unit_cost)
    res = NP[-1] - (GP[-1] - NP[-1]) * 3
    return res


def get_net_profit_ratio(ps, data, unit_cost, type='delta_ratio_close'):
    GP, NP = get_accu_profits(ps, data, unit_cost, type)
    #mid = NP[-1] / GP[-1] * 100
    res = 0 if (GP[-1] <= 0) or (NP[-1] <= 0) else np.power(NP[-1], 2) / GP[-1]
    return res


def get_win_rate(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close
    profits = get_trade_profits(ps, delta_close, unit_cost)
    return get_win_rate_by_profits(profits)


def get_performance(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close
    profits = get_trade_profits(ps, delta_close, unit_cost)
    trading_times = len(profits)
    total_profit = np.sum(profits)
    ppt = total_profit / trading_times
    win_rate = get_win_rate_by_profits(profits)
    pf = get_profit_factor_by_profits(profits)
    assert isinstance(ps, np.ndarray)
    net_accu_profits = np.cumsum(ps * delta_close - np.abs(eu.self_diff(ps, 1)) * .5 * unit_cost[0]) * unit_cost[1]
    draw_down = get_draw_down(net_accu_profits)
    md = np.min(draw_down)
    return trading_times, win_rate, pf, md, ppt, total_profit


def get_trades_num(ps, data=None, unit_cost=None, type='delta_ratio_close'):
    nums2 = np.abs(ps - eu.ref(ps, 1))
    trade_num = np.sum(nums2) / 2
    return trade_num


def get_profit_tarde_ratio(ps, sdata, unit_cost, type='delta_ratio_close'):
    ps_1 = eu.ref(ps, 1)
    df_close = pd.DataFrame({'close': sdata.close, 'timestamp': sdata.timestamps})
    df_close = df_close.set_index(['timestamp'])
    df_first = df_close[(ps_1 != ps) & (ps_1 == 0)]
    df_first = df_first.rename(columns={'close': 'first'})
    df = pd.concat([df_close, df_first], axis=1)
    df = df.fillna(method='ffill')
    df = df.fillna(0)
    first = df['first'].values
    delta_close = sdata.delta_close
    ind, tmp = eu.remove_zero(first)
    ratio = eu.iif(ind, 0, delta_close / tmp)
    GP = ps * ratio * unit_cost[1]
    GP[1: ] = GP[: -1]
    GP[0] = 0
    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    NP = GP - ps_change * .5 * unit_cost[0] * unit_cost[1]
    return GP, NP


def get_accu_profits_trade_ratio(ps, data, unit_cost, type='delta_ratio_close'):
    if type == 'delta_close':
        delta_close = data.delta_close
    elif type == 'delta_ratio_close':
        delta_close = data.delta_ratio_close
    elif type == 'delta_open':
        delta_close = data.delta_open
    elif type == 'delta_ratio_open':
        delta_close = data.delta_ratio_open
    else:
        delta_close = data.delta_ratio_close
    accu_profits = np.cumsum(ps * delta_close) * unit_cost[1]
    accu_profits[1:] = accu_profits[:-1]
    accu_profits[0] = 0

    ps_change = np.abs(eu.self_diff(ps, 1))
    ps_change[0] = np.abs(ps[0])
    net_accu_profits = accu_profits - np.cumsum(ps_change * .5 * unit_cost[0]) * unit_cost[1]

    return accu_profits, net_accu_profits

@jit
def get_trade_profits(ps, delta_close, unit_cost):

    previous = ps[0]
    profit = delta_close[0] * previous
    # num_trades = 0
    profits = []
    for i in xrange(1, len(ps)):
        current = ps[i]
        if previous != current and previous != 0:
            profits.append(profit)
            profit = 0
            # num_trades += 1
        profit += delta_close[i] * current
        previous = current
    return np.array(profits) - unit_cost[0]