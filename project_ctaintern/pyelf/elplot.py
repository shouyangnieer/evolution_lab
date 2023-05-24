from . import elutil as eu
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def plot_profits(ax, accu_profits, net_accu_profits, draw_down, timestamps, fonsize=16, title=''):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(fonsize)
    l = len(accu_profits)
    x_ = range(l)
    ind, date_str = get_timestamp_ticks(timestamps)
    ax.plot(accu_profits, 'y')
    # y_max = np.max(accu_profits)
    ax.plot(net_accu_profits, 'g')
    ax.fill_between(x_, accu_profits, 0, where=accu_profits >= 0, facecolor='yellow', interpolate=True)
    ax.fill_between(x_, net_accu_profits, 0, where=net_accu_profits >= 0, facecolor='green', alpha=.5, interpolate=True)
    ax.plot(draw_down, 'r')
    # y_min = np.min(draw_down)
    ax.fill_between(x_, draw_down, 0, where=draw_down <= 0, facecolor='red', alpha=.5, interpolate=True)
    # ax[0].plot([train_size, train_size], [y_min, y_max], 'k--')
    ax.grid()
    ax.set_title(title, size=fonsize + 8)
    ax.set_xlim([0, l])
    ax.set_xticks(ind)
    ax.set_xticklabels(date_str)


def plot_profit_per_month(ax, net_accu_profits, timestamp, fonsize=16):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(fonsize)
    timestamp_m, df_mon_diff = eu.resample_diff(timestamp=timestamp, data=net_accu_profits, period='1M')
    df_mon_diff1 = df_mon_diff.copy()
    df_mon_diff2 = df_mon_diff.copy()
    df_mon_diff1[df_mon_diff1 < 0] = 0
    df_mon_diff2[df_mon_diff2 > 0] = 0
    ax.bar(timestamp_m, df_mon_diff1, width=fonsize/2, color='g')
    ax.bar(timestamp_m, df_mon_diff2, width=fonsize/2, color='r')
    mons = mdates.MonthLocator(interval=1)
    mons_ = mdates.DateFormatter('%m')
    ax.xaxis.set_major_locator(mons)
    ax.xaxis.set_major_formatter(mons_)



def plot_margin(ax, margin_all, timestamp, fonsize=16):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(fonsize)
    ax.axis([timestamp[0], timestamp[-1], -1, max(margin_all) * 1.1])
    ax.fill_between(timestamp, margin_all, 0, where=None, facecolor='blue')
    # plt.plot(tdatetime,margin_all)
    # plt.xlabel('Date',fontsize = fonsiz)
    plt.ylabel('Margin', fontsize=fonsize)

def plot_probability(ax, prob, timestamp, ylabel='Probability',fonsize=16):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(fonsize)
    ax.axis([timestamp[0], timestamp[-1], 0.9*min(prob), max(prob) *1.1])
    ax.plot(timestamp, prob)
    ax.grid()
    plt.ylabel(ylabel, fontsize=fonsize)


def plot_trades_num(ax, ps, timestamp, fonsize=16):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(fonsize)
    ax.axis([timestamp[0], timestamp[-1], -1, max(ps) * 1.1])
    ax.fill_between(timestamp, ps, 0, where=None, facecolor='yellow')
    plt.xlabel('Date', fontsize=fonsize)
    plt.ylabel('# Products are trading', fontsize=fonsize)


def get_timestamp_ticks(timestamps):
    ind = []
    date_str = []
    i = 0
    months = [4, 10]
    flags = [0, 0] 
    for t in timestamps:
        for j in  range(len(months)):
            if t.month == months[j] and flags[j] == 0:
                flags[j-1] = 0
                ind.append(i)
                s = "%d'%02d" % (t.year, t.month)
                date_str.append(s[2:])
                flags[j] = 1
        i += 1
    return ind, date_str
