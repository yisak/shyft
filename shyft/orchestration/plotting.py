import numpy as np
import matplotlib.pylab as plt
from matplotlib import ticker
from matplotlib.patches import Rectangle
from shyft.api.__init__ import TsFixed
from shyft.api import deltahours


def blend_colors(color1, color2):
    if len(color1) != 4 or len(color2) != 4:
        raise ValueError("Both colors must be of length 4")
    r_alpha = 1 - (1 - color1[-1])*(1 - color2[-1])
    # r_alpha = color1[-1] + color2[-1]*(1 - color1[-1])  # Alternative blending strategy for alpha layer
    return list(np.asarray(color1[:-1])*color1[-1]/r_alpha +
                np.asarray(color2[:-1])*color2[-1]*(1 - color1[-1])/r_alpha) + [r_alpha]


def plot_np_percentiles(time, percentiles, base_color=1.0, alpha=0.5, plw=0.5, linewidth=1, mean_color=0.0, label=None):
    if base_color is not None:
        if not isinstance(base_color, np.ndarray):
            if isinstance(base_color, (int, float)):
                base_color = 3*[base_color]
            base_color = np.array(base_color)
    if not isinstance(mean_color, np.ndarray):
        if isinstance(mean_color, (int, float)):
            mean_color = 3*[mean_color]
        mean_color = np.array(mean_color)
    percentiles = list(percentiles)
    num_intervals = len(percentiles)//2
    f_handles = []
    proxy_handles = []
    prev_facecolor = None
    for i in range(num_intervals):
        facecolor = list(base_color) + [alpha]
        f_handles.append(plt.fill_between(time, percentiles[i], percentiles[-(i+1)],
                         edgecolor=(0, 0, 0, 0), facecolor=facecolor))
        proxy_handles.append(Rectangle((0, 0), 1, 1, fc=blend_colors(prev_facecolor, facecolor) if
                             prev_facecolor is not None else facecolor))
        prev_facecolor = facecolor
    linewidths = len(percentiles)*[plw]
    linecols = len(percentiles)*[(0.7, 0.7, 0.7, 1.0)]
    labels = len(percentiles)*[None]
    if len(percentiles) % 2:
        mid = len(percentiles)//2
        linewidths[mid] = linewidth
        linecols[mid] = mean_color
        labels[mid] = label
    handles = []
    for p, lw, lc, label in zip(percentiles, linewidths, linecols, labels):
        h, = plt.plot(time, p, linewidth=lw, color=lc, label=label)
        handles.append(h)
    if len(percentiles) % 2:
        mean_h = handles.pop(len(handles)//2)
        handles = [mean_h] + handles
    return (handles + f_handles), proxy_handles


def plot_timeseries(time_series, calendar, labels=None, use_week_marks=False):
    n_figs = len(time_series)
    ax = plt.subplot(n_figs, 1, 1)
    t_start = None
    t_stop = None
    # Convert time series to simple structures and find min/max in time dimension
    times = []
    values = []
    for i, ts in enumerate(time_series):
        if isinstance(ts, TsFixed):
            times.append(utc_to_greg([ts.time(i) for i in range(ts.size())]))
            values.append(np.array(ts.v))
        else:
            times.append(utc_to_greg(ts[0]))
            values.append(ts[1])
        if t_start is None or t_start > times[-1][0]:
            t_start = times[-1][0]
        if t_stop is None or t_stop < times[-1][-1]:
            t_stop = times[-1][-1]

    for i, (t, v) in enumerate(zip(times, values)):
        if i > 0:
            plt.subplot(n_figs, 1, i + 1, sharex=ax)
        plt.hold(1)
        if use_week_marks:
            mark_weeks(calendar, t_start, t_stop)
        plt.plot(t, v)
        plt.gca().grid(b=True, color=(51/256, 102/256, 193/256), linewidth=0.1, linestyle='-', axis='y')
        if labels is not None:
            plt.ylabel(labels[i])
    ax.set_xlim(t_start, t_stop)
    set_calendar_formatter(calendar)


def mark_weeks(cal, t_start, t_stop):
    tick_start = int(greg_to_utc(t_start))
    tick_stop = int(greg_to_utc(t_stop))
    week_starts = [cal.trim(tick_start, cal.WEEK)]
    ax = plt.gca()
    while week_starts[-1] < tick_stop:
        week_starts.append(cal.trim(week_starts[-1] + cal.WEEK, cal.WEEK))
    for i in range(0, len(week_starts) - 1, 2):
        ax.axvspan(utc_to_greg(week_starts[i]), utc_to_greg(week_starts[i+1]), color=[0.7, 0.7, 0.7, 0.4])




def set_calendar_formatter(cal, str_format="{year:04d}.{month:02d}.{day:02d}", format_major=True):
    fields = {"year": None,
              "month": None,
              "day": None,
              "hour": None,
              "minute": None,
              "second": None}
    ax = plt.gca()
    fig = plt.gcf()

    def format_date(x, pos=None):
        t_utc = cal.trim(int(round(greg_to_utc(x))), deltahours(1))
        ymd = cal.calendar_units(t_utc)
        for f in fields:
            fields[f] = getattr(ymd, f)
        return str_format.format(**fields)
    if format_major:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        plt.setp(ax.get_xminorticklabels(), rotation=45, horizontalalignment='right')
    else:
        ax.xaxis.set_minor_formatter(ticker.FuncFormatter(format_date))
        plt.setp(ax.get_xmajorticklabels(), rotation=45, horizontalalignment='right')
    fig.autofmt_xdate()


def greg_to_utc(t):
    a = 3600*24.0
    b = 719164.0
    return (np.asarray(t) - b)*a


def utc_to_greg(t):
    a = 3600*24.0
    b = 719164.0
    return np.asarray(t)/a + b
