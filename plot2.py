import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

# Configuration (via CLI params!)
args = {}
pending = None
for i in sys.argv[1:]:
    if pending is not None:
        args[pending] = i
        pending = None
    elif i.startswith('--'):
        pending = i[2:]
    elif 'plot' not in args:
        args['plot'] = i
    elif 'data' not in args:
        args['data'] = i
    else:
        print(f"Error: unknown argument '{i}'")
print('Config:', args)
def cbool(s): # bool('False') returns True 😔
    return s in [True, 'True', 'true', '1', 1]
invalid = args.copy()
def check(argname, default, options=None):
    arg = default
    if argname in args:
        arg = invalid.pop(argname)
    if options is not None:
        if arg != default and arg not in options:
            print(f"Warning: invalid value '{arg}' for argument '{argname}'; valid values are {options}")
    return arg

plot = check('plot', 'cdf', ['box', 'tables'])
dataset = check('data', 'all')
fontsize = int(check('fontsize', 20))
legend = check('legend', 'top', ['middle', 'none'])
selected_tasks = check('tasks', 'short' if dataset == 'control' else 'all', ['short', '1', 'long', '2', 'all'])
fill_screen = check('fill-screen', False, ['true', 'false', 'half'])
suffix = check('file-suffix', '_T3_L6')
title = check('title', 'Task Response Times')

for k, v in invalid.items():
    print(f"Warning: invalid argument '{k}'")

# Build combined dataframe for both tasks
df_long = pd.read_csv(f"out/cu_log{suffix}.csv")
df_short = pd.read_csv(f"out/cu_log_task{suffix}.csv")
df_long['task'] = 'long'
df_short['task'] = 'short'
df_long['time'] = df_long['time'].map(lambda x: x * 10) # make the scales more comparable. still not ideal since the scales are different (ms vs ds) but we'll figure something out
df = pd.concat([df_short, df_long])

s_labels_n = { 2: 'Only task 1', 3: 'No locking', 4: 'Locking every GPU call' }

plt.rcParams.update({'font.size': fontsize})

match selected_tasks:
    case 'short' | '1':
        df = df_short
    case 'long' | '2':
        df = df_long
    # default is 'all'

if dataset == 'control':
    df = df[(df['window_size'] == 2) | (df['window_size'] == 3)]
elif dataset != 'all':
    # hopefully dataset is a comma-separated list of window sizes
    sizes = list(map(lambda x: float(x), dataset.split(',')))
    print('using custom dataset', sizes)
    df = df[df['window_size'].apply(lambda x: float(x) in sizes)]
    print('window sizes:', df['window_size'].unique())

# if selected_tasks in ['1', 'short']:
#     s_labels_n = { 2: 'Running alone', 3: 'Running alongside task 2',  4: 'Locking every GPU call' }
# if selected_tasks in ['2', 'long']:
#     s_labels_n = { 2: 'Running alone', 3: 'Running alongside task 1',  4: 'Locking every GPU call' }
s_labels = { k: v.replace('\n', ' ') for k, v in s_labels_n.items() }

if plot == 'box':
    # Box plot
    plt.figure(figsize=(12, 6))

    ax_l = plt.gca()
    ax_r = ax_l.twinx()
    plt.title('Time by Window Size and Task')

    sns.boxplot(ax=ax_l, x='window_size', y='time', hue='task', data=df)

    palette = sns.color_palette()

    ax_l.yaxis.label.set_color(palette[0])
    ax_l.yaxis.label.set_weight('bold')
    ax_l.tick_params(axis='y', colors=palette[0])

    ax_r.yaxis.label.set_color(palette[1])
    ax_r.yaxis.label.set_weight('bold')
    ax_r.tick_params(axis='y', colors=palette[1])

    ax_l.set_ylabel('Time for Short (ms)')
    ax_r.set_ylabel('Time for Long (s)')

    # set ylim to 99.95th percentile to avoid the most egregious outliers, so we can actually see the bars
    lo = 0
    hi = df['time'].quantile(0.9995)
    ax_l.set_ylim(lo, hi)
    ax_r.set_ylim(lo/10, hi/10)

    # this really does seem like the best way to change specific tick labels on a categorical axis
    ax_l.xaxis.get_major_formatter()._units = { (s_labels_n[float(k)] if float(k) in s_labels_n else k): v for k, v in ax_l.xaxis.get_major_locator()._units.items() }

    plt.xlabel('Window Size')
    plt.show()

if plot == 'cdf':
    # CDF plot
    tasks = df['task'].unique()
    window_sizes = sorted(df['window_size'].unique())
    n_controls = 1 if selected_tasks in ['long', '2'] else 2
    # use a perceptually uniform sequential color map (plasma) for the actual window sizes, and greens for the controls
    colors = np.concatenate([plt.cm.plasma(np.linspace(0, 1, max(len(window_sizes)-(n_controls+1), 0))), np.flip(plt.cm.Greens(np.linspace(0, 1, 6)), 0)[5 - 2*n_controls:5:2,:], plt.cm.spring(np.linspace(0, 1, 1))], 0)

    # if fill-screen then make it 1920x1080 for easy transferring to slides
    if fill_screen == 'half':
        kwargs = { 'figsize': (14.00, 10.80), 'dpi': 100.0 }
    elif cbool(fill_screen):
        kwargs = { 'figsize': (19.20, 10.80), 'dpi': 100.0 }
    else:
        kwargs = { 'figsize': (12, 4*len(tasks)) }
    fig, axs = plt.subplots(1, len(tasks), **kwargs)
    if len(tasks) == 1:
        axs = [axs] # this is dumb
    fig.suptitle(title)

    for i, task in enumerate(tasks):
        for window_size, color in zip(window_sizes, colors):
            task_data = df[(df['task'] == task) & (df['window_size'] == window_size)]['time']
            if len(task_data) > 0:
                x = np.sort(task_data)
                y = np.arange(1, len(x) + 1) / len(x)
                label = s_labels[window_size] if window_size in s_labels else f'{window_size}ms'
                axs[i].plot(x, y, color=color, label=label)

        if len(tasks) > 1:
            axs[i].set_ylabel(f'Task {task}')
        axs[i].grid(True)

    if legend == 'top':
        axs[0].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
             mode="expand", borderaxespad=0, ncols=4, markerscale=0.05)
    elif legend != 'none':
        axs[0].legend()
    units = { 'short': 'ms', 'long': 's' }
    lim = { 'short': 10, 'long': 7 }
    print(tasks, lim[tasks[0]])
    axs[0].set_xlabel(f'Time ({units[tasks[0]]})')
    axs[0].set_xlim(0, lim[tasks[0]]) # we don't super care about the very small number of outliers at the top
    axs[0].set_xticks(list(range(lim[tasks[0]] + 1))) # and make sure there are ticks at each millisecond/second
    if len(axs) > 1:
        axs[i].set_xlabel(f'Time ({units[tasks[1]]})')
        axs[1].set_xlim(0, lim[tasks[1]]) # make sure they both start at 0
        axs[1].xaxis.set_major_formatter(lambda x, pos: str(x / 10)) # the times in the dataframe are in ds, so convert to s for easier reading

    fig.text(0.01, 0.5, 'Cumulative Probability', va='center', rotation='vertical')
    plt.tight_layout()
    plt.subplots_adjust(left=0.075)
    plt.show()

if plot == 'tables':
    # Calculate quantiles
    print('short (ms):')
    print('90%:\n', df_short.groupby('window_size')['time'].apply(lambda x: x.quantile(0.90)), sep='')
    print('99%:\n', df_short.groupby('window_size')['time'].apply(lambda x: x.quantile(0.99)), sep='')
    print('99.5%:\n', df_short.groupby('window_size')['time'].apply(lambda x: x.quantile(0.995)), sep='')
    print('long (ds):')
    print('90%:\n', df_long.groupby('window_size')['time'].apply(lambda x: x.quantile(0.90)), sep='')
    print('99%:\n', df_long.groupby('window_size')['time'].apply(lambda x: x.quantile(0.99)), sep='')
    print('99.5%:\n', df_long.groupby('window_size')['time'].apply(lambda x: x.quantile(0.995)), sep='')

    print('\nshort percent above 5ms:', df_short.groupby('window_size')['time'].apply(lambda x: f'{len(x[x > 5])/len(x)*100 :.2f}%' ), sep='\n')
    print('\nshort percent above 6ms:', df_short.groupby('window_size')['time'].apply(lambda x: f'{len(x[x > 6])/len(x)*100 :.2f}%' ), sep='\n')
