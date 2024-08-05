import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Build combined dataframe for both tasks
suffix = '_T3_L5'
df_long = pd.read_csv(f"out/cu_log{suffix}.csv")
df_short = pd.read_csv(f"out/cu_log_task{suffix}.csv")
df_long['task'] = 'long'
df_short['task'] = 'short'
df_long['time'] = df_long['time'].map(lambda x: x * 10) # make the scales more comparable. still not ideal since the scales are different (ms vs ds) but we'll figure something out
df = pd.concat([df_short, df_long])

s_labels = { 2: 'Control A (no concurrent long task)', 2.5: 'Control A (no concurrent long task)', 3: 'Control B (both tasks, no locking)' }
s_labels2 = { 2: 'Control A\n(no concurrent long task)', 2.5: 'Control A\n(no concurrent long task)', 3: 'Control B\n(both tasks, no locking)' }

# Box plot
plt.figure(figsize=(12, 6))

ax_l = plt.gca()
ax_r = ax_l.twinx()
plt.title('Time by Window Size and Task')

sns.boxplot(ax=ax_l, x='window_size', y='time', hue='task', data=df)

palette = sns.color_palette()

# this is not amazing code and it gives a warning. but it does work
ax_l.yaxis.label.set_color(palette[0])
ax_l.yaxis.label.set_weight('bold')
ax_l.tick_params(axis='y', colors=palette[0])

ax_r.yaxis.label.set_color(palette[1])
ax_r.yaxis.label.set_weight('bold')
ax_r.tick_params(axis='y', colors=palette[1])

ax_l.set_ylabel('Time for Short (ms)')
ax_r.set_ylabel('Time for Long (s)')

lo = 0
hi = df['time'].quantile(0.9995)
#lo, hi = ax_l.get_ylim()
ax_l.set_ylim(lo, hi)
ax_r.set_ylim(lo/10, hi/10)

# This gives a warning, but it does work to label the controls 
labels = []
for tick in ax_l.xaxis.get_major_ticks():
    val = float(tick.label1.get_text())
    labels.append(s_labels2[val] if val in s_labels2 else tick.label1.get_text())
ax_l.set_xticklabels(labels)

plt.xlabel('Window Size')
plt.show()


# CDF plot
tasks = ['short', 'long']
window_sizes = sorted(df['window_size'].unique())
colors = plt.cm.plasma(np.linspace(0, 1, len(window_sizes)))

fig, axs = plt.subplots(len(tasks), 1, figsize=(12, 4*len(tasks)))
fig.suptitle('Cumulative Distribution Function of Time by Task and Window Size')

for i, task in enumerate(tasks):
    for window_size, color in zip(window_sizes, colors):
        task_data = df[(df['task'] == task) & (df['window_size'] == window_size)]['time']
        if len(task_data) > 0:
            x = np.sort(task_data)
            y = np.arange(1, len(x) + 1) / len(x)
            label = s_labels[window_size] if window_size in s_labels else f'Window Size {window_size}ms'
            axs[i].plot(x, y, color=color, label=label)
    
    axs[i].set_ylabel(f'Task {task}')
    axs[i].grid(True)
    axs[i].legend()

axs[0].set_xlabel('Time (ms)')
axs[0].set_xlim(0, 10) # we don't super care about the very small number of outliers at the top
axs[0].set_xticks(list(range(11))) # and make sure there are ticks at each millisecond
axs[1].set_xlabel('Time (s)')
axs[1].set_xlim(0) # make sure they both start at 0
axs[1].xaxis.set_major_formatter(lambda x, pos: str(x / 10)) # the times in the dataframe are in ds, so convert to s for easier reading

fig.text(0.01, 0.5, 'Cumulative Probability', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

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
