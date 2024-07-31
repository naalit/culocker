import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Build combined dataframe for both tasks
df_long = pd.read_csv("out/cu_log_T3_L3.csv")
df_short = pd.read_csv("out/cu_log_task_T3_L3.csv")
df_long['task'] = 'long'
df_short['task'] = 'short'
df_long['time'] = df_long['time'].map(lambda x: x * 10) # make the scales more comparable. still not ideal since the scales are different (ms vs ds) but we'll figure something out
df = pd.concat([df_short, df_long])

# Box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='window_size', y='time', hue='task', data=df)
plt.title('Time by Window Size and Task')
plt.xlabel('Window Size')
plt.ylabel('Time')
plt.show()

# CDF plot
tasks = ['short', 'long']
window_sizes = sorted(df['window_size'].unique())
colors = plt.cm.rainbow(np.linspace(0, 1, len(window_sizes)))

fig, axs = plt.subplots(len(tasks), 1, figsize=(12, 4*len(tasks)), sharex=True)
fig.suptitle('Cumulative Distribution Function of Time by Task and Window Size')

for i, task in enumerate(tasks):
    for window_size, color in zip(window_sizes, colors):
        task_data = df[(df['task'] == task) & (df['window_size'] == window_size)]['time']
        if len(task_data) > 0:
            x = np.sort(task_data)
            y = np.arange(1, len(x) + 1) / len(x)
            axs[i].plot(x, y, color=color, label=f'Window Size {window_size}')
    
    axs[i].set_ylabel(f'Task {task}')
    axs[i].grid(True)
    axs[i].legend()

plt.xlabel('Time')
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
