import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Build combined dataframe for both tasks
df_long = pd.read_csv("out/cu_log_T3_L3.csv")
df_short = pd.read_csv("out/cu_log_task_T3_L3.csv")
df_long['task'] = 'long'
df_short['task'] = 'short'
df_long['time'] = df_long['time'].map(lambda x: x * 10) # make the scales more comparable. still not ideal since the scales are different (ms vs ds) but we'll figure something out
df = pd.concat([df_short, df_long])

# Plot it
plt.figure(figsize=(12, 6))
sns.boxplot(x='window_size', y='time', hue='task', data=df)
plt.title('Time by Window Size and Task')
plt.xlabel('Window Size')
plt.ylabel('Time')
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
