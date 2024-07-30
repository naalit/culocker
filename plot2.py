import pandas as pd
import plotly.express as px

df_long = pd.read_csv("out/cu_log_T3_L2.csv")
df_short = pd.read_csv("out/cu_log_task_T3_L2.csv")

df_long['task'] = 'long'
df_short['task'] = 'short'
df_long['time'] = df_long['time'].map(lambda x: x * 10) # make the scales more comparable. still not ideal since the scales are different (ms vs ds) but we'll figure something out
df = pd.concat([df_short, df_long])

#fig = px.line(df, x='frame', y='time', color='window_size', title='Frame times')
#fig = px.box(df, x='window_size', y='time', title='Frame times', log_x=True, labels={'window_size': 'Critical section length (ms)', 'time': 'Frame time (s)'})
fig = px.box(df, x='window_size', y='time', color='task', title='Response times', log_x=True, labels={'window_size': 'Critical section length (ms)', 'time': 'Response time (ms/ds)'})
fig.show()

print('short (ms):')
print('90%:\n', df_short.groupby('window_size')['time'].apply(lambda x: x.quantile(0.90)), sep='')
print('99%:\n', df_short.groupby('window_size')['time'].apply(lambda x: x.quantile(0.99)), sep='')
print('99.5%:\n', df_short.groupby('window_size')['time'].apply(lambda x: x.quantile(0.995)), sep='')
print('long (ds):')
print('90%:\n', df_long.groupby('window_size')['time'].apply(lambda x: x.quantile(0.90)), sep='')
print('99%:\n', df_long.groupby('window_size')['time'].apply(lambda x: x.quantile(0.99)), sep='')
print('99.5%:\n', df_long.groupby('window_size')['time'].apply(lambda x: x.quantile(0.995)), sep='')
