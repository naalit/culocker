import pandas as pd
import plotly.express as px

#df = pd.read_csv("out/cu_log_T2_L2.csv")
df = pd.read_csv("out/cu_log_task_T2_L2.csv")

#fig = px.line(df, x='frame', y='time', color='window_size', title='Frame times')
#fig = px.box(df, x='window_size', y='time', title='Frame times', log_x=True, labels={'window_size': 'Critical section length (ms)', 'time': 'Frame time (s)'})
fig = px.box(df, x='window_size', y='time', title='Response times', log_x=True, labels={'window_size': 'Critical section length (ms)', 'time': 'Response time (ms)'})
fig.show()

print('95%:\n', df.groupby('window_size')['time'].apply(lambda x: x.quantile(0.95)), sep='')
print('99%:\n', df.groupby('window_size')['time'].apply(lambda x: x.quantile(0.99)), sep='')
print('99.5%:\n', df.groupby('window_size')['time'].apply(lambda x: x.quantile(0.995)), sep='')
