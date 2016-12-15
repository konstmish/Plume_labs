import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('challenge_output_data_training_file_predict_air_quality_at_the_street_level.csv')
X_test  = pd.read_csv('X_test.csv')

X_train = X_train.drop('ID', 1)
X_test  = X_test.drop('ID', 1)
y_train = y_train['TARGET']
X_train.describe()

columns = list(X_train.columns.values)
for column in columns:
    if (X_train[column].isnull().values.any()):
        print(column)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0);

np.unique(X_train['pollutant'])

X_train = pd.concat([X_train, pd.get_dummies(X_train['station_id'], prefix='station_id')], axis=1)
X_train = X_train.drop('station_id', 1)
X_train_no2  = X_train.ix[X_train['pollutant'] == 'NO2', :]
y_train_no2  = y_train.ix[X_train['pollutant'] == 'NO2']
X_train_pm10 = X_train.ix[X_train['pollutant'] == 'PM10', :]
y_train_pm10 = y_train.ix[X_train['pollutant'] == 'PM10']
X_train_pm25 = X_train.ix[X_train['pollutant'] == 'PM2_5', :]
y_train_pm25 = y_train.ix[X_train['pollutant'] == 'PM2_5']

X_train_no2  = X_train_no2.drop('pollutant', 1)
X_train_pm10 = X_train_pm10.drop('pollutant', 1)
X_train_pm25 = X_train_pm25.drop('pollutant', 1)

X_test = pd.concat([X_test, pd.get_dummies(X_test['station_id'], prefix='station_id')], axis=1)
X_test = X_test.drop('station_id', 1)
X_test_no2  = X_test.ix[X_test['pollutant'] == 'NO2', :]
X_test_pm10 = X_test.ix[X_test['pollutant'] == 'PM10', :]
X_test_pm25 = X_test.ix[X_test['pollutant'] == 'PM2_5', :]

X_train_no2.describe()

print(all((X_train_no2['station_id_16.0'] == 1)[0:3000]))
time = X_train_no2['daytime']
data = [go.Scatter(x = time[0:300], y = y_train_no2[0:300])]
layout = go.Layout(
    title = 'Station 16, NO2'
)
fig = go.Figure(data = data, layout = layout)
iplot(fig)

time = X_train_no2['daytime'].ix[X_train_no2['station_id_26.0'] == 1]
trace1 = go.Scatter(x=time[0:300], y =y_train_no2.ix[X_train_no2['station_id_26.0'] == 1][0:300], name ='NO2')

time = X_train_pm10['daytime'].ix[X_train_pm10['station_id_26.0'] == 1]
trace2 = go.Scatter(x=time[0:300], y = 2 * y_train_pm10.ix[X_train_pm10['station_id_26.0'] == 1][0:300], name ='2 x PM10')

time = X_train_pm25['daytime'].ix[X_train_pm25['station_id_26.0'] == 1]
trace3 = go.Scatter(x=time[0:300], y = 4 * y_train_pm25.ix[X_train_pm25['station_id_26.0'] == 1][0:300], name ='4 x PM25')

data = [trace1, trace2, trace3]
layout = go.Layout(
    title = 'Station 26, rescaled polution levels, hourly'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

time = X_train_no2['daytime'].ix[X_train_no2['station_id_26.0'] == 1]
y_station4_no2  = y_train_no2.ix[X_train_no2['station_id_26.0'] == 1]
trace1 = go.Scatter(x=time[0:3000:24], y=[np.mean(y_station4_no2[i:i+24]) for i in range(0, int(3000/24))], name = 'NO2')

time = X_train_pm10['daytime'].ix[X_train_pm10['station_id_26.0'] == 1]
y_station4_pm10 = y_train_pm10.ix[X_train_pm10['station_id_26.0'] == 1]
trace2 = go.Scatter(x = time[0:3000:24], y = [2 * np.mean(y_station4_pm10[i:i+24]) for i in range(0, int(3000/24))], name = '2 x PM10')

time = X_train_pm25['daytime'].ix[X_train_pm25['station_id_26.0'] == 1]
y_station4_pm25 = y_train_pm25.ix[X_train_pm25['station_id_26.0'] == 1]
trace3 = go.Scatter(x = time[0:3000:24], y = [4 * np.mean(y_station4_pm25[i:i+24]) for i in range(0, int(3000/24))], name = '4 x PM25')

data = [trace1, trace2, trace3]
layout = go.Layout(
    title = 'Station 26, rescaled polution levels, daily'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

time = X_train_no2['daytime'].ix[X_train_no2['station_id_11.0'] == 1]
y_station4_no2  = y_train_no2.ix[X_train_no2['station_id_11.0'] == 1]
trace1 = go.Scatter(x=time[0:3000:24], y =y_train_no2.ix[X_train_no2['station_id_11.0'] == 1][0:300], name ='Station 11')

time = X_train_no2['daytime'].ix[X_train_no2['station_id_16.0'] == 1]
y_station4_no2  = y_train_no2.ix[X_train_no2['station_id_16.0'] == 1]
trace2 = go.Scatter(x=time[0:3000:24], y =y_train_no2.ix[X_train_no2['station_id_16.0'] == 1][0:300], name ='Station 16')

data = [trace1, trace2]
layout = go.Layout(
    title = 'NO2, hourly'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

time = X_train_no2['daytime'].ix[X_train_no2['station_id_11.0'] == 1]
y_station4_no2  = y_train_no2.ix[X_train_no2['station_id_11.0'] == 1]
trace1 = go.Scatter(x = time[0:3000:24], y = [np.mean(y_station4_no2[i:i+24]) for i in range(0, int(3000/24))], name = 'Station 11')

time = X_train_no2['daytime'].ix[X_train_no2['station_id_16.0'] == 1]
y_station4_no2  = y_train_no2.ix[X_train_no2['station_id_16.0'] == 1]
trace2 = go.Scatter(x = time[0:3000:24], y = [np.mean(y_station4_no2[i:i+24]) for i in range(0, int(3000/24))], name = 'Station 16')

data = [trace1, trace2]
layout = go.Layout(
    title = 'NO2, daily'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

from sklearn.linear_model import LinearRegression
lin_reg_no2 = LinearRegression()
lin_reg_no2.fit(X_train_no2, y_train_no2)

lin_reg_pm10 = LinearRegression()
lin_reg_pm10.fit(X_train_pm10, y_train_pm10)

lin_reg_pm25 = LinearRegression()
lin_reg_pm25.fit(X_train_pm25, y_train_pm25);

pollut = X_test['pollutant']
X_test = X_test.drop('pollutant', 1)
y_no2 = lin_reg_no2.predict(X_test)
y_pm10 = lin_reg_pm10.predict(X_test)
y_pm25 = lin_reg_pm25.predict(X_test)

y_pred = y_no2 * (pollut == 'NO2') + y_pm10 * (pollut == 'PM10') + y_pm10 * (pollut == 'PM2_5')

df = pd.DataFrame
df['TARGET'] = y_pred
df.to_csv('y_pred.csv', index = False)