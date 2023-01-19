

#import 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA


data = pd.read_csv("H:\My_app\Gold_data.csv",parse_dates=True, index_col="date")
data.skew()

#corelation
data.corr()

sns.kdeplot(data['price'])

data.plot(figsize = (15,8),color = 'red')

data.plot(xlim = ['2019-01-01','2021-12-01'],ylim = [2700,5000],figsize = (15,8),ls='--',c='green')

data.resample(rule='A').mean().plot()
data.resample(rule='A').max().plot()
data.resample(rule ='A').min().plot()
print(data.resample(rule='A').mean())
print(data.resample(rule='A').max())
print(data.resample(rule ='A').min())
#checking the mean max and minimam price in yearwise


# 1so here we can see in the price mean is not much deviated from past year price we can see the deviation only is 1K for 3 year but from 2019 the price went to its all time high at 2020 2.and the min and max value show us that the maximam yaer has gold price treded 7K from its all time low 3 the graph show us that the people who have perchesed gold at the all time hight those only need to worry but the people who have buys on its avg price or min price they still in profit

# In[ ]:


data_temp = data.copy()
data_temp['Year'] = pd.DatetimeIndex(data_temp.index).year
data_temp['quarter'] = pd.DatetimeIndex(data_temp.index).quarter
data_temp['Month'] = pd.DatetimeIndex(data_temp.index).month
data_temp['Weeks'] = pd.DatetimeIndex(data_temp.index).week


# In[ ]:


plt.figure(figsize=(15,6))
plt.title('Seasonality of the Time Series by over the year')
sns.boxplot(x='Year',y='price',hue='Year',data=data_temp)


# in the year of 2020 we got all time high price and 2016 and 2021 we have some outlier

# In[ ]:


plt.figure(figsize=(12,8))
plt.title('Seasonality of the Time Series quarter wise')
heatmap=pd.pivot_table(data=data_temp,values='price',index=data_temp['quarter'],columns='Year',aggfunc='max',fill_value=0)
sns.heatmap(heatmap,annot = True ,fmt='g')


# so here we got in the year of 2020 quarter 2 we got all time high and quarter 2 3 and 4 we can see in the every year of quarter the price in pickup

# In[ ]:


plt.figure(figsize=(20,10))
plt.title('Seasonality of the Time Series month wise')
sns.barplot(x='Month',y='price',hue='Year',data=data_temp)


# here we got in the every month of aug the price went high and in jan the price is all time low

# In[ ]:


plt.figure(figsize=(20,10))
plt.title('Seasonality of the Time Series week wise')
sns.pointplot(x='Weeks',y='price',hue='Year',data=data_temp)


# so here we observe 3 major things 1 we can see the week wise fluctuation is very high compare to month and quarterly wise 
# 2 here we see for the most of year the week of 34 the price went hight for so week 34 is very importnat for year
# 3 we can see in the year of 2016 2017 2018 2019 the price is in between 2500 to 3500 but in last two year the price went high more then 10K just in one year so there is something external fector like (covid)

# ## checking the seasnoality

# In[ ]:


#Additive Decomposition
decomposition = seasonal_decompose(data, model='additive')
print(decomposition.trend)
print(decomposition.seasonal)
print(decomposition.resid)
print(decomposition.observed)
fig = decomposition.plot()
plt.rcParams['figure.figsize'] = (25, 8)


# In[ ]:


#Additive Decomposition
decomposition1 = seasonal_decompose(data.iloc[1:1461], model='additive')
print(decomposition1.trend)
print(decomposition1.seasonal)
print(decomposition1.resid)
print(decomposition1.observed)
fig = decomposition1.plot()
plt.rcParams['figure.figsize'] = (25, 8)


# In[ ]:


decomposition2 = seasonal_decompose(data.iloc[1:731], model='additive')
print(decomposition2.trend)
print(decomposition2.seasonal)
print(decomposition2.resid)
print(decomposition2.observed)
fig = decomposition2.plot()
plt.rcParams['figure.figsize'] = (25, 8)


# In[ ]:


decomposition3 = seasonal_decompose(data.iloc[1:366], model='additive')
print(decomposition3.trend)
print(decomposition3.seasonal)
print(decomposition3.resid)
print(decomposition3.observed)
fig = decomposition3.plot()
plt.rcParams['figure.figsize'] = (25, 8)


# In[ ]:


decomposition4 = seasonal_decompose(data.iloc[1:31], model='additive')
print(decomposition4.trend)
print(decomposition4.seasonal)
print(decomposition4.resid)
print(decomposition4.observed)
fig = decomposition4.plot()
plt.rcParams['figure.figsize'] = (25, 8)


# In[ ]:


decomposition5 = seasonal_decompose(data.iloc[1:366], model='multiplicative')
print(decomposition5.trend)
print(decomposition5.seasonal)
print(decomposition5.resid)
print(decomposition5.observed)
fig = decomposition5.plot()
plt.rcParams['figure.figsize'] = (25, 8)


# In[ ]:


decomposition6 = seasonal_decompose(data.iloc[1:31], model='multiplicative')
print(decomposition6.trend)
print(decomposition6.seasonal)
print(decomposition6.resid)
print(decomposition6.observed)
fig = decomposition6.plot()
plt.rcParams['figure.figsize'] = (25, 8)


# #so from the graph we got to know that our data has weekly seasonality and our data is multiplicative series

# ## checking for stationrity

# In[ ]:


#ADFuller Test for stationarity
adf = adfuller(data["price"])[1]
print(f"p value:{adf.round(4)}", ", Series is Stationary" if adf <0.05 else ", Series is Non-Stationary")


# # Differencing

# In[ ]:


#differencing to make data into stationarity
de_trended = data.diff(1).dropna()
adf2 = adfuller(de_trended)[1]
print(f"p value:{adf2}", ", Series is Stationary" if adf2 <0.05 else ", Series is Non-Stationary")
de_trended.plot()


# # ACF plot

# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data,lags = 100)
plt.figure(figsize = (40,10))
plt.show()


# is graph show as that our data is highly coreleted but for predition we need non coreleted and this is non stationarity data acf plot

# In[ ]:


#ACF Plot after Differencing
plot_acf(de_trended,lags = 100)
plt.figure(figsize = (20,10))
plt.show()


# from here we can decied q value is 0 , 2 and 6 or 7  

# # PACF Plots

# In[ ]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data,lags = 100)
plt.figure(figsize = (20,10))
plt.show()


# is graph show as that our data is coreleted but for predition we need non coreleted and this is non stationarity data pacf plot

# In[ ]:


#PACF Plot after Differencing
plot_pacf(de_trended,lags = 100)
plt.figure(figsize = (20,10))
plt.show()


# here we can see clearly the pattern for acf so from here we got p value form 0 ,2 ,4 ,6 7

# ## Splitting the data into train and test

# In[ ]:


data_train = data.iloc[0:1530]
data_test = data.iloc[1530:]


# In[ ]:


data_train.shape
data_train.head


# In[ ]:


data_test.shape
data_test.head


# # ARIMA

# In[ ]:


# 1,1,1 ARIMA Model
model = ARIMA(data_train, order=(2,1,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[ ]:


# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1],figsize=(20,10))
plt.show()
print(model_fit.resid)


# In[ ]:


residuals.describe()


# In[ ]:


# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
figsize = (20,10)
plt.show()


# In[ ]:


fc, se, conf = model_fit.forecast(652, alpha=0.05)  # 95% conf
fc, se, conf


# In[ ]:


# Make as pandas series
fc_series = pd.Series(fc, index=data_test.index)
lower_series = pd.Series(conf[:, 0], index=data_test.index)
upper_series = pd.Series(conf[:, 1], index=data_test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(data_train, label='training')
plt.plot(data_test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
print(np.sqrt(metrics.mean_squared_error(data_test,fc_series)))
print(mean_absolute_percentage_error(data_test,fc_series)*100)


# # GRID search

# In[ ]:


import itertools
#set parameter range
p = range(0,8)
q = range(0,6)
d = range(0,3)
# list of all parameter combos
pdq = list(itertools.product(p, d, q))
# SARIMA model pipeline
for param in pdq:
        try:
            mod = ARIMA(data_train,order=param)
            results = mod.fit(max_iter = 70)
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
        except:
            continue


# so from here we got 7 1 4 is the best pdq value as per lowest AIC

# In[ ]:


get_ipython().system('pip install pmdarima')


# In[ ]:


import pmdarima as pm
model = pm.auto_arima(data_train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=7, max_q=5, # maximum p and q
                      m=0,              # frequency of series
                      d=1,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())


# from this we got 1.0.1 is best model but AIC is 18031 and from grid we got 17871 so we ll go with grid

# In[ ]:


model_1 = ARIMA(data_train, order=(7,1,4))
model_1_fit = model_1.fit(disp=0)
print(model_1_fit.summary())


# In[ ]:


# Plot residual errors
residuals1 = pd.DataFrame(model_1_fit.resid)
fig, ax = plt.subplots(1,2)
residuals1.plot(title="Residuals", ax=ax[0])
residuals1.plot(kind='kde', title='Density', ax=ax[1],figsize=(20,10))
plt.show()
print(model_1_fit.resid)


# In[ ]:


residuals1.describe()


# In[ ]:


# Actual vs Fitted
model_1_fit.plot_predict(dynamic=False)
figsize = (20,10)
plt.show()


# In[ ]:


# Forecast
fc, se, conf = model_1_fit.forecast(652, alpha=0.05)  # 95% conf
fc, se, conf 


# In[ ]:


# Make as pandas series
fc_series1 = pd.Series(fc, index=data_test.index)
lower_series = pd.Series(conf[:, 0], index=data_test.index)
upper_series = pd.Series(conf[:, 1], index=data_test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(data_train, label='training')
plt.plot(data_test, label='actual')
plt.plot(fc_series1, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
print(np.sqrt(metrics.mean_squared_error(data_test,fc_series1)))
print(mean_absolute_percentage_error(data_test,fc_series1)*100)


# # SARIMA

# In[ ]:


import statsmodels.api as sm
#mod1 = sm.tsa.statespace.SARIMAX(data_train,order = (1,1,1),seasonal_order = (1,1,1,4))
mod1 = sm.tsa.statespace.SARIMAX(data_train,order = (2,1,2),seasonal_order = (2,1,2,12))
#mod1 = sm.tsa.statespace.SARIMAX(data_train,order = (1,1,1),seasonal_order = (1,1,1,52))
result = mod1.fit()
print(result.summary())


# In[ ]:


residuals2 = pd.DataFrame(result.resid)
fig, ax = plt.subplots(1,2)
residuals2.plot(title="Residuals", ax=ax[0])
residuals2.plot(kind='kde', title='Density', ax=ax[1],figsize=(20,10))
plt.show()
print(result.resid)


# In[ ]:


residuals2.describe()


# In[ ]:


# Actual vs Fitted
forecast = result.predict(dynamic=False)
pd.concat([data_train,forecast],axis=1).plot()
figsize = (20,10)
plt.show()


# In[ ]:


# Forecast
fc = result.forecast(652, alpha=0.05)  # 95% conf
fc


# In[ ]:


fc_series2 = pd.Series(fc, index=data_test.index)
# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(data_train, label='training')
plt.plot(data_test, label='actual')
plt.plot(fc_series2, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
print(np.sqrt(metrics.mean_squared_error(data_test,fc_series2)))
print(mean_absolute_percentage_error(data_test,fc_series2)*100)


# # GRID for SRIMA

# In[ ]:


import itertools
p = range(2, 8)
d = range(1,2)
q = range(4, 6)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data_train,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[ ]:


from pmdarima import auto_arima
import pmdarima as pm
model_s = pm.auto_arima(data_train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=7, max_q=5, # maximum p and q
                      m=12,              # frequency of series
                      d=1,           # let model determine 'd'
                      seasonal=True,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)


# In[ ]:



#mod2 = sm.tsa.statespace.SARIMAX(data_train,order = (6,1,6),seasonal_order = (6,1,6,4))
mod2 = sm.tsa.statespace.SARIMAX(data_train,order = (6,1,6),seasonal_order = (6,1,6,12))
#mod2 = sm.tsa.statespace.SARIMAX(data_train,order = (6,1,6),seasonal_order = (6,1,6,52))
result_p = mod2.fit()
print(result_p.summary())


# In[ ]:



#mod3 = sm.tsa.statespace.SARIMAX(data_train,order = (7,1,7),seasonal_order = (7,1,7,4))
mod3 = sm.tsa.statespace.SARIMAX(data_train,order = (7,1,7),seasonal_order = (7,1,7,12))
#mod3 = sm.tsa.statespace.SARIMAX(data_train,order = (7,1,7),seasonal_order = (7,1,7,52))
result_r = mod3.fit()
print(result_r.summary())


# In[ ]:



mod4 = sm.tsa.statespace.SARIMAX(data_train,order = (0,1,0),seasonal_order = (0,1,0,12))
result_q = mod4.fit()
print(result_q.summary())


# In[ ]:


residuals3 = pd.DataFrame(result_r.resid)
fig, ax = plt.subplots(1,2)
residuals3.plot(title="Residuals", ax=ax[0])
residuals3.plot(kind='kde', title='Density', ax=ax[1],figsize=(20,10))
plt.show()
print(result_r.resid)


# In[ ]:


residuals4 = pd.DataFrame(result_q.resid)
fig, ax = plt.subplots(1,2)
residuals4.plot(title="Residuals", ax=ax[0])
residuals4.plot(kind='kde', title='Density', ax=ax[1],figsize=(20,10))
plt.show()
print(result_q.resid)


# In[ ]:


residuals3.describe()


# In[ ]:


residuals4.describe()


# In[ ]:


# Actual vs Fitted
forecast_2 = result_r.predict(dynamic=False)
pd.concat([data_train,forecast_2],axis=1).plot()
figsize = (20,10)
plt.show()


# In[ ]:


# Actual vs Fitted
forecast_3 = result_q.predict(dynamic=False)
pd.concat([data_train,forecast_3],axis=1).plot()
figsize = (20,10)
plt.show()


# In[ ]:


# Forecast
fc = result_r.forecast(652, alpha=0.05)  # 95% conf
fc


# In[ ]:


fc_r = result_q.forecast(652, alpha=0.05)  # 95% conf
fc_r


# In[ ]:


fc_series3 = pd.Series(fc, index=data_test.index)
# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(data_train, label='training')
plt.plot(data_test, label='actual')
plt.plot(fc_series3, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[ ]:


fc_series4 = pd.Series(fc_r, index=data_test.index)
# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(data_train, label='training')
plt.plot(data_test, label='actual')
plt.plot(fc_series4, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
print(np.sqrt(metrics.mean_squared_error(data_test,fc_series3)))
print(mean_absolute_percentage_error(data_test,fc_series3)*100)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
print(np.sqrt(metrics.mean_squared_error(data_test,fc_series4)))
print(mean_absolute_percentage_error(data_test,fc_series4)*100)


# # FB Prophet

# In[ ]:


get_ipython().system('pip install prophet')


# In[ ]:


from prophet import Prophet


# In[ ]:


m = Prophet()


# In[ ]:


data1=data.reset_index()


# In[ ]:


data1


# In[ ]:


data1.rename(columns = {'date' : 'ds', 'price' : 'y'}, inplace = True)


# In[ ]:


data1


# In[ ]:


model4 = m.fit(data1)


# In[ ]:


future = m.make_future_dataframe(periods=30,freq ='D')
future.tail()


# In[ ]:


forecast_fb = m.predict(future)
forecast_fb[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


fig1 = m.plot(forecast_fb)


# In[ ]:


fig2 = m.plot_components(forecast_fb)


# In[ ]:


plt.figure(figsize=(12,5), dpi=100)
plt.plot(data1, label='training')
plt.plot(forecast_fb['yhat'], label='forecast')


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
print(np.sqrt(metrics.mean_squared_error(forecast_fb['yhat_lower'],forecast_fb['yhat_upper'])))
print(mean_absolute_percentage_error(forecast_fb['yhat_lower'],forecast_fb['yhat_upper'])*100)


# # OUT of all this 3 model with different value we got to know that SARIMA model with 2 , 1, 2 value is best performing so we chose SARIMA model

# In[ ]:




