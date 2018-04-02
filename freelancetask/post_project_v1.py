
# coding: utf-8

# In[1]:


import pandas as pd
import re
import datetime
import math
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import pandas as pd
import itertools


# In[54]:
def run1(d):
    ans=dict()
    o_data = pd.read_excel(r'freelancetask/Post_detction_Task_cpie.xlsx',encoding="utf-8")


    # In[56]:


    o_data.drop(o_data.index[[1326,24652,53921]], inplace=True)

    timeSeriesData = pd.DataFrame()
    timeSeriesData['date'] = o_data['Created Time']
    timeSeriesData['totalreactions'] = o_data['Reaction Count']
    timeSeriesData['Share Count'] = o_data['Share Count']
    timeSeriesData['Comment Count'] = o_data['Comment Count']
    timeSeriesData1 = timeSeriesData.dropna(how='all')

    # In[26]:


    len(timeSeriesData1)

    # In[57]:


    date = []
    totalCount = []
    month = []
    for i in timeSeriesData1.index:
        print(i)

        if type(timeSeriesData1['date'][i]) == str and timeSeriesData1['totalreactions'][i] > 0:
            match = re.search('\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}', timeSeriesData1['date'][i])
            month.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').month)
            date.append(timeSeriesData1['date'][i])
            share = timeSeriesData1['Share Count'][i]
            comment = timeSeriesData1['Comment Count'][i]
            total = timeSeriesData1['totalreactions'][i]
            if (math.isnan(share)):
                share = 0

            if (math.isnan(comment)):
                comment = 0

            if (math.isnan(total)):
                total = 0

            totalCount.append(share + comment + total)

    # In[58]:


    new_Data = pd.DataFrame()

    # In[59]:


    new_Data['date'] = date

    # In[60]:


    new_Data['totalCount'] = totalCount

    # In[143]:


    # new_Data.apply(lambda s: pd.datetime.strptime(s['date'], '%Y-%m-%d-%H:%M:%S') , axis=1).rename('date').reset_index()


    # In[61]:


    newDataDate = new_Data.groupby('date')['totalCount'].sum()

    # In[144]:


    # newDataDate


    # In[34]:


    # import matplotlib.pylab as plt


    # In[35]:


    # plt.plot(newDataDate.head(100))


    # In[36]:


    # from statsmodels.graphics.tsaplots import plot_acf


    # In[145]:


    # series=newDataDate
    # plot_acf(series, lags=31)
    # pyplot.show()


    # In[146]:


    # from pandas import Series
    # from pandas import DataFrame
    # from pandas import concat
    # from matplotlib import pyplot
    # from sklearn.metrics import mean_squared_error
    # #series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
    # # create lagged dataset
    # values = DataFrame(series.values)
    # dataframe = concat([values.shift(1), values], axis=1)
    # dataframe.columns = ['t-1', 't+1']
    # # split into train and test sets
    # X = dataframe.values
    # train, test = X[1:len(X)-7], X[len(X)-7:]
    # train_X, train_y = train[:,0], train[:,1]
    # test_X, test_y = test[:,0], test[:,1]

    # # persistence model
    # def model_persistence(x):
    # 	return x

    # # walk-forward validation
    # predictions = list()
    # for x in test_X:
    # 	yhat = model_persistence(x)
    # 	predictions.append(yhat)
    # test_score = mean_squared_error(test_y, predictions)
    # print('Test MSE: %.3f' % test_score)
    # # plot predictions vs expected
    # pyplot.plot(test_y)
    # pyplot.plot(predictions, color='red')
    # pyplot.show()


    # In[147]:


    # from pandas import Series
    # from matplotlib import pyplot
    # from statsmodels.tsa.ar_model import AR
    # from sklearn.metrics import mean_squared_error
    # #series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
    # # split dataset
    # X = series.values
    # train, test = X[1:len(X)-7], X[len(X)-7:]
    # # train autoregression
    # model = AR(train)
    # model_fit = model.fit()
    # print('Lag: %s' % model_fit.k_ar)
    # print('Coefficients: %s' % model_fit.params)
    # # make predictions
    # predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    # for i in range(len(predictions)):
    # 	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    # error = mean_squared_error(test, predictions)
    # print('Test MSE: %.3f' % error)
    # # plot results
    # pyplot.plot(test)
    # pyplot.plot(predictions, color='red')
    # pyplot.show()


    # In[6]:


    data = pd.read_csv(r'freelancetask/out1.csv')

    # In[7]:


    df = data.drop('Unnamed: 0', 1)

    # In[14]:


    import pickle

    # model = Prophet()
    # model.fit(df);
    #model = pickle.load(open('model.sav', 'rb'))

    # In[13]:





    # # predicting best hours in next 3 days

    # In[15]:


    # future = model.make_future_dataframe(periods=72, freq = 'h')
    # future.tail(73)


    # In[55]:


    # forecast = model.predict(future.tail(20000))


    # In[52]:


    # import pickle
    # filename = 'finalized_model.sav'
    # pickle.dump(forecast, open(filename, 'wb'))


    # In[16]:


    forecast = pickle.load(open('freelancetask/predec_model.sav', 'rb'))

    # In[21]:


    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # In[22]:



    ans_hourly = result.sort_values(by='yhat', ascending=False).tail()

    # # Best 5 hours in next 3 days

    # In[26]:


    ans['hourly']=ans_hourly['ds']

    # In[61]:


    # month=[]
    # day=[]
    # hour=[]
    # year=[]

    # for i in ans_hourly.index:
    #         convert=datetime.strftime(ans_hourly['ds'][i], "%Y-%m-%d-%H:%M:%S")
    #         match = re.search('\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}', convert)
    #         year.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').year)
    #         month.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').month)
    #         day.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').day)
    #         hour.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').hour)


    # In[27]:


    # future_day = model.make_future_dataframe(periods=30, freq = 'd')


    # In[33]:


    forecast_day = pickle.load(open('freelancetask/predec_model_day.sav', 'rb'))
    # forecast_day = model.predict(future_day)
    result_day = forecast_day[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)

    # In[36]:



    ans_day = result_day.sort_values(by='yhat', ascending=False).tail()

    # # Best 5 days for this month

    # In[35]:


    print(result_day['ds'])
    ans['daily']=result_day['ds']
    # In[37]:


    # future_week = model.make_future_dataframe(periods=30, freq = 'w')
    # future_week.tail(4)


    # In[43]:


    # forecast_week = model.predict(future_week)
    forecast_week = pickle.load(open('freelancetask/predec_model_weekly.sav', 'rb'))
    result_week = forecast_week[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(4)

    # In[ ]:


    ans_week = result_week.sort_values(by='yhat', ascending=False).tail()

    # # best week for current month

    # In[87]:


    #print(ans_week['ds'])
    ans['weekly']=ans_week['ds']

    # In[74]:


    # month=[]
    # day=[]
    # hour=[]
    # year=[]

    # for i in ans_day.index:
    #         convert=datetime.strftime(ans_day['ds'][i], "%Y-%m-%d-%H:%M:%S")
    #         match = re.search('\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}', convert)
    #         year.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').year)
    #         month.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').month)
    #         day.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').day)
    #         hour.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').hour)


    # In[75]:





    # In[76]:


    # month=[]
    # day=[]
    # hour=[]
    # year=[]

    # for i in ans_week.index:
    #         convert=datetime.strftime(ans_week['ds'][i], "%Y-%m-%d-%H:%M:%S")
    #         match = re.search('\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}', convert)
    #         year.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').year)
    #         month.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').month)
    #         day.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').day)
    #         hour.append(datetime.strptime(match.group(), '%Y-%m-%d-%H:%M:%S').hour)


    # In[77]:





    # In[51]:





    # In[48]:
    return ans

    # In[4]:


