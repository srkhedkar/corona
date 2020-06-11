#import packages
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# read the Indian Corona data file
df = pd.read_csv('D:\\python3\\data\\coronaCsv.csv')

# calculate recover percentage
df['recoverPercentage'] = (df['recovered'] * 100) / df['confirmed']

# preparing data. Prophet only understands y and ds columns. Hence we need to rename
# our data frame columns
df.rename(columns={'recoverPercentage': 'y', 'Date': 'ds'}, inplace=True)

# Model initialization. Create an object of class Prophet.
model = Prophet()

# Fit the data(train the model)
model.fit(df)

# Create a future data frame of future dates. Here 80 is number of days for which we want predictions.
future = model.make_future_dataframe(periods=80)

# Prediction for future dates.
forecast = model.predict(future)

# forecast has number of various columns. In this exercise we are considering only two of them.
# ds is a date column and yhat is the median predicated value.
forecast_valid = forecast[['ds','yhat']][:]
forecast_valid.rename(columns={'yhat': 'y'}, inplace=True)

#print the last predicted value
print ("The pandemic will be over by  ", forecast_valid[['ds']].iloc[-1])

# create a date index for input data frame.
df['Date'] = pd.to_datetime(df.ds)
df.index = df['Date']

# Create a date index for forecast data frame.
forecast_valid['Date'] = pd.to_datetime(forecast_valid.ds)
forecast_valid.index = forecast_valid['Date']

# plot the actual data
plt.figure(figsize=(16,8))
plt.plot(df['y'], label='Recover Percentage')

# plot the prophet predictions
plt.plot(forecast_valid[['y']], label='Future Predictions')

#set the title of the graph
plt.suptitle('Corona Predictions using The Prophet', fontsize=16)

#set the title of the graph window
fig = plt.gcf()
fig.canvas.set_window_title('Corona Predictions using The Prophet')

#display the legends
plt.legend()

#display the graph
plt.show()