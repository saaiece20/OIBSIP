import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv('path_to_dataset/unemployment_data.csv')
date = data['date'].values.reshape(-1, 1) 
unemployment_rate = data['unemployment_rate'] 
plt.plot(date, unemployment_rate)
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment Rate During COVID-19')
plt.show()
regression_model = LinearRegression()
regression_model.fit(date, unemployment_rate)
trend = regression_model.predict(date)
plt.plot(date, unemployment_rate, label='Actual')
plt.plot(date, trend, label='Trend')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment Rate Trend During COVID-19')
plt.legend()
plt.show()



