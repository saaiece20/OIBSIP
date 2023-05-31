import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
car_data = pd.read_csv('path_to_dataset/car_data.csv')
features = car_data.drop('price', axis=1)
target = car_data['price']
label_encoder = LabelEncoder()
for column in features.select_dtypes(include=['object']):
    features[column] = label_encoder.fit_transform(features[column])
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
