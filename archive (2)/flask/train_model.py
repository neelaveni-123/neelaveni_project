print("✅ Creating motor data...")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle

# Synthetic motor data
np.random.seed(42)
data = {
    'ambient': np.random.uniform(20,40,10000),
    'coolant': np.random.uniform(30,50,10000),
    'u_d': np.random.uniform(0,1,10000),
    'u_q': np.random.uniform(0,1,10000),
    'motor_speed': np.random.uniform(1000,3000,10000),
    'i_d': np.random.uniform(-1,1,10000),
    'i_q': np.random.uniform(-1,1,10000),
    'pm': np.random.uniform(30,80,10000)
}
df = pd.DataFrame(data)

X = df[['ambient','coolant','u_d','u_q','motor_speed','i_d','i_q']]
y = df['pm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = MinMaxScaler().fit(X_train)
model = DecisionTreeRegressor().fit(scaler.transform(X_train), y_train)

pickle.dump(model, open('model.save','wb'))
pickle.dump(scaler, open('transform.save','wb'))
print("✅ SUCCESS - model.save + transform.save created!")
