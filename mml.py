
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("FuelConsumption.csv")
#use required features
cdf = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']]


x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]
regressor = LinearRegression()

regressor.fit(x, y)

pickle.dump(regressor, open('model.pkl','wb'))