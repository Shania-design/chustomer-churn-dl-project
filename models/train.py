import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tenflow.keras.models import load_model

df = pd.read_csv('data/chustomer_churn.csv')

x = df.drop(['customer Id', 'surname','exited'],axis=1)
y = df('exited')

X['Geography'] = LabelEncoder().fit_transform(X['Geography'])
X['Gender'] = LabelEncoder().fit_transform(x['Gender'])

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)