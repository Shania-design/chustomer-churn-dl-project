import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tenflow.keras.models import load_model

df = pd.read_csv('data/chustomer_churn.csv')

x = df.drop(['customer Id', 'surname','exited'],axis=1)
y = df('exited')