import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import pandas as pd

data = pd.read_csv('../Data/GermanCreditDataset/germanC.csv', delimiter=';' )

print(data.shape)  # This will print (number_of_rows, number_of_columns)
print(data.head())

X = data.drop('Creditability', axis=1)  # Features
y = data['Creditability']               # Labels

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
#Building the NN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1], 20)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', test_accuracy)

#model.save('../Models/CreditRiskAssesmentModel.h5')  # Creates a HDF5 file 'CreditRiskAssesmentModel.h5'

# Load the new test data
new_test_data = pd.read_csv('../Data/new_test_data.csv')
new_data_scaled = scaler.transform(new_test_data)
predictions = model.predict(new_data_scaled)

# Set a threshold (commonly 0.5) to interpret the probabilities
class_predictions = (predictions > 0.5).astype(int)

# View the predictions
print(class_predictions)