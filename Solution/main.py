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

# train_test_split is a function from Scikit-learn used to split datasets into two parts: in this case, a training set and a test set.
# X_scaled and y are the features and labels of your dataset, respectively.
# test_size=0.2 means 20% of the data is reserved for the test set, and the remaining 80% is used for training.
# random_state=42 ensures that the split is reproducible. It means you'll get the same split every time you run this code.

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# This line further splits the training set into two parts: a smaller training set and a validation set.
# The validation set is 25% of the (already 80% reduced) training data, which is effectively 20% of the original dataset (0.25 x 0.8 = 0.2). This leaves 60% of the original data for training, 20% for validation, and 20% for testing.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

#Building the NN
# Sequential model in Keras is a linear stack of layers. It's like constructing a pipeline of neural network layers.
# The first Dense layer is the first hidden layer of the neural network. It has 64 neurons (or units). The activation='relu' means it uses the ReLU (Rectified Linear Unit) activation function, which is a common choice for hidden layers.
# input_shape=(X_train.shape[1], 20) sets the shape of the input data that the model will receive. However, there seems to be an error here. It should be just input_shape=(X_train.shape[1],). This defines the number of features the model expects as input.
# The second Dense layer is another hidden layer with 32 neurons, also using the ReLU activation function.
# The final Dense layer is the output layer of the network. It has 1 neuron with a sigmoid activation function, making it suitable for binary classification (predicting 0 or 1).

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# model.compile() is a method to configure the model for training.
# optimizer='adam': This specifies the optimization algorithm to use while training. Adam is a popular optimization algorithm in deep learning because it combines the best properties of the AdaGrad and RMSProp algorithms to handle sparse gradients on noisy problems.
# loss='binary_crossentropy': This sets the loss function for the model. Since this is a binary classification problem (predicting two classes, 0 or 1), 'binary_crossentropy' is used. Crossentropy loss measures the performance of a classification model whose output is a probability value between 0 and 1. It increases as the predicted probability diverges from the actual label.
# metrics=['accuracy']: This defines the list of metrics to be evaluated by the model during training and testing. Here, 'accuracy' is used, which is the fraction of correctly classified instances.

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit() is used to train the model for a fixed number of epochs (iterations on a dataset).
# X_train, y_train: These are the training input features and labels.
# epochs=10: This defines the number of times the learning algorithm will work through the entire training dataset. Here, it's set to 10, meaning the algorithm will pass through the data 10 times.
# validation_data=(X_val, y_val): This is your validation data, used to evaluate the loss and any model metrics at the end of each epoch. The model is not trained on this data. It's a check to see how well the model is generalizing to unseen data.
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# model.evaluate() is used to evaluate the performance of the model on the test dataset.
# X_test, y_test: These are the test input features and labels, used to evaluate the model after training.
# The function returns the loss value (test_loss) and metrics values (here, test_accuracy) for the model in test mode. In this case, it returns the accuracy of the model on the test data, which is a key metric for understanding how well the model performs on unseen data.
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', test_accuracy)

#model.save('../Models/CreditRiskAssesmentModel.h5')  # Creates a HDF5 file 'CreditRiskAssesmentModel.h5'

# Load new test data
new_test_data = pd.read_csv('../Data/new_test_data.csv')
new_data_scaled = scaler.transform(new_test_data)
predictions = model.predict(new_data_scaled)

# Set a threshold (commonly 0.5) to interpret the probabilities
class_predictions = (predictions > 0.5).astype(int)

# View the predictions
print(class_predictions)