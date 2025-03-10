import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
import tensorflow as tf
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay 
from tensorflow import squeeze 
import matplotlib.pyplot as plt 

# Set the random seed for reproducibility
tf.random.set_seed(0)

# Load the CSV file where the first row is the header data 
df = pd.read_csv('mushroom_data_less.csv', header=0) 

feature_names = [
    'bruises', 'odor', 'gill-spacing', 'gill-size', 
    'gill-color', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'ring-type',
    'spore-print-color', 'population', 'habitat'
]

# Encode the categorical columns using LabelEncoder
le = LabelEncoder() 
label_mappings = {}

for column in df.columns:
    if df[column].dtype == 'object': 
        df[column] = le.fit_transform(df[column])
        label_mappings[column] = dict(zip(le.classes_, range(len(le.classes_))))

# Split the data into features and target
features = df.iloc[:, :-1] # all columns except the last one
target = df.iloc[:, -1] # last column

# Split the dataset into training, validation and testing sets with a ratio of 6:2:2
# Step 1: Split data into (training + validation) and testing sets, splitting ratio ==> 0.8 : 0.2
X_train_val, X_test, y_train_val, y_test = train_test_split(features, target, test_size=0.2, random_state=42) 

# Step 2: Split the training + validation set into separate training and validation sets, splitting ratio ==> 0.75 : 0.25 
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42) 

# Create a feed-forward network using the Keras API with one input layer, one hidden layer, and one output layer
initializer = RandomNormal(mean=0.0, stddev=1.0, seed=42)
model = Sequential() 
model.add(Dense(10, input_shape=(13,), activation = 'sigmoid', kernel_initializer=initializer))
model.add(Dense(1, activation = 'sigmoid', kernel_initializer=initializer))

# Compile the model with the Adam optimizer
adam_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy']) 

# Define the early stopping callback
callback = EarlyStopping(monitor='val_loss', verbose=1, patience=20)

# Train the model and store training history
history = model.fit(X_train, y_train, epochs=1000, batch_size=25, verbose=1, validation_data = (X_val,y_val), shuffle=True, callbacks=[callback])

# Plot the loss during the training and validation for each epoch 
plt.plot(history.epoch, history.history["loss"], 'g', label='Training loss') 
plt.plot(history.epoch, history.history["val_loss"], 'r', label='Validation loss') 
plt.title('Training Loss Plot') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend(loc='upper right') 
plt.show()

# Plot the accuracy during the training and validation for each epoch 
plt.plot(history.epoch, history.history["accuracy"], 'g', label='Training Accuracy') 
plt.plot(history.epoch, history.history["val_accuracy"], 'r', label='Validation Accuracy') 
plt.title('Training Accuracy Plot') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend(loc='upper right') 
plt.show() 

# Evaluate the model using the test dataset
loss_and_metrics_for_test_dataset = model.evaluate(X_test, y_test) 
print('Loss(Test Dataset) = ',loss_and_metrics_for_test_dataset[0]) 
print('Accuracy(Test Dataset) = ',loss_and_metrics_for_test_dataset[1]) 
predicted_y_for_test_dataset = model.predict(X_test) 
predicted_for_test_dataset = squeeze(predicted_y_for_test_dataset) 
predicted_for_test_dataset = np.array([1 if x >= 0.5 else 0 for x in predicted_for_test_dataset]) 
print('Classification report for test dataset\n',classification_report(y_test, predicted_for_test_dataset))

# Evaluate the model using the whole dataset
loss_and_metrics_for_whole_dataset = model.evaluate(features, target) 
print('Loss(Whole Dataset) = ',loss_and_metrics_for_whole_dataset[0]) 
print('Accuracy(Whole Dataset) = ',loss_and_metrics_for_whole_dataset[1]) 
predicted_y = model.predict(features) 
predicted = squeeze(predicted_y) 
predicted = np.array([1 if x >= 0.5 else 0 for x in predicted]) 
print('Classification report for whole dataset\n',classification_report(target, predicted)) 

# Display Confusion Matrix 
actual = np.array(target) 
conf_mat = confusion_matrix(actual, predicted) 
print('Confusion Matrix:\n',conf_mat) 

# Plot confusion matrix with number of true positives and negatives 
displ1 = ConfusionMatrixDisplay.from_predictions(actual, predicted) 
plt.show() 

# Plot confusion matrix with the percentage of true positives and negatives 
displ2 = ConfusionMatrixDisplay.from_predictions(actual, predicted, normalize='true') 
plt.show()