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
from SALib.analyze import sobol 
from SALib.sample import saltelli

# Set the random seed for reproducibility
tf.random.set_seed(0)

# Load the CSV file where the first row is the header data 
df = pd.read_csv('mushroom_data_less.csv', header=0) 

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

# Find maximum value for each encoded feature
max_values = [df[feature].max() for feature in features.columns]
print("Maximum values for each encoded feature:", max_values)

# Define the Sobol problem with correct bounds for each encoded feature
# Bounds are set based on the number of subcategories for each feature, which is specified by the data donator
# For example, if a feature has 5 subcategories, the bound would be [0, 4] as the subcategories are indexed from 0 to 4
problem = {
    'num_vars': 13,
    'names': [
    'bruises', 'odor', 'gill-spacing', 'gill-size', 
    'gill-color', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'ring-type',
    'spore-print-color', 'population', 'habitat'
    ],
    'bounds': [
        [0, 1], [0, 8], [0, 2], [0, 1], [0, 11], 
        [0, 3], [0, 3], [0, 8], [0, 8], [0, 7], 
        [0, 8], [0, 5], [0, 6]                    
    ]
}

# Generate sample points using Saltelli's sampling method 
param_values_continuous = saltelli.sample(problem, 2048)
param_values = np.round(param_values_continuous).astype(int)

# Evaluate the model on the sample points 
predictions = model.predict(param_values) 

# Flatten the predictions to a 1D array (if necessary) 
predictions = predictions.flatten() 

# Perform Sobol sensitivity analysis 
Si = sobol.analyze(problem, predictions) 

# Output sensitivity indices 
print("Total-order sensitivity indices:", Si['ST'])