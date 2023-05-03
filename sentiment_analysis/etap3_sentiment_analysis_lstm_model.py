import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load preprocessed data
df = pd.read_csv('../csv_files/02_preprocessed_data/preprocessed_data.tsv', sep='\t')
df['avg_vector'] = df['avg_vector'].apply(lambda x: ast.literal_eval(x))

# Split data into training and testing sets
X = np.vstack(df['avg_vector'].values)
y = df['cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to categorical format
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Reshape the input data for use in the LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train_cat.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_cat, epochs=100, batch_size=128, validation_data=(X_test, y_test_cat))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f'Test Loss: {loss:.3f}\nTest Accuracy: {accuracy:.3f}')
