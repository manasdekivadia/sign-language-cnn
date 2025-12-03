import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Paths
dataset_path = r"slr\model\keypoint.csv"
best_model_path = r"slr_model_best.hdf5"    
tflite_output_path = r"slr\model\slr_model.tflite"

# Number of classes 
NUM_CLASSES = 24

# Load dataset
print("Loading dataset...")
X = np.loadtxt(dataset_path, delimiter=',', dtype='float32', usecols=list(range(1, 43)))
y = np.loadtxt(dataset_path, delimiter=',', dtype='int32', usecols=(0,))

# Split into training and test sets
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=RANDOM_SEED)

# Define the model
print("Building model...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((42,)),  # 21 keypoints * 2 (x, y)
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training model...")
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save the model in HDF5 format
print(f"Saving model to {model_h5_path}...")
model.save(model_h5_path)

# Convert to TFLite
print(f"Converting model to TFLite and saving to {model_tflite_path}...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(model_tflite_path, 'wb') as f:
    f.write(tflite_model)

print("Training complete.")
