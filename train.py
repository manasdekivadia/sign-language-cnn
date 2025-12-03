import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Paths
dataset_path = r"slr\model\keypoint.csv"
best_model_path = r"slr_model_best.hdf5"     # best model
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

# Build model
print("Building model...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((42,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === CALLBACKS ===
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=50,                # stop if no improvement for 50 epochs
    restore_best_weights=True
)

# Train with callbacks
print("Training model...")
model.fit(
    X_train,
    y_train,
    epochs=1000,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# =============================
# Load best model before TFLite
# =============================
print("Loading best model...")
best_model = tf.keras.models.load_model(best_model_path)

# Convert to TFLite
print(f"Converting to TFLite: {tflite_output_path}")

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()

with open(tflite_output_path, "wb") as f:
    f.write(tflite_model)

print("Training complete! Best model saved and exported to TFLite.")
