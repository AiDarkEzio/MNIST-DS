import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

# Load data
data = pd.read_csv('input/train.csv')
print(data.head())

# Convert to numpy array
data = np.array(data)
np.random.shuffle(data)

# Split data into training and validation sets
X_train = data[1000:, 1:] / 255.0
Y_train = to_categorical(data[1000:, 0])
X_dev = data[:1000, 1:] / 255.0
Y_dev = to_categorical(data[:1000, 0])

# Build the model
model = Sequential([
    Dense(128, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_dev, Y_dev))

# Save the entire model
model.save('mnist_model.h5')

# Save only the model weights
model.save_weights('mnist_model.weights.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_dev, Y_dev)
print(f"Validation Accuracy: {accuracy}")

# Making predictions
def make_predictions(index):
    current_image = X_train[index].reshape(28, 28)
    prediction = model.predict(X_train[index].reshape(1, 784))
    label = np.argmax(Y_train[index])
    predicted_label = np.argmax(prediction)
    print(f"Prediction: {predicted_label}")
    print(f"Label: {label}")
    
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Test prediction on a specific index
make_predictions(150)

# Load the saved model (optional)
loaded_model = tf.keras.models.load_model('mnist_model.h5')

# Evaluate the loaded model (optional)
loaded_loss, loaded_accuracy = loaded_model.evaluate(X_dev, Y_dev)
print(f"Loaded Model Validation Accuracy: {loaded_accuracy}")
