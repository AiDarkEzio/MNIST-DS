import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('mnist_model.h5')

def generate_image_for_digit(digit, model, iterations=1000, learning_rate=0.1):
    # Create a random image
    image = np.random.rand(1, 784).astype(np.float32)

    # Convert the image to a TensorFlow variable
    image = tf.Variable(image)

    # Define the target output
    target = tf.one_hot(digit, 10)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = model(image)
            loss = -tf.reduce_mean(prediction * target)  # We use negative loss for gradient ascent

        gradients = tape.gradient(loss, image)
        image.assign_add(learning_rate * gradients)  # Update the image using gradient ascent

        # Clip the image values to be between 0 and 1
        image.assign(tf.clip_by_value(image, 0.0, 1.0))
        
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.numpy()}")

    return image.numpy().reshape(28, 28)

# Specify the digit you want to generate
numb = int(input("Enter the number: "))
digit = numb if (numb >= 0 and numb <= 9) else (numb % 10)
generated_image = generate_image_for_digit(digit, model)

# Display the generated image
plt.gray()
plt.imshow(generated_image, interpolation='nearest')
plt.title(f"Generated Image for Digit {digit}")
plt.show()


## Image Isn't show any number. it just random noise