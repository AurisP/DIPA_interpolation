# Import necessary modules:
# NumPy for numerical computations,
# Keras for creating the deep learning model,
# Matplotlib for plotting
# and OpenCV for image processing.

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2

# Load the MNIST dataset, which contains 28x28 grayscale images of handwritten digits.
# The dataset is split into training and test sets.
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1, which helps the model to learn more effectively.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshape the data to add an extra dimension for the grayscale color channel.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Resize the images to half the original size (14x14) to create low-resolution versions of the images.
x_train_lowres = np.array([cv2.resize(img, (14, 14)) for img in x_train])
x_test_lowres = np.array([cv2.resize(img, (14, 14)) for img in x_test])

# Define the autoencoder model. The model has a convolutional layer, an upsampling layer t
# hat doubles the size of the image, another convolutional layer, and a final convolutional layer
# with a sigmoid activation function to produce the final upscaled image.
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu',
          padding='same', input_shape=(14, 14, 1)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# Compile the model with the Adam optimizer and binary crossentropy as the loss function.
model.compile(optimizer=Adam(), loss='binary_crossentropy')

# Train the autoencoder using the low-resolution images as inputs and the original high-resolution images as targets.
model.fit(x_train_lowres, x_train, epochs=10, batch_size=128,
          validation_data=(x_test_lowres, x_test))

# Choose a random image from the test set and add an extra dimension for the batch size.
idx = np.random.randint(len(x_test_lowres))
test_img_lowres = x_test_lowres[idx]
test_img_highres = x_test[idx]
test_img_input = np.expand_dims(test_img_lowres, axis=0)

# Use the model to upscale the image.
upscaled_img = model.predict(test_img_input)

# Remove the extra batch size dimension.
upscaled_img = np.squeeze(upscaled_img, axis=0)
test_img_input = np.squeeze(test_img_input, axis=0)

# Display the original low-resolution image, the upscaled image, and the original high-resolution image.
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(test_img_lowres, cmap='gray')
ax[0].set_title("Original Low-res Image")
ax[1].imshow(upscaled_img, cmap='gray')
ax[1].set_title("Upscaled Image")
ax[2].imshow(test_img_highres, cmap='gray')
ax[2].set_title("Original High-res Image")
plt.savefig('neural_network.png')
plt.show()
