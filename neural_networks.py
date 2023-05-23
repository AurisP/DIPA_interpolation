# Import necessary modules:
# NumPy for numerical computations,
# Keras for creating the deep learning model,
# Matplotlib for plotting
# and OpenCV for image processing.

import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2

(x_train, _), (x_test, _) = cifar10.load_data()

# Lets determine the dataset characteristics
print('Training Images: {}'.format(x_train.shape))
print('Testing Images: {}'.format(x_test.shape))

# Normalize the pixel values to be between 0 and 1, which helps the model to learn more effectively.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshape the data to add an extra dimension for the grayscale color channel.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

# Resize the images to half the original size (16x16) to create low-resolution versions of the images.
x_train_lowres = np.array([cv2.resize(img, (16, 16)) for img in x_train])
x_test_lowres = np.array([cv2.resize(img, (16, 16)) for img in x_test])

# Define the autoencoder model. The model has a convolutional layer, 
# an upsampling layer that doubles the size of the image, another convolutional layer, 
# and a final convolutional layer with a sigmoid activation function to produce the final upscaled image.
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(16, 16, 3)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

# Compile the model with the Adam optimizer and mean squared error as the loss function.
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the autoencoder using the low-resolution images as inputs and the original high-resolution images as targets.
model.fit(x_train_lowres, x_train, epochs=50, batch_size=256, validation_data=(x_test_lowres, x_test))

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
plt.figure(figsize=(16,8))
plt.subplot(131),plt.imshow(test_img_lowres,'gray'),plt.title('Original Low-res Image')
plt.subplot(132),plt.imshow(test_img_highres,'gray'),plt.title('Original High-res Image')
plt.subplot(133),plt.imshow(upscaled_img,'gray'),plt.title('Upscaled Image')
plt.savefig('neural_network.png')
plt.show()
