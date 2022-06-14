from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape
train_labels.shape
test_images.shape
test_labels.shape


import matplotlib.pyplot as plt
# running this once shows the plts in gray scale as default
# https://stackoverflow.com/questions/3823752/display-image-as-grayscale-using-matplotlib
plt.gray()

# if we do run the plt.gray() ... below code would have shown a color image
plt.imshow(train_images[0])

train_labels[0]

from keras import models
from keras import layers
model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model_cnn.add(layers.MaxPooling2D(2,2))
model_cnn.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model_cnn.add(layers.MaxPooling2D(2,2))
model_cnn.add(layers.Conv2D(64, (3,3), activation='relu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation = 'relu'))
model_cnn.add(layers.Dense(10, activation = 'softmax'))
train_images.shape
train_images_cnn = train_images.reshape(60000, 28, 28, 1)
train_images_cnn.shape
train_images_cnn = train_images_cnn.astype('float32') / 255
test_images_cnn = test_images.reshape(10000, 28, 28, 1)
test_images_cnn = test_images_cnn.astype('float32') / 255

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
train_labels_cnn = to_categorical(train_labels)
test_labels_cnn = to_categorical(test_labels)
model_cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model_cnn.fit(train_images_cnn, train_labels_cnn, epochs = 5, batch_size = 60)

test_loss_cnn, test_acc_cnn = model_cnn.evaluate(test_images_cnn, test_labels_cnn)

model_cnn.save('mni2st.h5')
print("Saving the model as mni2st.h5")