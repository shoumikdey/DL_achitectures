# -*- coding: utf-8 -*-
"""ATTENTION_UNET

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bd-ws0HHjqBK2GGnAnsQ6-Szy1WWX81Y
"""

from google.colab import drive
drive.mount('/content/drive')

!mkdir /content/data/

!cp /content/drive/My\ Drive/datasets/sar_dataset.zip -d /content/data/

!unzip /content/data/sar_dataset.zip -d /content/data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import PIL
import pandas as pd
import shutil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import skimage.io
import skimage.transform as trans

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    rescale = 1./255,
    fill_mode = 'nearest'
)
mask_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    rescale = 1./255,
    fill_mode = 'nearest'
)

SEED = 42

train_img = img_gen.flow_from_directory(
    "/content/data/train",
    target_size = (256, 256),
    seed = SEED,
    class_mode = None,
    batch_size = 16,
    color_mode='grayscale'
)

train_mask = mask_gen.flow_from_directory(
    "/content/data/train_labels",
    target_size = (256, 256),
    seed = SEED,
    class_mode = None,
    batch_size = 16,
    color_mode='grayscale'
)

test_img = img_gen.flow_from_directory(
    "/content/data/test",
    target_size = (256, 256),
    seed = SEED,
    class_mode = None,
    batch_size = 16,
    color_mode='grayscale'
)

test_mask = mask_gen.flow_from_directory(
    "/content/data/test_labels",
    target_size = (256, 256),
    seed = SEED,
    class_mode = None,
    batch_size = 16,
    color_mode = 'grayscale'
)

train_set = zip(train_img, train_mask)
test_set = zip(test_img, test_mask)

def attentionGate(x, g, fl, fg, fint):
  
  x1 = tf.keras.layers.Conv2D(fl, (1,1), padding="same", activation=None, kernel_initializer='he_normal')(x)
  #x1 = tf.keras.layers.BatchNormalization()(x1)
  g1 = tf.keras.layers.Conv2D(fg, (1,1), padding = "same", activation = None, kernel_initializer='he_normal')(g)
  #g1 = tf.keras.layers.BatchNormalization()(g1)
  #x1 = tf.keras.layers.Reshape(list(g1.get_shape()[1:]))(x1)
  #g1 = tf.keras.layers.Reshape(x.get_shape()[1:])(g1)
  #g1 = tf.keras.layers.Reshape(x.shape)(g1)
  print(x1.get_shape())
  int_sig = tf.keras.layers.Add()([x1, g1])
  int_sig = tf.keras.layers.Activation('relu')(int_sig)
  int_sig = tf.keras.layers.Conv2D(fint, (1,1), padding = "same", activation=None, kernel_initializer = 'he_normal')(int_sig)
  #int_sig = tf.keras.layers.BatchNormalization()(int_sig)
  int_sig = tf.keras.layers.Activation('sigmoid')(int_sig)
  int_sig = tf.keras.layers.Multiply()([x, int_sig])
  print(x1.get_shape())
  return int_sig

def get_model(input_size = (256, 256, 1)):
  filters = [64, 128, 256, 512, 1024]
  input_layer = tf.keras.layers.Input(input_size)
  
  conv1 = tf.keras.layers.Conv2D(filters[0], (3,3), padding="same", activation='relu', kernel_initializer='he_normal')(input_layer)
  conv1 = tf.keras.layers.Conv2D(filters[0], (3,3), padding="same", activation='relu', kernel_initializer='he_normal')(conv1)
  conv1 = tf.keras.layers.Dropout(0.25)(conv1)
  mp1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)

  conv2 = tf.keras.layers.Conv2D(filters[1], (3,3), padding="same", activation='relu', kernel_initializer='he_normal')(mp1)
  conv2 = tf.keras.layers.Conv2D(filters[1], (3,3), padding="same", activation='relu', kernel_initializer='he_normal')(conv2)
  conv2 = tf.keras.layers.Dropout(0.5)(conv2)
  mp2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)

  conv3 = tf.keras.layers.Conv2D(filters[2], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(mp2)
  conv3 = tf.keras.layers.Conv2D(filters[2], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
  conv3 = tf.keras.layers.Dropout(0.5)(conv3)
  mp3 = tf.keras.layers.MaxPooling2D((2,2))(conv3)

  conv4 = tf.keras.layers.Conv2D(filters[3], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(mp3)
  conv4 = tf.keras.layers.Conv2D(filters[3], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv4)
  conv4 = tf.keras.layers.Dropout(0.25)(conv4)
  mp4 = tf.keras.layers.MaxPooling2D((2,2))(conv4)

  conv5 = tf.keras.layers.Conv2D(filters[4], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(mp4)
  conv5 = tf.keras.layers.Conv2D(filters[4], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv5)

 

  upconv4 = tf.keras.layers.Conv2DTranspose(filters[4], (3,3), strides=(2,2), padding='same', kernel_initializer='he_normal')(conv5)
  ag4 = attentionGate(x = conv4, g = upconv4, fl = filters[3], fg = filters[3], fint = filters[3])
  upconv4 = tf.keras.layers.concatenate([upconv4, ag4])

  c_upconv4 = tf.keras.layers.Conv2D(filters[3], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(upconv4)
  c_upconv4 = tf.keras.layers.Conv2D(filters[3], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c_upconv4)
  
  
  upconv3 = tf.keras.layers.Conv2DTranspose(filters[3], (3,3),strides=(2,2), padding='same', kernel_initializer='he_normal')(c_upconv4)
  ag3 = attentionGate(x = conv3, g = upconv3, fl = filters[2], fg =filters[2], fint = filters[2])
  upconv3 = tf.keras.layers.concatenate([upconv3, ag3])

  c_upconv3 = tf.keras.layers.Conv2D(filters[2], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(upconv3)
  c_upconv3 = tf.keras.layers.Conv2D(filters[2], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c_upconv3)

  
  upconv2 = tf.keras.layers.Conv2DTranspose(filters[2],(3,3), strides=(2,2), padding='same', kernel_initializer='he_normal')(c_upconv3)
  ag2 = attentionGate(x = conv2, g = upconv2, fl = filters[1], fg = filters[1], fint = filters[1])
  upconv2 = tf.keras.layers.concatenate([upconv2, ag2])

  c_upconv2 = tf.keras.layers.Conv2D(filters[1], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(upconv2)
  c_upconv2 = tf.keras.layers.Conv2D(filters[1], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c_upconv2)

  
  upconv1 = tf.keras.layers.Conv2DTranspose(filters[1],(3,3), strides=(2,2), padding='same', kernel_initializer='he_normal')(c_upconv2)
  ag1 = attentionGate(x = conv1, g = upconv1, fl = filters[0], fg = filters[0], fint = filters[0])
  upconv1 = tf.keras.layers.concatenate([upconv1, ag1])

  output_layer = tf.keras.layers.Conv2D(1, (1,1), padding="same", activation="sigmoid", kernel_initializer='he_normal')(upconv1)

  return input_layer, output_layer

input, output = get_model()
model = tf.keras.models.Model(inputs = [input], outputs = [output])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

img = cv2.imread("/content/data/test/test_set/100.png", cv2.IMREAD_GRAYSCALE)
print(img[tf.newaxis,...,tf.newaxis].shape)
img_pred = model.predict(img[tf.newaxis,...,tf.newaxis])

print("test image")
plt.imshow(img)
plt.show()
print("actual mask")
plt.imshow(cv2.imread("/content/data/test_labels/test_labels/100.png", cv2.IMREAD_GRAYSCALE).squeeze())
plt.show()
print("prediction prior to training")
plt.imshow(img_pred.squeeze())
plt.show()

STEPS_PER_EPOCH = 89
EPOCHS = 100
VALIDATION_STEPS = 22
save_model = tf.keras.callbacks.ModelCheckpoint("unet_2.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

hist = model.fit(
    train_set,
    epochs = EPOCHS,
    steps_per_epoch = STEPS_PER_EPOCH,
    validation_data = test_set,
    validation_steps = VALIDATION_STEPS,
    callbacks = [save_model, early]
)

img = cv2.imread("/content/data/test/test_set/100.png", cv2.IMREAD_GRAYSCALE)
saved_model = tf.keras.models.load_model("unet_2.h5")
p = model.predict(img[tf.newaxis,...,tf.newaxis])
print(np.amin(p))
#ret, p = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
print("test image")
plt.imshow(img, cmap = 'gray')
plt.show()
print("actual mask")
plt.imshow(cv2.imread("/content/data/test_labels/test_labels/100.png", cv2.IMREAD_GRAYSCALE).squeeze(), cmap = 'gray')
plt.show()
print("prediction")
plt.imshow(p.squeeze(), cmap = 'gray')
plt.show()

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(100)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

!cp /content/att_unet.h5 -d /content/drive/My\ Drive/datasets/att_unet.h5