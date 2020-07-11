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

x, y = next (train_set)
for i in range(0, 16):
  print(y[i].shape)
  plt.imshow(x[i].transpose(0, 1, 2).squeeze())
  plt.show()
  plt.imshow(y[i].transpose(0,1,2).squeeze())
  plt.show()

def get_model(input_size=(256, 256, 1)):
  input_layer = tf.keras.layers.Input(input_size)
  c1 = tf.keras.layers.Conv2D(64, (3,3), padding="same", activation='relu', kernel_initializer='he_normal')(input_layer)
  c2 = tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu", kernel_initializer='he_normal')(c1)
  mp1 = tf.keras.layers.MaxPooling2D((2,2))(c2)
  #mp1 = tf.keras.layers.Dropout(0.25)(mp1)

  c3 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(mp1)
  c4 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c3)
  mp2 = tf.keras.layers.MaxPooling2D((2,2))(c4)
  #mp2 = tf.keras.layers.Dropout(0.5)(mp2)

  c5 = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(mp2)
  c6 = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c5)
  mp3 = tf.keras.layers.MaxPooling2D((2,2))(c6)
 # mp3 = tf.keras.layers.Dropout(0.5)(mp3)

  c7 = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(mp3)
  c8 = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c7)
  mp4 = tf.keras.layers.MaxPooling2D((2,2))(c8)
 #mp4 = tf.keras.layers.Dropout(0.5)(mp4)

  c9 = tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(mp4)
  c10 = tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c9)

  dc1 = tf.keras.layers.Conv2DTranspose(1024, (3, 3), strides=(2,2), padding='same')(c10)
  print(c10.get_shape(), c8.get_shape())
  concat1 = tf.keras.layers.concatenate([dc1, c8])
  #concat1 = tf.keras.layers.Dropout(0.5)(concat1)
  c11 = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(concat1)
  c12 = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c11)

  dc2 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2,2), padding='same')(c12)
  concat2 = tf.keras.layers.concatenate([dc2, c6])
  #concat2 = tf.keras.layers.Dropout(0.5)(concat2)
  c21 = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(concat2)
  c22 = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c21)

  dc3 = tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(c22)
  concat3 = tf.keras.layers.concatenate([dc3, c4])
  #concat3 = tf.keras.layers.Dropout(0.5)(concat3)
  c31 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(concat3)
  c32 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c31)

  dc4 = tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(c32)
  concat4 = tf.keras.layers.concatenate([dc4, c2])
  #concat4 = tf.keras.layers.Dropout(0.5)(concat4)
  c41 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(concat4)
  c42 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c41)

  output_layer = tf.keras.layers.Conv2D(1, (1,1), padding="same", activation="sigmoid")(c42)
  
  return input_layer, output_layer

input_layer, output_layer = get_model()
model = tf.keras.models.Model(inputs = [input_layer], outputs = [output_layer])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

img = cv2.imread("/content/data/test/test_set/100.png", cv2.IMREAD_GRAYSCALE)
print(img[tf.newaxis,...].shape)
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

STEPS_PER_EPOCH = 88
EPOCHS = 50
VALIDATION_STEPS = 24
save_model = tf.keras.callbacks.ModelCheckpoint("unet_2.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')

hist = model.fit(
    train_set,
    epochs = EPOCHS,
    steps_per_epoch = STEPS_PER_EPOCH,
    validation_data = test_set,
    validation_steps = VALIDATION_STEPS,
    callbacks = [save_model, early]
)

img = cv2.imread("/content/data/test/test_set/162.png", cv2.IMREAD_GRAYSCALE)
saved_model = tf.keras.models.load_model("unet_2.h5")
p = saved_model.predict(img[tf.newaxis,...,tf.newaxis])
print(np.amin(p))
#ret, p = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
print("test image")
plt.imshow(img, cmap = 'gray')
plt.show()
print("actual mask")
plt.imshow(cv2.imread("/content/data/test_labels/test_labels/162.png", cv2.IMREAD_GRAYSCALE).squeeze(), cmap = 'gray')
plt.show()
print("prediction")
plt.imshow(p.squeeze(), cmap = 'gray')
plt.show()

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(27)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

