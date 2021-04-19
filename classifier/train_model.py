import os
import zipfile
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


MODEL_NAME = 'CarClass-3'
base_dir = "D:/Data Warehouse/thecarconnection"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training pictures
train_front_dir = os.path.join(train_dir, 'front')
train_rear_dir = os.path.join(train_dir, 'rear')
train_front_left_dir = os.path.join(train_dir, 'front_left')
train_front_right_dir = os.path.join(train_dir, 'front_right')
train_left_dir = os.path.join(train_dir, 'left')
train_right_dir = os.path.join(train_dir, 'right')
train_rear_left_dir = os.path.join(train_dir, 'rear_left')
train_rear_right_dir = os.path.join(train_dir, 'rear_right')
train_other_dir = os.path.join(train_dir, 'other')

# Directory with our validation pictures
validation_front_dir = os.path.join(validation_dir, 'front')
validation_rear_dir = os.path.join(validation_dir, 'rear')
validation_front_left_dir = os.path.join(validation_dir, 'front_left')
validation_front_right_dir = os.path.join(validation_dir, 'front_right')
validation_left_dir = os.path.join(validation_dir, 'left')
validation_right_dir = os.path.join(validation_dir, 'right')
validation_rear_left_dir = os.path.join(validation_dir, 'rear_left')
validation_rear_right_dir = os.path.join(validation_dir, 'rear_right')
validation_other_dir = os.path.join(validation_dir, 'other')


# Adding rescale, rotation_range, width_shift_range, height_shift_range,
# shear_range, zoom_range, and horizontal flip to our ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    # horizontal_flip=True,
    )

# Note that the validation data should not be augmented!
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

# Flow validation images in batches of 32 using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Convolution2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(9, activation='softmax')(x)

# Configure and compile the model
model = Model(img_input, output)
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)

print('\nSAVING MODEL!\n')
save_name = 'models/{}'.format(MODEL_NAME)
model.save(save_name)
