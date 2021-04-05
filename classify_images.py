import os
import numpy as np

import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


MODEL_NAME = 'CarClass-3'
base_dir = "D:/Data Warehouse/thecarconnection/pictures"
count = 0
confidence_treshold = 0.7

# %%
# Load Saved Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~
load_name = 'models/{}'.format(MODEL_NAME)
model = tf.keras.models.load_model(load_name)
model.summary()

print('We have loaded a previous model!!!!')


# %%
# Load and Preprocess Images
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# if not os.path.exists(destination_dir):
#     os.makedirs(destination_dir)

def save_image(image, base_dir, category):
    '''
    Save input image in category folder inside base_dir
    '''
    destination_dir = os.path.join(base_dir, category)
    if os.path.isdir(destination_dir):
        pass
    else:
        os.makedirs(os.path.join(base_dir, category))

    original_file_path = os.path.join(base_dir, image)
    destination_file_path = os.path.join(destination_dir, image)
    os.replace(original_file_path, destination_file_path)

    return

for file in os.listdir(base_dir):
    img_path = os.path.join(base_dir, file)
    if os.path.isdir(img_path):
        # skip directories
        continue
    else:
        im = image.load_img(img_path, target_size=(150, 150), color_mode="rgb")
        im = image.img_to_array(im)
        im = im / 255.0
        im = np.expand_dims(im, axis=0)
        count += 1

        # %%
        # Predict to which category belongs each image
        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        prediction = model.predict(im)
        category = tf.argmax(prediction, 1)[0]
        confidence = np.amax(prediction)
        if count == 1 or count % 100 == 0:
            remaining = len(os.listdir(base_dir))

        # Save images into folders depending on their category
        print('\ncategory: {}, confidence: {}, remaining: {}'.format(category, confidence, remaining))
        if confidence >= confidence_treshold:
            if category == 0:
                save_image(file, base_dir, 'front')
            elif category == 1:
                save_image(file, base_dir, 'front_left')
            elif category == 2:
                save_image(file, base_dir, 'front_right')
            elif category == 3:
                save_image(file, base_dir, 'left')
            elif category == 4:
                save_image(file, base_dir, 'other')
            elif category == 5:
                save_image(file, base_dir, 'rear')
            elif category == 6:
                save_image(file, base_dir, 'rear_left')
            elif category == 7:
                save_image(file, base_dir, 'rear_right')
            elif category == 8:
                save_image(file, base_dir, 'right')
        else:
            save_image(file, base_dir, 'unknown')
