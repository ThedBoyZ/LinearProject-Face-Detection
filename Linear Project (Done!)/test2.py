import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
# Load your image
image = cv2.imread('profile.jpg')
image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# create data augmention generator
datagen = ImageDataGenerator(
    rotation_range=40,  # randomly rotate images in the range 0-40 degrees
    zoom_range=0.1,  # Randomly zoom image 10%
    width_shift_range=0.1,  # randomly shift images horizontally 10%
    height_shift_range=0.1,  # randomly shift images vertically 10%
)


path = 'dataset/'
save_here = 'dataset/validation_data/'

train_generator = datagen.flow_from_directory(
    path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print(train_generator.class_indices)


for batch in train_generator:
    print(batch[0].shape)
    print(batch[1].shape)
    break



# Load your image
# image = cv2.imread('profile.jpg')
# image = cv2.resize(image, (224, 224))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Reshape your image
# image = image.reshape((1,) + image.shape)

# # Generate batches of augumented images from this image
# save_prefix = 'aug_profile'
# i = 0
# for batch in datagen.flow(image, save_to_dir=save_here, save_prefix=save_prefix, save_format='jpg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely

