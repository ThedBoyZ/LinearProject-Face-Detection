from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization,ZeroPadding2D, Input, Activation
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
num_classes = 5
img_rows, img_cols = 224, 224

vgg = ResNet50(include_top=False, input_shape=(img_rows, img_cols, 3))

for layer in vgg.layers:
    layer.trainable = False

def build_model(pre_model):
    f1 = Flatten()(pre_model.output)
    d1 = Dense(1024, activation='relu')(f1)
    d2 = Dense(512, activation='relu')(d1)
    d3 = Dense(256, activation='relu')(d2)
    d4 = Dense(128, activation='relu')(d3)
    d5 = Dense(64, activation='relu')(d4)
    d6 = Dense(32, activation='relu')(d5)
    output = Dense(num_classes, activation='softmax')(d6)
    return Model(inputs=pre_model.input, outputs=output)

face_recog_model = build_model(vgg)
face_recog_model.compile(

print(face_recog_model.summary())