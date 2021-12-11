from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout
from keras.layers.pooling import GlobalMaxPooling2D
from skimage.io import imread
from skimage.transform import resize
from pathlib import Path
from keras import Model
from keras.metrics import CategoricalAccuracy,TruePositives,TrueNegatives,FalseNegatives,FalsePositives
import numpy as np


#Training Data
train_x = list()
train_y = list()
j=0
for directory in ['Airplanes','Bikes','Cars','Faces']:
    for path in Path(f'./Training/{directory}').iterdir():
        train_x.append(resize(imread(path),(224,224)))
        y = np.zeros(4)
        y[j]=1
        train_y.append(y)
    j+=1
train_x = np.array(train_x)
train_y = np.array(train_y)

#Testing Data
test_x = list()
test_y = list()
j=0
for directory in ['Airplanes','Bikes','Cars','Faces']:
    for path in Path(f'./Testing/{directory}').iterdir():
        test_x.append(resize(imread(path),(224,224)))
        y = np.zeros(4)
        y[j]=1
        test_y.append(y)
    j+=1
test_x = np.array(test_x)
test_y = np.array(test_y)

#Import VGG16
vgg = VGG16(include_top=False,weights="imagenet")
#Freeze VGG16 layers from training
for layer in vgg.layers:
    layer.trainable = False

#NN classifier
x2 = GlobalMaxPooling2D()(vgg.get_layer('block5_pool').output)
x3 = Dense(512, activation='relu')(x2)
x4 = Dropout(0.5)(x3)
out = Dense(4,activation='softmax')(x4)

model = Model(vgg.input,out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[CategoricalAccuracy(),TruePositives(),FalsePositives(),TrueNegatives(),FalseNegatives()])

#Training
classifier = model.fit(train_x,train_y,epochs=5)

#Testing
evaluation = model.evaluate(x=test_x,y=test_y)
print(evaluation)