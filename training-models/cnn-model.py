# importing dependencies
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# training and testing dataset directories path
TRAIN_DATA_PATH = 'X-ray Images/train'
TEST_DATA_PATH = 'X-ray Images/test'
VALID_DATA_Path = 'X-ray Images/validation'
# cnn-model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# compiling the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(TRAIN_DATA_PATH, target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_set = test_datagen.flow_from_directory(VALID_DATA_Path, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Train the model
history = model.fit(training_set, epochs=10, validation_data=validation_set)

