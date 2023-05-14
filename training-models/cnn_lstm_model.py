# import necessary libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, LSTM, Dropout, Flatten, Reshape
from keras.models import Sequential

# set up the data generators
train_dir = 'X-ray Images/train'
val_dir = 'X-ray Images/validation'
test_dir = 'X-ray Images/test'
img_height = 224
img_width = 224
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# load the pre-trained VGG16 model and extract features from the images
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# freeze the layers in the VGG16 model
for layer in vgg_model.layers:
    layer.trainable = False

# create the CNN-LSTM model
model = Sequential()
model.add(vgg_model)
model.add(Flatten())
model.add(Reshape((1, -1)))
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_generator.num_classes, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.n // batch_size)

# evaluate the model on the test data
test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.n // batch_size)
print('Test accuracy:', test_acc)

# make predictions on new data
predictions = model.predict_generator(test_generator, steps=test_generator.n // batch_size)
