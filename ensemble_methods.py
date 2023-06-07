import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image

test_image_path = 'X-ray Images/train/Pneumonia-Bacterial/Pneumonia-Bacterial (1).jpg'
img = image.load_img(test_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Load the trained models
cnn_model = load_model('Deep Learning Models/CNN-CUSTOM.h5')
vgg16_model = load_model('Deep Learning Models/Feature-Extraction-VGG16.h5')
vgg19_model = load_model('Deep Learning Models/VGG19.h5')
resnet50_model = load_model('Deep Learning Models/ResNet50.h5')

# Create input tensor
input_tensor = Input(shape=(224, 224, 3))

# Preprocess the image
img_array = img_array / 255.0  # Normalize pixel values

# Make predictions with each model
cnn_predictions = cnn_model.predict(img_array)
vgg16_predictions = vgg16_model.predict(img_array)
vgg19_predictions = vgg19_model.predict(img_array)
resnet50_predictions = resnet50_model.predict(img_array)

# Combine predictions using majority voting
ensemble_predictions = np.argmax(cnn_predictions + vgg16_predictions + vgg19_predictions + resnet50_predictions, axis=1)

class_name = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']
predicted_label = class_name[ensemble_predictions[0]]

print("Predicted Label: ", predicted_label)

img.show()
