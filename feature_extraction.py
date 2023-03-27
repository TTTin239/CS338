from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.losses import CosineSimilarity
import numpy as np

model = VGG16(weights='imagenet', include_top=False)


def extract_feature(img_path):
    img = image.load_img(img_path, target_size=(244,244))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    return  features

feature1 = extract_feature('./ronaldo1.jpg')
feature4 = extract_feature('./ronaldo2.jpg')

consine_loss = CosineSimilarity()
print(consine_loss(feature4, feature4).numpy())
