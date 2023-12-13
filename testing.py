import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
classifier = tf.keras.models.load_model("saved_final_model/my_model")
img_size=48
batch_size=64
datagen_train=ImageDataGenerator(horizontal_flip=True)
train_generator=datagen_train.flow_from_directory(r"C:\Users\jaswanth\OneDrive\Desktop\plant disease detection\dataset\train",target_size=(48,48), batch_size=batch_size,class_mode='categorical', shuffle=True)
datagen_validation=ImageDataGenerator(horizontal_flip=True)
validation_generator=datagen_train.flow_from_directory(r"C:\Users\jaswanth\OneDrive\Desktop\plant disease detection\dataset\test",target_size=(48,48), batch_size=batch_size,class_mode='categorical', shuffle=True)
import numpy as np
path = r"C:\Users\jaswanth\OneDrive\Desktop\plant disease detection\dataset\test\Tomato___Spider_mites Two-spotted_spider_mite\0ade19e4-c48b-4e58-bddf-153d44d48c3e___Com.G_SpM_FL 9449.JPG"
test_image = tf.keras.utils.load_img(path)
import matplotlib.pyplot as plt
plt.show()
test_img = tf.keras.utils.load_img(path, target_size=(48,48))
test_img =tf.keras.utils.img_to_array(test_img)
test_img = np.expand_dims(test_img,axis=0)
result = classifier.predict(test_img)
a = result.argmax()
s = train_generator.class_indices
name = [ ]
for i in s:
 name.append(i)
for i in range(len(s)):
 if (i==a):
  p=name[i]
  print (p)