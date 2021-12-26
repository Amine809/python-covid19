#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow import keras


# In[3]:


from tensorflow.keras.preprocessing import image_dataset_from_directory


# In[4]:


train_data_url=("E:/covid 2021/Covid19-dataset/train")
test_data_url=("E:/covid 2021/Covid19-dataset/test")


# In[5]:


train_data = image_dataset_from_directory(directory="E:/covid 2021/Covid19-dataset/train",
                                          batch_size=5,
                                          image_size=(256, 256))


# In[6]:


test_data = image_dataset_from_directory(directory="E:/covid 2021/Covid19-dataset/test",
                                          batch_size=5,
                                          image_size=(256, 256))


# In[7]:


class_names=train_data.class_names
print(class_names)


# In[8]:


class_names=train_data.class_names
print(class_names)
class_names=test_data.class_names
print(class_names)


# In[9]:


import matplotlib.pyplot as plt
import numpy as np


# In[10]:


plt.figure(figsize=(10,10))
for images, labels in train_data.take(1):
    for i in range(4):
        ax=plt.subplot(2,2,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.preprocessing import image


# In[12]:


num_classes = 3

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255,input_shape=(256, 256, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='sigmoid'),
  layers.Dense(num_classes)
])


# In[13]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[14]:


model.summary()


# In[15]:


epochs=10
history = model.fit(
  train_data,
  validation_data=test_data,
  epochs=epochs
)


# In[17]:


test_img="E:/covid 2021/Covid19-dataset/test/Covid/094.png"
img=keras.preprocessing.image.load_img(test_img, target_size=(256, 256))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])


# In[18]:


print(score)


# In[19]:


print(
    "Image looking {} at most {:.2f} %"
    .format(class_names[np.argmax(score)], 100 * np.max(score)))


# In[22]:


test_img="E:/covid 2021/Covid19-dataset/test/Covid/098.jpeg"
img=keras.preprocessing.image.load_img(test_img, target_size=(256, 256))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])


# In[23]:


print(score)


# In[24]:


print(
    "Image looking {} at most {:.2f} %"
    .format(class_names[np.argmax(score)], 100 * np.max(score)))


# In[25]:


test_img="E:/covid 2021/Covid19-dataset/test/Normal/0106.jpeg"
img=keras.preprocessing.image.load_img(test_img, target_size=(256, 256))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])


# In[26]:


print(score)


# In[27]:


print(
    "Image looking {} at most {:.2f} %"
    .format(class_names[np.argmax(score)], 100 * np.max(score)))


# In[29]:


test_img="E:/covid 2021/Covid19-dataset/test/Viral Pneumonia/0103.jpeg"
img=keras.preprocessing.image.load_img(test_img, target_size=(256, 256))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])


# In[30]:


print(score)


# In[31]:


print(
    "Image looking {} at most {:.2f} %"
    .format(class_names[np.argmax(score)], 100 * np.max(score)))


# In[ ]:




