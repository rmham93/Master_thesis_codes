# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:10:41 2021

@author: Fatma Ridaoui
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

new_model = tf.keras.models.load_model('saved_model/my_model')

new_model.summary()


im_datagen = ImageDataGenerator(rescale=1./255)

seed=1

test_gen = im_datagen.flow_from_directory(
    'C:/Users/IDU/OneDrive - GTÜ/Desktop/TEZ/modanisa_test/',
    class_mode='categorical',
     target_size=(100,100),
    seed=seed)

loss, acc = new_model.evaluate(test_gen, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

result = new_model.predict(test_gen)

print(result)
