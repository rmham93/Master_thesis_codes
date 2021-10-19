# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:04:28 2021

@author: Fatma Ridaoui
"""

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image as im

images =[]
images = np.array(images)
images = np.array(np.load('data.npy', allow_pickle=True)) # load

y = ["casual", "ethnic", "formal", "party", "smartcasual", "sport", "travel"]

X_train, X_test, y_train, y_test = train_test_split(images, y, train_size=0.75)

#plotting original and gray. image "CTRL+1"
import matplotlib.pyplot as plt

print((np.array(X_train[0][0])).shape)

original = im.fromarray(np.array(X_train[2][5]))

plt.imshow(original)

print(y_train[2])
