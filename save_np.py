# -*- coding: utf-8 -*-

from PIL import Image
import glob
import numpy as np
import pickle
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

images= []

casual_list = []
for filename in glob.glob('C:/Users/IDU/OneDrive - GTÜ/Desktop/TEZ/smalldata/casual/*.jpg'): #assuming gif
    im=Image.open(filename).convert('LA')
    im = im.resize((50,50))
    casual_list.append(im)
    plt.imshow(casual_list[0])
    
#print(len(casual_list[0]))
images.append(casual_list)

#plt.imshow( img.fromarray(casual_list[0]))

#plotting original and gray. image "CTRL+1"
#import matplotlib.pyplot as plt

# original = casual_list[0]
# grayscale = rgb2gray(original)

# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# ax = axes.ravel()

# ax[0].imshow(original)
# ax[0].set_title("Original")
# ax[1].imshow(grayscale, cmap=plt.cm.gray)
# ax[1].set_title("Grayscale")

# fig.tight_layout()
# plt.show()


ethnic_list = []
for filename in glob.glob('C:/Users/IDU/OneDrive - GTÜ/Desktop/TEZ/smalldata/ethnic/*.jpg'): #assuming gif
    im=Image.open(filename).convert('LA')
    im = im.resize((50,50))
    ethnic_list.append(im)
#print("Ethnic ", len(ethnic_list))

images.append(ethnic_list)


formal_list = []
for filename in glob.glob('C:/Users/IDU/OneDrive - GTÜ/Desktop/TEZ/smalldata/formal/*.jpg'): #assuming gif
    im=Image.open(filename).convert('LA')
    im = im.resize((50,50))
    formal_list.append(im)
#print("Ethnic ", len(ethnic_list))

images.append(formal_list)

party_list = []
for filename in glob.glob('C:/Users/IDU/OneDrive - GTÜ/Desktop/TEZ/smalldata/party/*.jpg'): #assuming gif
    im=Image.open(filename).convert('LA')
    im = im.resize((50,50))
    party_list.append(im)
#print("Ethnic ", len(ethnic_list))

images.append(party_list)

smartcasual_list =[]
for filename in glob.glob('C:/Users/IDU/OneDrive - GTÜ/Desktop/TEZ/smalldata/smartcasual/*.jpg'): #assuming gif
    im = Image.open( filename ).convert('LA')
    im = im.resize((50,50))
    smartcasual_list.append(im)
#print("Ethnic ", len(ethnic_list))

images.append(smartcasual_list)


sport_list =[]
for filename in glob.glob('C:/Users/IDU/OneDrive - GTÜ/Desktop/TEZ/smalldata/sport/*.jpg'): #assuming gif
    im = Image.open( filename ).convert('LA')
    im = im.resize((50,50))
    sport_list.append(im)
#print("Ethnic ", len(ethnic_list))

images.append(sport_list)

travel_list =[]
for filename in glob.glob('C:/Users/IDU/OneDrive - GTÜ/Desktop/TEZ/smalldata/travel/*.jpg'): #assuming gif
    im = Image.open( filename ).convert('LA')
    im = im.resize((50,50))
    travel_list.append(im)
#print("Ethnic ", len(ethnic_list))

images.append(travel_list)

np.save('data.npy', images) # save
#new_num_arr = np.load('data.npy') # load









