import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Dense, GlobalAveragePooling2D,Flatten,Reshape
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

BATCH_SIZE = 32
num_classes =3

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, list_pose, labeldf, batch_size= BATCH_SIZE, dim=(32,32,32), n_channels= 1,n_classes=3, shuffle= True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_pose = list_pose
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.labeldf =labeldf
        self.on_epoch_end()
        self.n = 0
    
    def __len__(self):
        
        return int(np.floor(len(self.list_pose) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_pose_temp = [self.list_pose[k] for k in indexes]

        i,p,l = self.__data_generation(list_pose_temp)
        #print("here", i.shape,p.shape,l.shape)
        return [i, p], l #image pose label

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_pose))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

    def __data_generation(self, list_pose_temp):   
        X = np.empty((self.batch_size, 66,1))
        Z = np.empty((self.batch_size, 135,100, 3))
        l = np.empty((self.batch_size, self.n_classes ))
        
        for i, each in enumerate(list_pose_temp):          
            
            X[i,] =np.reshape(np.load('C:/Users/IDU/OneDrive - GTÜ/Desktop/TEZ/poses/NPdata/'+ each ,allow_pickle=True),(66,1))
            image = Image.open('C:/Users/IDU/Desktop/dataset/'+ each.split(".")[0] +".jpg")
            image = np.array((image.convert('RGB')).resize((100, 135)))
          
            Z[i,] = image / 255.0
            label = self.labeldf['category'][self.labeldf['productid']== int(each.split(".")[0] )].values[0]
            if label =='Takım':
                lab = [0,1,0]
            elif label =='Tesettür Elbise':
                lab = [1,0,0]
            else:
                lab = [0,0,1]
                
                
                
                
            l[i,]= lab
        
        
        return Z,X,l
    


params = {'dim': (32,32,32),
          'batch_size': BATCH_SIZE,
          'n_classes': 3,
          'n_channels': 1,""
          'shuffle': True}

df = pd.read_csv(r"C:\Users\IDU\OneDrive - GTÜ\Desktop\TEZ\MODANISA\prod_details.csv")



poselist = np.load("C:/Users/IDU/OneDrive - GTÜ/Desktop/TEZ/poses/poselist.npy", allow_pickle=True )        

labels =np.load(r"C:\Users\IDU\OneDrive - GTÜ\Desktop\TEZ\poses\labels_1.npy",  allow_pickle=True)
labels = labels.reshape(-1, 1)
enc = OneHotEncoder(handle_unknown='ignore').fit(labels)
labels = enc.transform(labels).toarray()

#print("debug",df['category'][df['productid']== int(8010707 )])

# Generators
pose_generator = DataGenerator(poselist[:25000],df, **params)
val_generator = DataGenerator(poselist[25001:],df, **params)

input_shape=(135,100,3)

pose_shape = (66,1)

model = ResNet50( include_top= False , input_shape= input_shape , classes= num_classes)


x = GlobalAveragePooling2D()(model.output)
x = Dense(256, activation ='relu')(x)
x = Dense(66, activation ='relu')(x)
x= tf.expand_dims(x, axis= -1)


pose_input = tf.keras.layers.Input(pose_shape)

concat = tf.concat([x, pose_input], axis=1 )
concat = Flatten()(concat)

#x = Dense(64, activation='relu')(concat)
x = tf.keras.layers.Dense(32, activation= 'relu')(concat)

y = tf.keras.layers.Dense(3)(x)


model = tf.keras.models.Model(inputs= [model.input, pose_input] , outputs= y)

model.compile( optimizer= "adam", loss='categorical_crossentropy', metrics= ['accuracy'] )

#print(model.summary())
history= model.fit(pose_generator, validation_data=val_generator,
          batch_size=BATCH_SIZE,
          epochs= 4)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('model_accuracy.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('model_loss.png')

model.save('model_poseResnet.h5')





