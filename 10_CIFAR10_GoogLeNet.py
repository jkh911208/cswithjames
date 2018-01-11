
# coding: utf-8

# In[1]:


# import pacakes and layer
import keras
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate
from keras.optimizers import Adam


# In[2]:


# backend
import tensorflow as tf
from keras import backend as k

# Don't pre-allocate memory; allocate as-needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))


# In[3]:


# Hyperparameters
batch_size = 128
num_classes = 10
epochs = 1000


# In[4]:


def add_module(input):
    #print(input.shape)
    
    Conv2D_reduce = Conv2D(16, (1,1), strides=(2,2), activation='relu', padding='same')(input)
    #print(Conv2D_reduce.shape)
    
    Conv2D_1_1 = Conv2D(16, (1,1), activation='relu', padding='same')(input)
    #print(Conv2D_1_1.shape)
    Conv2D_3_3 = Conv2D(16, (3,3),strides=(2,2), activation='relu', padding='same')(Conv2D_1_1)
    #print(Conv2D_3_3.shape)
    Conv2D_5_5 = Conv2D(16, (5,5),strides=(2,2), activation='relu', padding='same')(Conv2D_1_1)
    #print(Conv2D_5_5.shape)
    
    MaxPool2D_3_3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(input)
    #print(MaxPool2D_3_3.shape)
    Cov2D_Pool = Conv2D(16, (1,1), activation='relu', padding='same')(MaxPool2D_3_3)
    #print(Cov2D_Pool.shape)
    
    concat = Concatenate(axis=-1)([Conv2D_reduce,Conv2D_3_3,Conv2D_5_5,Cov2D_Pool])
    #print(concat.shape)
    
    return concat


# In[5]:


# Load CIFAR10 Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

# convert to one hot encoing 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[6]:


# GoogLeNet Define, not using the sequential model, becuase googlenet is not sequential
input = Input(shape=(img_height, img_width, channel,))

Conv2D_1 = Conv2D(64, (3,3), activation='relu', padding='same')(input)
MaxPool2D_1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(Conv2D_1)
BatchNorm_1 = BatchNormalization()(MaxPool2D_1)

Module_1 = add_module(BatchNorm_1)
Module_1 = add_module(Module_1)

Output = Flatten()(Module_1)
Output = Dense(num_classes, activation='softmax')(Output)


# In[7]:


model = Model(inputs=[input], outputs=[Output])
#model.summary()


# In[8]:


# determine Loss function and Optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


# In[9]:


model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))


# Epoch 984/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0555 - acc: 0.9835 - val_loss: 3.9292 - val_acc: 0.6711
# Epoch 985/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0521 - acc: 0.9853 - val_loss: 3.9071 - val_acc: 0.6746
# Epoch 986/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0450 - acc: 0.9866 - val_loss: 3.9003 - val_acc: 0.6766
# Epoch 987/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0516 - acc: 0.9853 - val_loss: 3.8423 - val_acc: 0.6795
# Epoch 988/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0527 - acc: 0.9858 - val_loss: 3.9306 - val_acc: 0.6731
# Epoch 989/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0487 - acc: 0.9851 - val_loss: 3.8229 - val_acc: 0.6762
# Epoch 990/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0553 - acc: 0.9846 - val_loss: 3.9366 - val_acc: 0.6725
# Epoch 991/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0571 - acc: 0.9845 - val_loss: 4.0078 - val_acc: 0.6729
# Epoch 992/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0414 - acc: 0.9882 - val_loss: 3.9755 - val_acc: 0.6731
# Epoch 993/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0425 - acc: 0.9873 - val_loss: 3.9870 - val_acc: 0.6700
# Epoch 994/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0460 - acc: 0.9865 - val_loss: 3.9641 - val_acc: 0.6680
# Epoch 995/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0514 - acc: 0.9860 - val_loss: 3.9267 - val_acc: 0.6729
# Epoch 996/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0476 - acc: 0.9863 - val_loss: 3.8823 - val_acc: 0.6785
# Epoch 997/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0533 - acc: 0.9853 - val_loss: 3.9038 - val_acc: 0.6756
# Epoch 998/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0369 - acc: 0.9893 - val_loss: 3.9139 - val_acc: 0.6745
# Epoch 999/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0629 - acc: 0.9833 - val_loss: 3.9070 - val_acc: 0.6746
# Epoch 1000/1000
# 50000/50000 [==============================] - 6s 125us/step - loss: 0.0658 - acc: 0.9821 - val_loss: 3.8867 - val_acc: 0.6738
