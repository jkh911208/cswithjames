
# coding: utf-8

# In[1]:


import keras
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate
from keras.optimizers import Adam


# In[2]:


# this part will prevent tensorflow to allocate all the avaliable GPU Memory
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
epochs = 300
l = 10
num_filter = 20


# In[4]:


# Load CIFAR10 Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

# convert to one hot encoing 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[5]:


# Dense Block
def add_denseblock(input):
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        Conv2D_3_3 = Conv2D(num_filter, (3,3), use_bias=False ,padding='same')(relu)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        temp = concat
        
    return temp


# In[6]:


def add_transition(input):
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(num_filter, (1,1), use_bias=False ,padding='same')(relu)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    
    return avg


# In[7]:


def output_layer(input):
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    output = Dense(num_classes, activation='softmax')(flat)
    
    return output


# input = Input(shape=(img_height, img_width, channel,))
# First_Conv2D = Conv2D(num_filter, (3,3), use_bias=False ,padding='same')(input)
# 
# First_Block = add_denseblock(First_Conv2D)
# First_Transition = add_transition(First_Block)
# 
# Last_Block = add_denseblock(First_Transition)
# 
# output = output_layer(Last_Block)
# 
# '''
# Epoch 50/50
# loss: 0.0460 - acc: 0.9848 - val_loss: 1.5572 - val_acc: 0.7365
# '''

# input = Input(shape=(img_height, img_width, channel,))
# First_Conv2D = Conv2D(num_filter, (3,3), use_bias=False ,padding='same')(input)
# 
# First_Block = add_denseblock(First_Conv2D)
# First_Transition = add_transition(First_Block)
# 
# Second_Block = add_denseblock(First_Transition)
# Second_Transition = add_transition(Second_Block)
# 
# Last_Block = add_denseblock(Second_Transition)
# output = output_layer(Last_Block)
# 
# 
# # Epoch 50/50
# # loss: 0.0611 - acc: 0.9784 - val_loss: 1.1269 - val_acc: 0.7881

# In[8]:


input = Input(shape=(img_height, img_width, channel,))
First_Conv2D = Conv2D(num_filter, (3,3), use_bias=False ,padding='same')(input)

First_Block = add_denseblock(First_Conv2D)
First_Transition = add_transition(First_Block)

Second_Block = add_denseblock(First_Transition)
Second_Transition = add_transition(Second_Block)

Third_Block = add_denseblock(Second_Transition)
Third_Transition = add_transition(Third_Block)

Last_Block = add_denseblock(Third_Transition)
output = output_layer(Last_Block)

'''
Epoch 50/50
loss: 0.0695 - acc: 0.9756 - val_loss: 1.1741 - val_acc: 0.7874
'''


# In[10]:


model = Model(inputs=[input], outputs=[output])
model.summary()


# In[11]:


# determine Loss function and Optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))


# In[ ]:


# Test the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

