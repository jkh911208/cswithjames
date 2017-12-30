import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Hyperparameters
batch_size = 500
num_classes = 10
epochs = 10

# Load Mnist Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_height, img_width = x_train.shape[1],x_train.shape[2]

# convert to one hot encoing 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Flatten the image
x_train = x_train.reshape(x_train.shape[0], img_height*img_width)
x_test = x_test.reshape(x_test.shape[0], img_height*img_width)

# Define the Model
model = Sequential()
model.add(Dense(1024, activation='sigmoid', input_shape=(784,)))
model.add(Dense(1024, activation='sigmoid'))
# 32 -> 1024 Neurons, Wider Network
model.add(Dense(num_classes, activation='softmax'))

# print the model summary
model.summary()

# determine Loss function and Optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Train the Model
model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Test the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
