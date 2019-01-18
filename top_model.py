import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications 

batch_size = 10
top_model_weights_path = 'top_model.h5'

def train_top_model():

	train_data = np.load(open('train_features.npy', 'rb'))
	train_labels = np.load(open('train_labels.npy', 'rb'))

	validation_data = np.load(open('validation_features.npy', 'rb'))
	validation_labels = np.load(open('validation_labels.npy', 'rb'))

	model = Sequential()
	model.add(Dense(256, activation = 'relu', input_shape = train_data.shape[1:]))
	model.add(Dropout(0.3))
	model.add(Dense(6, activation = 'softmax'))

	model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

	history = model.fit(train_data, train_labels, epochs = 10, batch_size = batch_size, validation_data = (validation_data, validation_labels))
	model.save_weights(top_model_weights_path)

train_top_model() 

