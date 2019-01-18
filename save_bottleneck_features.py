
from keras import applications
from keras.preprocessing.image import ImageDataGenerator, load_img
##from keras import Optimizers
from keras.models import Sequential 
from keras.layers import Dropout, Flatten, Dense 
from keras.applications import VGG16
from keras import models
##from keras import optimizers 

import numpy as np 
import matplotlib.pyplot as plt 

input_shape = (128, 128,3) 

train_dir = 'data/train'
validation_dir = 'data/validation'
top_model_weights_path = 'top_model.h5'

n_train = 1200
n_val = 300
batch_size = 10 


def save_features():
	datagen = ImageDataGenerator(rescale = 1./255)
	
	train_features = np.zeros(shape = (n_train, 4,4, 512))
	train_labels = np.zeros(shape = (n_train,6))

	vgg_conv = VGG16(weights = 'imagenet', include_top = False, input_shape = input_shape)

	train_generator = datagen.flow_from_directory(train_dir, 
		target_size = (128,128),
		color_mode = 'rgb',
		batch_size = batch_size,
		class_mode = 'categorical',
		shuffle = True)

	

	i = 0
	for inputs_batch, labels_batch in train_generator:
		features_batch = vgg_conv.predict(inputs_batch, verbose =1)
		train_features[i*batch_size: (i+1)*batch_size] = features_batch
		train_labels[i*batch_size: (i+1)*batch_size] = labels_batch
		print(i) 
		i += 1 
		if i*batch_size >= n_train:
			break

	train_features = np.reshape(train_features, (n_train, 4*4*512))
	np.save(open('train_features.npy', 'wb'), train_features)
	np.save(open('train_labels.npy', 'wb'), train_labels)

	validation_features = np.zeros(shape = (n_val, 4,4,512))
	validation_labels = np.zeros(shape = (n_val,6))

	validation_generator = datagen.flow_from_directory(validation_dir, 
		target_size = (128,128),
		color_mode = 'rgb',
		batch_size = batch_size, 
		class_mode = 'categorical',
		shuffle = False) 

	i = 0
	for inputs_batch, labels_batch in validation_generator:
		features_batch = vgg_conv.predict(inputs_batch, verbose =1)
		validation_features[i*batch_size: (i+1)*batch_size] = features_batch
		validation_labels[i*batch_size: (i+1)* batch_size] = labels_batch
		print(i) 
		i += 1 
		if i*batch_size >= n_val:
			break 
		

	validation_features = np.reshape(validation_features, (n_val, 4*4*512))
	np.save(open('validation_features.npy', 'wb'), validation_features)
	np.save(open('validation_labels.npy', 'wb'), validation_labels)

save_features() 





		
	
