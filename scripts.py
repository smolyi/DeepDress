# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:01:10 2020

@author: ilans
"""


'''
srcdir = './'
filesdir = "img/fashion-iq"
loss_func = 'binary_crossentropy' #'mean_squared_error'
modelfilepath = "model.h5"


protoPath = os.path.join(srcdir,"HED/deploy.prototxt")
modelPath = os.path.join(srcdir,"HED/hed_pretrained_bsds.caffemodel")
loading_batch_size = 1024
print(loading_batch_size,srcdir,filesdir,protoPath,modelPath)
'''
from mergeGen import mergeiter
import argparse
import os,cv2
import tensorflow as tf
import glob
from scipy import ndimage

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense,Reshape, Dropout, Input, Flatten,AveragePooling2D,UpSampling2D
from keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model,Sequential, load_model
from keras.optimizers import Adam,RMSprop,SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping,ReduceLROnPlateau
import keras.backend as K

parser = argparse.ArgumentParser()
parser.add_argument("img_dir", help="path to folder containing images")
parser.add_argument("--output_dir", help="where to put output files",default='./output')
parser.add_argument('--hed', help='path to HED files(prototxt,caffemodel)',default='HED')
parser.add_argument("--modelfile", type=str, help="path to model file (if not specified - create new one")
parser.add_argument("--epochs", type=int, default=1, help="number of training epochs (per batch)")
parser.add_argument("--batch_size", type=int, default=1024, help="number of images in batch")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--cvsplit", type=float, default=0.2, help="cross-validation split")
parser.add_argument("--rounds", type=int, default=7, help="total epoch rounds on all data")
parser.add_argument('-ws',"--windowsize", type=int, default=0, help="size of window for training")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()


srcdir = './'
filesdir = a.img_dir
loss_func = 'mean_squared_error'
modelfilepath = os.path.join(a.output_dir,"model.h5")
cvsplit = a.cvsplit


protoPath = os.path.join(a.hed,"deploy.prototxt")
modelPath = os.path.join(a.hed,"hed_pretrained_bsds.caffemodel")
loading_batch_size = a.batch_size
epochsperbatch = a.epochs
rounds = a.rounds
learningrate = a.learning_rate
windowsize = a.windowsize
dim = (500,300)

#print(loading_batch_size,srcdir,filesdir,protoPath,modelPath)

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]


def get_model_autoencoder(chan = 1):
    
    input_img = Input(shape=(dim[0], dim[1], chan), name="input_img")
    bn_model = 0
    x = Conv2D(64, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')((BatchNormalization(momentum=bn_model))(input_img))
    x = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (5, 5), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same',kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded  = BatchNormalization()(x)
    
    x = UpSampling2D((2, 2))(encoded)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (5, 5), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(128, (5, 5), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform')(x)
    
    autoencoder = Model(input_img, decoded)
    
    return autoencoder


def img2input(original):
    colors = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(colors,(dim[1],dim[0]))
    blob = cv2.dnn.blobFromImage(resized )
    net.setInput(blob)
    hed = net.forward()
    aslist = list(hed.shape)
    return np.reshape(hed[0,0],aslist[2:]+[1,])

def img2output(original):
    print(original.shape)
    resized = cv2.resize(original,(dim[1],dim[0]))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    '''
    blob = cv2.dnn.blobFromImage(original )
    net.setInput(blob)
    hed = net.forward()[0,0]
    dd = cv2.cvtColor(hed, cv2.COLOR_GRAY2RGB)
    '''
    print(gray.shape)

    return gray

def load_data(i=-1):
    ws = windowsize
    input_list = [ ]
    target_list = [ ]
    hed_list = [ ]
    
    startind = i*loading_batch_size
    endind = startind + loading_batch_size
    if i<0:
        startind = 0
        endind = len(input_paths)

    for p in input_paths[startind:endind]:
        original = cv2.imread(p)
        resized = cv2.resize(original,(dim[1],dim[0]))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        blob = cv2.dnn.blobFromImage(resized, scalefactor=1.0, )
        
        net.setInput(blob)
        hed = net.forward()
        hed = hed[0,0]
        
        if ws:
            newmat = np.zeros(gray.shape)
            inds = np.argwhere(hed > (np.max(hed)*0.05) )
            for (x,y) in inds:
                newmat[x-ws:x+ws,y-ws:y+ws] = gray[x-ws:x+ws,y-ws:y+ws]
            target_list.append(newmat)
        else:
            target_list.append(gray)
            
        input_list.append(gray)
        
        hed_list.append(hed)
    return np.array([target_list,hed_list,input_list])
    
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
cv2.dnn_registerLayer("Crop", CropLayer)

input_paths = glob.glob(os.path.join(filesdir, "*.jpg"))
if a.modelfile:
    model = load_model(a.modelfile)
else:
    model = get_model_autoencoder()
optimizer = Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss=loss_func,)
checkpoint = ModelCheckpoint(modelfilepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print (model.summary() )

if loading_batch_size==0:
    print('Loading images ...')
    mat = load_data()
    print('Done.')
    print('training set shape:',mat.shape)
    newmat = np.reshape(mat,list(mat.shape)+[1,])
    print('Done.')
    print('Fitting ...',epochsperbatch,'epoches')
    model.fit(newmat[1],newmat[0],epochs=epochsperbatch,shuffle=True,validation_split=cvsplit, callbacks=callbacks_list)

elif loading_batch_size<0:
    loading_batch_size =-loading_batch_size
    num_batches = int(len(input_paths)/loading_batch_size)+ 1 
    
    datagen_in = ImageDataGenerator(  preprocessing_function = img2input, validation_split=cvsplit)
    train_in= datagen_in.flow_from_directory(filesdir, color_mode='grayscale', target_size = dim,class_mode=None,shuffle=False, batch_size=loading_batch_size,subset='training')
    val_in= datagen_in.flow_from_directory(filesdir, color_mode='grayscale', target_size = dim,class_mode=None,shuffle=False, batch_size=loading_batch_size,subset='validation')
    
    datagen_out = ImageDataGenerator(  validation_split=cvsplit)
    train_out= datagen_out.flow_from_directory(filesdir, color_mode='grayscale', target_size = dim,class_mode=None, shuffle=False,batch_size=loading_batch_size,subset='training')
    val_out= datagen_out.flow_from_directory(filesdir, color_mode='grayscale', target_size = dim,class_mode=None, shuffle=False,batch_size=loading_batch_size,subset='validation')
    
    train_gen= zip(train_in, train_out)
    val_gen= zip(val_in, val_out)

    print('Fitting ...',epochsperbatch,'epoches')    
    model.fit_generator(train_gen,validation_data=val_gen,validation_steps=num_batches,  steps_per_epoch=epochsperbatch,callbacks=callbacks_list)   

else:
    num_batches = int(len(input_paths)/loading_batch_size)+ 1 
    for r in range( rounds ):
        for i in range( num_batches):    
            print('Round',r,'out of',rounds)
            print('Batch %d/%d'%(i,num_batches))
            print('Loading images ...')
            mat = load_data(i)
            print('Done.')
            print('training set shape:',mat.shape)
            newmat = np.reshape(mat,list(mat.shape)+[1,])
            print('Fitting ...',epochsperbatch,'epoches')    
            model.fit(newmat[1],newmat[0],epochs=epochsperbatch,shuffle=True,validation_split=cvsplit, callbacks=callbacks_list)
