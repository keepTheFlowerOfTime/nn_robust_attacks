## setup_cifar.py -- cifar data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from Crypto.Cipher import AES
from .index_helper import color_grad,threhold,focus,shuffle,draft

_Key='test123112345612'.encode('utf-8')

def load_data(fpath,label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
    final[:,:,:,0] = data[:,0,:,:]
    final[:,:,:,1] = data[:,1,:,:]
    final[:,:,:,2] = data[:,2,:,:]
    return final,labels

# def load_batch(fpath, label_key='labels'):
#     final,labels=load_data(fpath)

#     final /= 255
#     final -= .5
#     labels2 = np.zeros((len(labels), 10))
#     labels2[np.arange(len(labels2)), labels] = 1

#     return final, labels
_Rule=None
_Seed=123456789 #987654321
def load_batch(fpath,pre_fix='n',args=None):
    #Normal,White,Grad,Threhold,Shuffle,Draft
    #
    if args is None: args={}
    def fix_image(img,fix='n'):
        global _Rule
        if fix=='n':
            return (img/255)-.5
        elif fix=='w':
            std=np.std(img)
            u=np.average(img)
            return (img-u)/std
        elif fix=='g':
            return color_grad(img)[0]
        elif fix=='t':
            return focus(img,8,1)
        elif fix=='s':
            img,rule=shuffle(img,_Rule,CifarFix.Generator)
            _Rule=rule
            return img
        elif fix=='d':
            ratio,mode=args.get('d',[.5,'b'])
            return draft(img,ratio,mode)[0]
        else:
            return img
    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))
        labels.append(lab)
        for c in pre_fix:
            img=fix_image(img,c)
        images.append(img)
    return np.array(images),np.array(labels)
    
def load_batch_and_encrypt(fpath,key=_Key,label_key='labels'):

    def encrypt(image):
        crypto=AES.new(key,AES.MODE_ECB)
        shape=image.shape
        vec=image.reshape([-1])
        vec=crypto.encrypt(bytearray(vec))
        vec=[int(x) for x in vec]

        t=np.array(vec).reshape(shape)

        return t

    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((encrypt(img)/255)-.5)
    return np.array(images),np.array(labels)


class CIFAR:
    def __init__(self):
        train_data = []
        train_labels = []
        
        if not os.path.exists("cifar-10-batches-bin"):
            urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                       "cifar-data.tar.gz")
            os.popen("tar -xzf cifar-data.tar.gz").read()
            

        for i in range(5):
            r,s = load_batch("cifar-10-batches-bin/data_batch_"+str(i+1)+".bin")
            train_data.extend(r)
            train_labels.extend(s)
            
        train_data = np.array(train_data,dtype=np.float32)
        train_labels = np.array(train_labels)
        
        self.test_data, self.test_labels = load_batch("cifar-10-batches-bin/test_batch.bin")
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

class CifarFix:
    Generator=None
    """
    Normal,White,Grad,Threhold,Shuffle
    """
    def __init__(self,train_mode='n',test_mode=None,need_train_data=True,need_verify_data=True,seed=_Seed,args=None):
        global _Rule
        _Rule=None
        self.args=args
        CifarFix.Generator=np.random.default_rng(seed=seed)
        if test_mode==None:
            test_mode=train_mode
        train_data = []
        train_labels = []
        
        if not os.path.exists("cifar-10-batches-bin"):
            urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                       "cifar-data.tar.gz")
            os.popen("tar -xzf cifar-data.tar.gz").read()
            
        if need_train_data:
            for i in range(5):
                r,s = load_batch("cifar-10-batches-bin/data_batch_"+str(i+1)+".bin",train_mode,self.args)
                train_data.extend(r)
                train_labels.extend(s)
                print('{}/50000'.format(10000*(i+1)))

            train_data = np.array(train_data,dtype=np.float32)
            train_labels = np.array(train_labels)

            VALIDATION_SIZE = 5000
        
            self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
            self.validation_labels = train_labels[:VALIDATION_SIZE]
            self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
            self.train_labels = train_labels[VALIDATION_SIZE:]
            

        if need_verify_data:       
            self.test_data, self.test_labels = load_batch("cifar-10-batches-bin/test_batch.bin",test_mode,self.args)
        
        

class CIFARModel:
    def __init__(self, restore, session=None):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10

        model = Sequential()

        model.add(Conv2D(64, (3, 3),
                                input_shape=(32, 32, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(10))

        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)
        
class CIFARModel_express:
    def __init__(self, restore, session=None):
        self.num_channels = 1
        self.image_size = 32
        self.num_labels = 10

        model = Sequential()

        model.add(Conv2D(16, (3, 3),
                                input_shape=(32, 32, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(10))

        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)

    @staticmethod
    def params():
        return [16,16,32,32,64,64]

