# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 11:04:47 2018

@author: GanJinZERO
"""

import codecs
from gensim.models import Word2Vec
import pandas as pd
import numpy
from pandas import DataFrame
import re
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, GRU, SimpleRNN
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import keras
from keras.layers.core import Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from tensorflow.python.client import device_lib as _device_lib
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras import regularizers
import numpy as np
import random

random.seed(72)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
        
class roc_callback(keras.callbacks.Callback):
    def __init__(self,training_data, validation_data):
        
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
    
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):        
        y_pred_train = np.array(self.model.predict(self.x)).reshape(self.x.shape[0])
        judger_train = np.array(0.05*np.ones(self.x.shape[0]))
        delta_train = abs(y_pred_train-np.array(self.y))
        minus = np.sign(judger_train-delta_train)
        acc_class_train =(sum(minus)+self.x.shape[0])/(2*self.x.shape[0]) 
        
        y_pred_val = np.array(self.model.predict(self.x_val)).reshape(self.x_val.shape[0])
        judger_val = np.array(0.05*np.ones(self.x_val.shape[0]))
        delta_val = abs(y_pred_val-np.array(self.y_val))
        minus = np.sign(judger_val-delta_val)
        acc_class_val =(sum(minus)+self.x_val.shape[0])/(2*self.x_val.shape[0]) 
        
        #print(y_pred_val)
        #print(judger_val)
        ##print(delta_val)
        #print(minus)
        print ("acc_train %.4f"%(acc_class_train),"acc_val %.4f"%(acc_class_val))
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

train = pd.read_csv('training_set_rel3.tsv', sep='\t', header=0)
essay_count = numpy.size(train,0)

#valid = pd.read_csv('valid_set.tsv', sep='\t', header=0)

punc = '[:;,\"]'

sentences = []
for i in range(1783):
    essay = train["essay"][i]
    essay = re.sub(punc, '', essay)
    essay = re.sub('\.',' .',essay)
    essay = re.sub('!',' .',essay)
    essay = re.sub('\?',' .',essay)
    essay = essay.split(" ")
    for j in range(numpy.size(essay)):
        if (essay[j]!=''):
            if (essay[j][0]=='@'):
                essay[j] = essay[j][0:-1]
    #Deal With @PERSON1 -> @PERSON
    #Deal with punctuation
    
    sentences.append(essay)
    
number = np.arange(0,1783,1)
random.shuffle(number)
slice_train = number[0:1400]
slice_test = number[1400:1783]
    
tokenizer = Tokenizer(num_words=1000000)
tokenizer.fit_on_texts(train["essay"][0:1783])
sequences = tokenizer.texts_to_sequences(train["essay"][0:1783])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#data_x = pad_sequences(sequences, maxlen=600)


word_vector_dim = 100
model_word = Word2Vec(sentences, size=word_vector_dim, window=5, min_count=1, workers=1)

#generate input by word2vec
data_x = np.zeros(shape=[1783,600,word_vector_dim])
for i in range(1783):
    #1783*600*word_vector_dim
    for j in range(len(sentences[i])):
        try:
            now_word = model_word.wv.__getitem__(sentences[i][j])
        except KeyError:
            now_word = np.zeros((1,word_vector_dim))
        data_x[i,(600-len(sentences[i])+j)] = now_word   

#generate input by sentence2vec
data_x_sentence = np.zeros(shape=[1783,100,word_vector_dim])
for i in range(1783):
    #1783*100*word_vector_dim
    pun_list = []
    leng = len(sentences[i])
    for j in range(leng):
        if (sentences[i][j]=="."):
            pun_list += [j+600-leng]
    if pun_list==[]:
        pun_list = [600]
    if pun_list[len(pun_list)-1]!=599:
        pun_list += [600]
    #print(len(pun_list))
    for j in range(len(pun_list)):
        if (j==0):
            now_sentence = data_x[i][(600-leng):pun_list[0]]
        else:
            now_sentence = data_x[i][pun_list[j-1]+1:pun_list[j]]
        try:
            now_sentence = np.max(now_sentence,axis=0)
            data_x_sentence[i][(100-len(pun_list)+j)] = now_sentence
        except ValueError:
            test=1
               
data_x=data_x_sentence

len_sentence=np.zeros(1783)
for i in range(1783):
    len_sentence[i]=len(sentences[i])
#data_x=np.concatenate(data_x,len_sentence)
        
x_train = data_x[slice_train]
x_test = data_x[slice_test]         

data_y=train["domain1_score"][0:1783]
min_data_y=min(train["domain1_score"][0:1783])
max_data_y=max(train["domain1_score"][0:1783])
min_judge=0.5/(max_data_y-min_data_y)
data_y=(data_y-min_data_y)/(max_data_y-min_data_y)

data_y=(train["domain1_score"][0:1783]-2)
y_train = data_y[slice_train]
y_test = data_y[slice_test]

y_train = keras.utils.to_categorical(y_train, num_classes=11)
y_test = keras.utils.to_categorical(y_test, num_classes=11)

model = Sequential()

model.add(LSTM(256, input_shape=[100,100]))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))

model.add(Dense(11, activation='softmax'))
#model.add(Dense(1, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = LossHistory()
print(model.summary())

#model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=256, nb_epoch=70, verbose=1,callbacks=[history,roc_callback(training_data=[x_train, y_train], validation_data=[x_test, y_test])])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=256, nb_epoch=70, verbose=1,callbacks=[history])

history.loss_plot('epoch')

y_predict=model.predict(x_train)
dimx=1400
y_predict=y_predict.reshape(dimx)
y_delta=y_predict-y_train
y_w=np.sign(np.ones(dimx)*min_judge-abs(y_delta))
rate1=(sum(y_w)+dimx)/dimx/2
y_predict=model.predict(x_test)
dimx=383
y_predict=y_predict.reshape(dimx)
y_delta=y_predict-y_test
y_w=np.sign(np.ones(dimx)*min_judge-abs(y_delta))
rate2=(sum(y_w)+dimx)/dimx/2
print(rate1,rate2)