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
from keras.layers import Input,LSTM, GRU, SimpleRNN
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import keras
import keras.models
from keras.layers.core import Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from tensorflow.python.client import device_lib as _device_lib
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras import regularizers
import numpy as np
import random
import csv

random.seed(72)

epoch = 100
roc_save_train = []
roc_save_val = []

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
        
        self.x,self.x_len = training_data[0]
        self.y = training_data[1]
        self.x_val,self.x_val_len = validation_data[0]
        self.y_val = validation_data[1]
        
    
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        global roc_save_train
        global roc_save_val
        y_pred_train = np.array(self.model.predict([self.x,self.x_len])).reshape(self.x.shape[0])
        judger_train = np.array(0.05*np.ones(self.x.shape[0]))
        delta_train = abs(y_pred_train-np.array(self.y))
        minus = np.sign(judger_train-delta_train)
        acc_class_train =(sum(minus)+self.x.shape[0])/(2*self.x.shape[0]) 
        
        y_pred_val = np.array(self.model.predict([self.x_val,self.x_val_len])).reshape(self.x_val.shape[0])
        judger_val = np.array(0.05*np.ones(self.x_val.shape[0]))
        delta_val = abs(y_pred_val-np.array(self.y_val))
        minus = np.sign(judger_val-delta_val)
        acc_class_val =(sum(minus)+self.x_val.shape[0])/(2*self.x_val.shape[0]) 

        print ("acc_train %.4f"%(acc_class_train),"acc_val %.4f"%(acc_class_val))
        roc_save_train = roc_save_train+[acc_class_train]
        roc_save_val = roc_save_val+[acc_class_val]
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

train = pd.read_csv('training_set_rel3.tsv', sep='\t', header=0)
essay_count = numpy.size(train,0)

max_length=870

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
    space_place = []
    all_place = []
    for j in range(numpy.size(essay)):
        all_place = all_place + [j]
        if (essay[j]!=''):
            if (essay[j][0]=='@'):
                essay[j] = essay[j][0:-1]
        if (essay[j]==''):
            space_place = space_place + [j]
    use_place = [item for item in all_place if item not in space_place]
    use = []
    for j in range(len(use_place)):
        use = use + [essay[use_place[j]]]
    #Deal With @PERSON1 -> @PERSON
    #Deal with punctuation
    sentences.append(use)
    
number = np.arange(0,1783,1)
random.shuffle(number)
slice_train = number[0:1400]
slice_test = number[1400:1783]
    
tokenizer = Tokenizer(num_words=1000000)
tokenizer.fit_on_texts(train["essay"][0:1783])
sequences = tokenizer.texts_to_sequences(train["essay"][0:1783])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#data_x = pad_sequences(sequences, maxlen=700)


word_vector_dim = 100
model_word = Word2Vec(sentences, size=word_vector_dim, window=5, min_count=1, workers=1)

#generate input by word2vec
data_x = np.zeros(shape=[1783,max_length,word_vector_dim])
for i in range(1783):
    #1783*700*word_vector_dim
    for j in range(len(sentences[i])):
        try:
            now_word = model_word.wv.__getitem__(sentences[i][j])
        except KeyError:
            now_word = np.zeros((1,word_vector_dim))
        data_x[i,(max_length-len(sentences[i])+j)] = now_word  
        
post=[]
csvfile = open('essay1.csv','r')
reader = csv.reader(csvfile)
for line in reader:
    if line !=[]:
        post.append(line)
csvfile.close()
for i in range(len(post)):
    for j in range(len(post[i])):
        post[i][j]=int(post[i][j])
                
max_post = 41
post_onehot = np.zeros(shape=[len(post),max_length,max_post+1])
for i in range(len(post)):
    for j in range(len(post[i])):
        k=post[i][j]
        try:
            post_onehot[i][j][k]=1
        except IndexError:
            print(i,j,k)
            
data_x=np.concatenate((data_x,post_onehot),axis=2)
        
#generate input by sentence2vec
data_x_sentence = np.zeros(shape=[1783,100,word_vector_dim+max_post+1])
for i in range(1783):
    #1783*100*word_vector_dim
    pun_list = []
    leng = len(sentences[i])
    for j in range(leng):
        if (sentences[i][j]=="."):
            pun_list += [j+max_length-leng]
    if pun_list==[]:
        pun_list = [max_length]
    if pun_list[len(pun_list)-1]!=max_length-1:
        pun_list += [max_length]
    #print(len(pun_list))
    for j in range(len(pun_list)):
        if (j==0):
            now_sentence = data_x[i][(max_length-leng):pun_list[0]]
        else:
            now_sentence = data_x[i][pun_list[j-1]+1:pun_list[j]]
        try:
            now_sentence = np.max(now_sentence,axis=0)
            data_x_sentence[i][(100-len(pun_list)+j)] = now_sentence
        except ValueError:
            tt=1
               
data_x=data_x_sentence

len_sentence=np.zeros(1783)
for i in range(1783):
    len_sentence[i]=len(sentences[i])

lenx_train = len_sentence[slice_train]
lenx_test = len_sentence[slice_test]        
x_train = data_x[slice_train]
x_test = data_x[slice_test]         

data_y=train["domain1_score"][0:1783]
min_data_y=min(train["domain1_score"][0:1783])
max_data_y=max(train["domain1_score"][0:1783])
data_y=(data_y-min_data_y)/(max_data_y-min_data_y)
y_train = data_y[slice_train]
y_test = data_y[slice_test]

#y_train = keras.utils.to_categorical(y_train, num_classes=11)
#y_test = keras.utils.to_categorical(y_test, num_classes=11)

input1 = Input(shape=(100,142,))
input2 = Input(shape=(1,))
lstm_out = LSTM(256, input_shape=[100,142], return_sequences=False)(input1)
y = keras.layers.concatenate([lstm_out, input2])
x = Dense(512, activation='relu')(y)
x = Dense(256, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = keras.models.Model(inputs=[input1, input2], outputs=output)
#model = keras.models.Model(inputs=[input1], outputs=output)
adam = keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer=adam , metrics=[])
history = LossHistory()
print(model.summary())


#model.fit([x_train,lenx_train], y_train, validation_data=([x_test,lenx_test], y_test), batch_size=256, nb_epoch=epoch, verbose=1,callbacks=[history])
model.fit([x_train,lenx_train], y_train, validation_data=([x_test,lenx_test], y_test), batch_size=256, nb_epoch=epoch, verbose=1,callbacks=[history,roc_callback(training_data=[[x_train,lenx_train], y_train], validation_data=[[x_test,lenx_test], y_test])])

iters = range(len(history.losses['epoch']))
ax1 = plt.figure().add_subplot(111)
ax1.plot(iters, history.losses['epoch'], 'g', label='train loss')
ax1.plot(iters, history.val_loss['epoch'], 'k', label='val loss')
ax1.grid(True)
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.legend(loc="upper right")
ax2 = ax1.twinx()
ax2.set_ylim([0, 1])
ax2.plot(iters, roc_save_train, 'r', label='train acc')
ax2.plot(iters, roc_save_val, 'b', label='val acc')
ax2.legend(loc="upper left")
plt.show()

#y_predict=model.predict([x_train,lenx_train])
#dimx=1400
#y_predict=y_predict.reshape(dimx)
#y_delta=y_predict-y_train
#y_w=np.sign(np.ones(dimx)*min_judge-abs(y_delta))
#rate1=(sum(y_w)+dimx)/dimx/2
#y_predict=model.predict([x_test,lenx_test])
#dimx=383
#y_predict=y_predict.reshape(dimx)
#y_delta=y_predict-y_test
#y_w=np.sign(np.ones(dimx)*min_judge-abs(y_delta))
#rate2=(sum(y_w)+dimx)/dimx/2
#print(rate1,rate2)