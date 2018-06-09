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
import random
import numpy as np

#import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
        #plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            #plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

train = pd.read_csv('training_set_rel3.tsv', sep='\t', header=0)
essay_count = numpy.size(train,0)

#valid = pd.read_csv('valid_set.tsv', sep='\t', header=0)

punc = '[:;,!\"]'

sentences = []
for i in range(1783):
    essay = train["essay"][i]
    essay = re.sub(punc, '', essay)
    essay = re.sub('\.',' .',essay)
    essay = essay.split(" ")
    for j in range(numpy.size(essay)):
        if (essay[j]!=''):
            if (essay[j][0]=='@'):
                essay[j] = essay[j][0:-1]
    #Deal With @PERSON1 -> @PERSON
    #Deal with punctuation
    
    sentences.append(essay)
    
tokenizer = Tokenizer(num_words=1000000)
tokenizer.fit_on_texts(train["essay"][0:1783])
sequences = tokenizer.texts_to_sequences(train["essay"][0:1783])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data_x = pad_sequences(sequences, maxlen=600)


word_vector_dim = 300
#model_word = Word2Vec(sentences, size=word_vector_dim, window=5, min_count=1, workers=1)

number = np.arange(0,1783,1)
random.shuffle(number)
slice_train = number[0:1400]
slice_test = number[1400:1783]

x_train = data_x[slice_train]
x_test = data_x[slice_test]

data_y=(train["domain1_score"][0:1783]-2)

y_train = data_y[slice_train]
y_train = keras.utils.to_categorical(y_train, num_classes=11)
y_test = data_y[slice_test]
y_test = keras.utils.to_categorical(y_test, num_classes=11)
#x_train=train["essay"][0:1783]
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))    
model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1,output_dim=word_vector_dim))
model.add(GRU(256))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Dense(11, activation='softmax',activity_regularizer=regularizers.l2(0.01)))
#model.add(Dense(1, activation='sigmoid',activity_regularizer=regularizers.l1(0.01)))

#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = LossHistory()
print(model.summary())
#print(type(x_train))
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=256, nb_epoch=30, verbose=1,callbacks=[history])
#score = model.evaluate(x_test, y_test)
#print(score)

#y_predict=model.predict(x_test)
#y_predict=y_predict.reshape(383)
#y_delta=y_predict-y_test
history.loss_plot('epoch')