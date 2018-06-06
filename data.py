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
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences

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
    
    #Need to deal with punctuation
    sentences.append(essay)
    
tokenizer = Tokenizer(num_words=1000000)
tokenizer.fit_on_texts(train["essay"][0:1783])
sequences = tokenizer.texts_to_sequences(train["essay"][0:1783])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data_x = pad_sequences(sequences, maxlen=600)
x_train = data_x[0:1400]
x_test = data_x[1400:1783]

word_vector_dim = 100
model_word = Word2Vec(sentences, size=word_vector_dim, window=5, min_count=1, workers=1)

data_y=(train["domain1_score"][0:1783]-2)/10
y_train = data_y[0:1400]
y_test = data_y[1400:1783]
#x_train=train["essay"][0:1783]
    
model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1,output_dim=word_vector_dim))
model.add(LSTM(128,input_dim=word_vector_dim,activation='sigmoid'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print(type(x_train))

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, nb_epoch=10, verbose=1)
score = model.evaluate(x_test, y_test)
print(score)

y_predict=model.predict(x_test)
y_predict=y_predict.reshape(383)
y_delta=y_predict-y_test