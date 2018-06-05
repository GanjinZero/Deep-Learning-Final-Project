import codecs
import gensim
import pandas as pd
import numpy
from pandas import DataFrame

train = pd.read_csv('training_set_rel3.tsv', sep='\t', header=0)
essay_count = numpy.size(train,0)

for i in range(essay_count):
    essay = train["essay"][i]
    essay = essay.split(" ")
    #Need to deal with punctuation