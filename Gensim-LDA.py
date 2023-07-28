### Gensim-LDA ###

import numpy as np
import json
import glob

#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from gensim.corpora import Dictionary

#spacy
import spacy
from nltk.corpus import stopwords

#vis
import pyLDAvis
import pyLDAvis.gensim_models

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

with open("/mnt/c/TopicModelling/corpus/full_corpus.txt") as f:
    corpus = f.read()



list_of_sentence = sent_tokenize(corpus)
print(

list_of_sentence

)

list_of_simple_preprocess_data = []
for i in list_of_sentence:
    list_of_simple_preprocess_data.append(gensim.utils.simple_preprocess(i, deacc=True, min_len=3))

texts = list_of_simple_preprocess_data


bigram = gensim.models.Phrases(list_of_simple_preprocess_data)

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(
   corpus=corpus, id2word=dictionary, num_topics=10, random_state=100, 
   update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True
)

pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)

ldamodel.show_topics()